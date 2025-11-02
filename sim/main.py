#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import enum
import hashlib
import heapq
import io
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import zstandard


LOGGER = logging.getLogger("cfcache.simulator")


@dataclass(frozen=True)
class Request:
    """A single cache access pulled from a trace."""

    timestamp: int
    key: str
    zone: str
    size: int
    ttl: int
    stale_time: int
    method: str
    mime: str

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "Request":
        try:
            timestamp = int(float(row["timestamp"]))
            size = int(row["size"])
            ttl = int(row.get("expiry_time", 0))
            stale = int(row.get("stale_time", 0))
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - logged for visibility
            raise ValueError(f"Invalid row: {row!r}") from exc

        key = row.get("key") or ""
        zone = row.get("zone") or ""
        method = row.get("method") or ""
        mime = row.get("mime") or ""

        return cls(
            timestamp=timestamp,
            key=key,
            zone=zone,
            size=size,
            ttl=ttl,
            stale_time=stale,
            method=method,
            mime=mime,
        )

    @property
    def expiry_at(self) -> Optional[int]:
        if self.ttl <= 0:
            return None
        return self.timestamp + self.ttl


@dataclass
class CacheEntry:
    key: str
    size: int
    expiry_at: Optional[int]
    zone: str
    mime: str
    inserted_at: int
    last_access_at: int

    def is_expired(self, now: int) -> bool:
        return self.expiry_at is not None and now >= self.expiry_at


class MissReason(str, enum.Enum):
    FIRST = "first"
    EVICTED = "evicted"
    EXPIRED = "expired"
    OTHER = "other"


class RemovalReason(enum.Enum):
    EVICTED = enum.auto()
    EXPIRED = enum.auto()
    MANUAL = enum.auto()


@dataclass
class CacheStats:
    requests: int = 0
    hits: int = 0
    miss_first: int = 0
    miss_evicted: int = 0
    miss_expired: int = 0
    miss_other: int = 0
    evictions: int = 0
    expirations: int = 0
    skipped_stores: int = 0
    skipped_store_bytes: int = 0
    bytes_written: int = 0
    bytes_evicted: int = 0
    max_bytes_used: int = 0

    def record_miss(self, reason: MissReason) -> None:
        if reason == MissReason.FIRST:
            self.miss_first += 1
        elif reason == MissReason.EVICTED:
            self.miss_evicted += 1
        elif reason == MissReason.EXPIRED:
            self.miss_expired += 1
        else:
            self.miss_other += 1

    def update_usage(self, current_bytes: int) -> None:
        if current_bytes > self.max_bytes_used:
            self.max_bytes_used = current_bytes

    @property
    def misses(self) -> int:
        return self.miss_first + self.miss_evicted + self.miss_expired + self.miss_other

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "hit_rate": self.hit_rate,
            "misses": {
                "first": self.miss_first,
                "evicted": self.miss_evicted,
                "expired": self.miss_expired,
                "other": self.miss_other,
            },
            "evictions": self.evictions,
            "expirations": self.expirations,
            "skipped_stores": self.skipped_stores,
            "skipped_store_bytes": self.skipped_store_bytes,
            "bytes_written": self.bytes_written,
            "bytes_evicted": self.bytes_evicted,
            "max_bytes_used": self.max_bytes_used,
        }


class CachePolicy:
    name: str

    def on_insert(self, key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_access(self, key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_remove(self, key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def choose_victim(self) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError


class LRUPolicy(CachePolicy):
    name = "lru"

    def __init__(self) -> None:
        self._order: "OrderedDict[str, None]" = OrderedDict()

    def on_insert(self, key: str) -> None:
        self._order[key] = None

    def on_access(self, key: str) -> None:
        if key in self._order:
            self._order.move_to_end(key)

    def on_remove(self, key: str) -> None:
        self._order.pop(key, None)

    def choose_victim(self) -> Optional[str]:
        try:
            return next(iter(self._order))
        except StopIteration:
            return None


class FIFOPolicy(CachePolicy):
    name = "fifo"

    def __init__(self) -> None:
        self._order: "OrderedDict[str, None]" = OrderedDict()

    def on_insert(self, key: str) -> None:
        self._order[key] = None

    def on_access(self, key: str) -> None:
        # FIFO does not update ordering on access.
        pass

    def on_remove(self, key: str) -> None:
        self._order.pop(key, None)

    def choose_victim(self) -> Optional[str]:
        try:
            return next(iter(self._order))
        except StopIteration:
            return None


POLICY_REGISTRY = {
    LRUPolicy.name: LRUPolicy,
    FIFOPolicy.name: FIFOPolicy,
}


class CacheSimulator:
    def __init__(self, capacity: int, policy: CachePolicy, auto_expire: bool = True) -> None:
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")
        self.capacity = capacity
        self.policy = policy
        self.auto_expire = auto_expire
        self.entries: Dict[str, CacheEntry] = {}
        self.expiry_heap: List[Tuple[int, str]] = []
        self.stats = CacheStats()
        self.seen_keys: set[str] = set()
        self.evicted_keys: set[str] = set()
        self.expired_keys: set[str] = set()
        self.current_bytes = 0

    def purge_expired(self, now: int) -> None:
        if not self.auto_expire:
            return
        while self.expiry_heap and self.expiry_heap[0][0] <= now:
            expiry, key = heapq.heappop(self.expiry_heap)
            entry = self.entries.get(key)
            if entry is None:
                continue
            if entry.expiry_at is None or entry.expiry_at > now:
                continue
            LOGGER.debug("Auto-expiring %s at %s (expiry=%s)",
                         key, now, expiry)
            self._remove_entry(key, RemovalReason.EXPIRED)

    def process_request(self, request: Request) -> None:
        self.stats.requests += 1
        if self.auto_expire:
            self.purge_expired(request.timestamp)

        entry = self.entries.get(request.key)

        if entry and not entry.is_expired(request.timestamp):
            entry.last_access_at = request.timestamp
            self.policy.on_access(request.key)
            self.stats.hits += 1
            self.seen_keys.add(request.key)
            return

        miss_reason = self._classify_miss(request, entry)
        self.stats.record_miss(miss_reason)
        self.seen_keys.add(request.key)

        if entry and entry.is_expired(request.timestamp):
            self._remove_entry(request.key, RemovalReason.EXPIRED)

        if request.size > self.capacity:
            LOGGER.debug("Skipping cache for %s; item size %s > capacity %s",
                         request.key, request.size, self.capacity)
            self.stats.skipped_stores += 1
            self.stats.skipped_store_bytes += request.size
            return

        expiry_at = request.expiry_at
        if expiry_at is not None and expiry_at <= request.timestamp:
            LOGGER.debug(
                "Skipping cache for %s; already expired at %s", request.key, expiry_at)
            self.stats.skipped_stores += 1
            self.stats.skipped_store_bytes += request.size
            self.expired_keys.add(request.key)
            return

        if not self._ensure_capacity(request.size):
            LOGGER.debug("Unable to free enough space for %s", request.key)
            self.stats.skipped_stores += 1
            self.stats.skipped_store_bytes += request.size
            return

        self._add_entry(request)

    def _classify_miss(self, request: Request, entry: Optional[CacheEntry]) -> MissReason:
        if entry and entry.is_expired(request.timestamp):
            return MissReason.EXPIRED
        if request.key not in self.seen_keys:
            return MissReason.FIRST
        if request.key in self.expired_keys:
            return MissReason.EXPIRED
        if request.key in self.evicted_keys:
            return MissReason.EVICTED
        return MissReason.OTHER

    def _ensure_capacity(self, space_needed: int) -> bool:
        if space_needed > self.capacity:
            return False

        while self.current_bytes + space_needed > self.capacity:
            victim = self.policy.choose_victim()
            if victim is None:
                break
            LOGGER.debug("Evicting %s to reclaim space", victim)
            self._remove_entry(victim, RemovalReason.EVICTED)

        return self.current_bytes + space_needed <= self.capacity

    def _add_entry(self, request: Request) -> None:
        expiry_at = request.expiry_at
        entry = CacheEntry(
            key=request.key,
            size=request.size,
            expiry_at=expiry_at,
            zone=request.zone,
            mime=request.mime,
            inserted_at=request.timestamp,
            last_access_at=request.timestamp,
        )
        self.entries[request.key] = entry
        self.policy.on_insert(request.key)
        self.current_bytes += entry.size
        self.stats.bytes_written += entry.size
        self.stats.update_usage(self.current_bytes)
        if expiry_at is not None:
            heapq.heappush(self.expiry_heap, (expiry_at, request.key))
        self.evicted_keys.discard(request.key)
        self.expired_keys.discard(request.key)

    def _remove_entry(self, key: str, reason: RemovalReason) -> Optional[CacheEntry]:
        entry = self.entries.pop(key, None)
        self.policy.on_remove(key)
        if not entry:
            return None

        self.current_bytes -= entry.size
        if self.current_bytes < 0:
            self.current_bytes = 0
        self.stats.update_usage(self.current_bytes)

        if reason == RemovalReason.EVICTED:
            self.stats.evictions += 1
            self.stats.bytes_evicted += entry.size
            self.evicted_keys.add(key)
            self.expired_keys.discard(key)
        elif reason == RemovalReason.EXPIRED:
            self.stats.expirations += 1
            self.expired_keys.add(key)
            self.evicted_keys.discard(key)
        else:
            self.evicted_keys.discard(key)
            self.expired_keys.discard(key)

        return entry

    def finalize(self) -> None:
        # Update usage stats with final occupancy.
        self.stats.update_usage(self.current_bytes)

    def snapshot(self) -> Dict[str, object]:
        self.finalize()
        return {
            "policy": getattr(self.policy, "name", type(self.policy).__name__),
            "auto_expire": self.auto_expire,
            "capacity_bytes": self.capacity,
            "capacity_human": format_bytes(self.capacity),
            "current_bytes": self.current_bytes,
            "current_items": len(self.entries),
            "unique_keys": len(self.seen_keys),
            "stats": self.stats.as_dict(),
        }


def create_policy(name: str) -> CachePolicy:
    policy_cls = POLICY_REGISTRY.get(name.lower())
    if not policy_cls:
        raise ValueError(
            f"Unknown policy '{name}'. Available policies: {', '.join(sorted(POLICY_REGISTRY))}")
    return policy_cls()


def parse_size(size_str: str) -> int:
    normalized = size_str.strip().lower()
    if not normalized:
        raise ValueError("cache size string cannot be empty")

    suffixes = {
        "k": 1_000,
        "kb": 1_000,
        "ki": 1 << 10,
        "kib": 1 << 10,
        "m": 1_000_000,
        "mb": 1_000_000,
        "mi": 1 << 20,
        "mib": 1 << 20,
        "g": 1_000_000_000,
        "gb": 1_000_000_000,
        "gi": 1 << 30,
        "gib": 1 << 30,
        "t": 1_000_000_000_000,
        "tb": 1_000_000_000_000,
        "ti": 1 << 40,
        "tib": 1 << 40,
    }

    for suffix, multiplier in suffixes.items():
        if normalized.endswith(suffix):
            numeric = normalized[: -len(suffix)]
            value = float(numeric)
            return int(value * multiplier)

    return int(float(normalized))


def format_bytes(num_bytes: int) -> str:
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    for unit in units:
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} EiB"


def discover_trace_paths(inputs: Sequence[str], default_dir: Path) -> List[Path]:
    paths: List[Path] = []

    if not inputs:
        inputs = [str(default_dir)]

    for entry in inputs:
        path = Path(entry).expanduser().resolve()
        if path.is_dir():
            for candidate in sorted(path.glob("**/*.csv")):
                paths.append(candidate)
            for candidate in sorted(path.glob("**/*.csv.zst")):
                paths.append(candidate)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Trace path {path} does not exist")

    if not paths:
        raise FileNotFoundError("No trace files found")

    return paths


def _key_in_sample(key: str, ratio: int) -> bool:
    if ratio <= 1:
        return True
    # Use a stable hash so sampling decisions remain deterministic across runs.
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return (value % ratio) == 0


class ProgressReporter:
    def __init__(self, enabled: bool, interval: int = 10_000) -> None:
        self.enabled = enabled
        self.interval = max(1, interval)
        self._last_report = 0
        self._start = perf_counter()

    def tick(self, count: int) -> None:
        if not self.enabled:
            return
        if count - self._last_report < self.interval:
            return
        self._emit(count)

    def done(self, count: int) -> None:
        if not self.enabled:
            return
        self._emit(count, final=True)

    def _emit(self, count: int, final: bool = False) -> None:
        now = perf_counter()
        elapsed = max(now - self._start, 1e-9)
        rate = count / elapsed
        label = "total" if final else "processed"
        print(f"[progress] {label} {count:,} requests ({rate:,.0f}/s)", file=sys.stderr)
        self._last_report = count


def iter_requests(
        paths: Iterable[Path],
        limit: Optional[int] = None,
        key_sample_ratio: int = 1,
        progress: Optional[ProgressReporter] = None,
) -> Iterator[Request]:
    count = 0
    completed = False
    for trace_path in paths:
        LOGGER.info("Reading trace %s", trace_path)
        for request in read_trace(trace_path):
            if not _key_in_sample(request.key, key_sample_ratio):
                continue
            yield request
            count += 1
            if progress:
                progress.tick(count)
            if limit is not None and count >= limit:
                LOGGER.info("Reached max request limit of %s", limit)
                if progress:
                    progress.done(count)
                completed = True
                return
    if progress and not completed:
        progress.done(count)


def read_trace(path: Path) -> Iterator[Request]:
    if path.suffix == ".zst":
        with path.open("rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                yield from _read_csv_stream(text_stream, path)
    else:
        with path.open("r", encoding="utf-8") as fh:
            yield from _read_csv_stream(fh, path)


def _read_csv_stream(stream: io.TextIOBase, path: Path) -> Iterator[Request]:
    reader = csv.DictReader(stream)
    try:
        for row in reader:
            if not row:
                continue
            try:
                yield Request.from_row(row)
            except ValueError as exc:
                LOGGER.warning("Skipping malformed row in %s: %s", path, exc)
    except Exception as exc:
        LOGGER.error("Error reading CSV from %s: %s", path, exc)


def run_simulation(
        trace_paths: Sequence[Path],
        policies: Sequence[str],
        cache_sizes: Sequence[int],
        auto_expire: bool,
        limit: Optional[int] = None,
        key_sample_ratio: int = 1,
        progress: Optional[ProgressReporter] = None,
) -> List[Dict[str, Any]]:
    simulators: List[Tuple[str, int, CacheSimulator]] = []
    for policy_name in policies:
        for capacity in cache_sizes:
            simulator = CacheSimulator(capacity=capacity, policy=create_policy(
                policy_name), auto_expire=auto_expire)
            simulators.append((policy_name, capacity, simulator))

    for request in iter_requests(trace_paths, limit=limit, key_sample_ratio=key_sample_ratio, progress=progress):
        for _, _, simulator in simulators:
            simulator.process_request(request)

    results: List[Dict[str, object]] = []
    for policy_name, capacity, simulator in simulators:
        snapshot = simulator.snapshot()
        snapshot["policy"] = policy_name
        snapshot["capacity_bytes"] = capacity
        snapshot["capacity_human"] = format_bytes(capacity)
        results.append(snapshot)

    return results


def configure_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache simulator for CDN traces")
    parser.add_argument(
        "--trace", nargs="*", help="Trace files or directories to consume. Defaults to ./traces")
    parser.add_argument("--policy", nargs="+",
                        default=["lru"], help="Cache policies to evaluate (e.g. lru fifo)")
    parser.add_argument("--cache-size", dest="cache_sizes", nargs="+",
                        required=True, help="Cache sizes (e.g. 256MiB 1GiB)")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Limit the number of requests processed")
    parser.add_argument("--auto-expire", action=argparse.BooleanOptionalAction,
                        default=True, help="Automatically evict entries when they expire")
    parser.add_argument("--verbose", "-v", action="count",
                        default=0, help="Increase logging verbosity")
    parser.add_argument("--list-policies", action="store_true",
                        help="List supported cache policies and exit")
    parser.add_argument("--key-sample", type=int, default=1,
                        help="Sample every Nth key using a deterministic hash (e.g. 3 keeps roughly one third of keys). Also adjusts cache size automatically.")
    parser.add_argument("--progress", action="store_true",
                        help="Display a simple progress indicator while processing requests")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_policies:
        print("Available policies:")
        for name in sorted(POLICY_REGISTRY):
            print(f"  - {name}")
        return 0

    configure_logging(args.verbose)

    repo_root = Path(__file__).resolve().parent.parent
    default_traces = repo_root / "traces"

    try:
        cache_sizes = [parse_size(size) / args.key_sample for size in args.cache_sizes]
    except ValueError as exc:
        LOGGER.error("Failed to parse cache sizes: %s", exc)
        return 2

    if args.key_sample < 1:
        LOGGER.error("Key sample ratio must be a positive integer")
        return 2

    try:
        trace_paths = discover_trace_paths(args.trace or [], default_traces)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 2

    try:
        results = run_simulation(
            trace_paths=trace_paths,
            policies=args.policy,
            cache_sizes=cache_sizes,
            auto_expire=args.auto_expire,
            limit=args.max_requests,
            key_sample_ratio=args.key_sample,
            progress=ProgressReporter(enabled=args.progress),
        )
    except KeyboardInterrupt:  # pragma: no cover - handled for CLI use
        LOGGER.warning("Simulation interrupted")
        return 130

    for result in results:
        stats = cast(Dict[str, Any], result.pop("stats"))
        print("=" * 80)
        print(
            f"policy={result['policy']} cache={result['capacity_human']} auto_expire={result['auto_expire']}")
        print(
            f"requests={stats['requests']} hits={stats['hits']} hit_rate={stats['hit_rate']*100:.2f}%")
        print("misses:")
        for miss_label, miss_value in stats["misses"].items():
            print(f"  {miss_label:>7}: {miss_value}")
        print(
            f"evictions={stats['evictions']} expirations={stats['expirations']}")
        print(
            f"current_bytes={result['current_bytes']} current_items={result['current_items']} unique_keys={result['unique_keys']}")
        print(
            f"skipped_stores={stats['skipped_stores']} skipped_bytes={stats['skipped_store_bytes']}")
        print(f"max_bytes_used={stats['max_bytes_used']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
