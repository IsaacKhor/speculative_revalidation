#!/usr/bin/env python3
"""Download and format the public traces consumed by ``reproduce.py``.

Cloudflare and Meta publish Zstandard-compressed CSV files in (or very close
to) the required layout.  Wikimedia publishes daily gzip-compressed TSV files;
those are projected to the simulator's three input columns and assembled as
concatenated Zstandard frames without materializing the much larger plain CSV.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
FORMAT_VERSION = 1
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@dataclass(frozen=True)
class DirectTrace:
    dataset: str
    url: str
    relative_output: str
    expected_header: bytes


CF_BASE = "https://objects.research.cloudflare.com/@ikhor/cdn-traces"
CF_HEADER = b"timestamp,key,zone,size,expiry_time,stale_time,method,mime"
CF_FILES = (
    "106m105.csv.zst", "106m106.csv.zst", "243m12.csv.zst", "243m13.csv.zst",
    "411m264.csv.zst", "411m325.csv.zst", "472m378.csv.zst", "472m379.csv.zst",
)

META_BASE = "https://s3.amazonaws.com/cache-datasets/cache_dataset_txt/2023_metaCDN"
META_HEADER = (
    b"timestamp,cacheKey,OpType,objectSize,responseSize,responseHeaderSize,"
    b"rangeStart,rangeEnd,TTL,SamplingRate,cache_hit,item_value,RequestHandler,"
    b"cdn_content_type_id,vip_type"
)
META_FILES = (
    ("reag0c01_20230315_20230322_0.2000.csv.zst", "reag.csv.zst"),
    # The paper's local trace name is ``rhna``; Meta's published object is ``rnha``.
    ("rnha0c01_20230315_20230322_0.8000.csv.zst", "rhna.csv.zst"),
    ("rprn0c01_20230315_20230322_0.2000.csv.zst", "rprn.csv.zst"),
)

DIRECT_TRACES = tuple(
    DirectTrace(
        "cf", f"{CF_BASE}/{name}", f"cdn_cf25_csv/{name}", CF_HEADER
    )
    for name in CF_FILES
) + tuple(
    DirectTrace(
        "fb", f"{META_BASE}/{remote}", f"cdn_fb23_csv/{local}", META_HEADER
    )
    for remote, local in META_FILES
)


@dataclass(frozen=True)
class WikimediaTrace:
    name: str
    directory: str
    days: tuple[int, ...]
    source_header: bytes
    fields: str

    @property
    def output_name(self) -> str:
        return f"{self.name}-all.csv.zst"


WM_BASE = "https://analytics.wikimedia.org/published/datasets/caching/2019"
WM_EXISTING_SIZES = {
    "t-all.csv.zst": 1_875_341_727,
    "u-all.csv.zst": 23_087_105_338,
}
WM_TRACES = (
    # The committed experiment's wm_t input spans timestamps 86400..1814399
    # and 197,819,321 requests, corresponding to published text days 01--20.
    WikimediaTrace(
        "t", "text", tuple(range(1, 21)),
        b"relative_unix\thashed_host_path_query\tresponse_size\ttime_firstbyte",
        "1-3",
    ),
    WikimediaTrace(
        "u", "upload", tuple(range(21)),
        b"relative_unix\thashed_path_query\timage_type\tresponse_size\ttime_firstbyte",
        "1,2,4",
    ),
)


class Downloader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.trace_root = args.trace_root.expanduser().resolve()
        if self.trace_root == Path(self.trace_root.anchor):
            raise ValueError("refusing to use a filesystem root as --trace-root")
        self.stage = self.trace_root / ".reproduction-downloads"

    def announce(self, *parts: object) -> None:
        print(" ".join(str(part) for part in parts), flush=True)

    def command(self, command: Sequence[object]) -> None:
        self.announce("$", shlex.join(str(part) for part in command))

    def require_tools(self, selected: set[str]) -> None:
        required = ["curl", "zstd"]
        if "wm" in selected:
            required.extend(("gzip", "tail", "cut"))
        missing = [
            name for name in required if shutil.which(name) is None
        ]
        if missing:
            raise RuntimeError("missing required command(s): " + ", ".join(missing))

    def marker_path(self, output: Path) -> Path:
        safe = str(output.relative_to(self.trace_root)).replace("/", "__")
        return self.stage / "complete" / f"{safe}.json"

    def marker_matches(self, output: Path, sources: Sequence[str]) -> bool:
        try:
            record = json.loads(self.marker_path(output).read_text(encoding="utf-8"))
            stat = output.stat()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        return (
            record.get("format_version") == FORMAT_VERSION
            and record.get("sources") == list(sources)
            and record.get("bytes") == stat.st_size
            and stat.st_size > 0
        )

    def mark_complete(self, output: Path, sources: Sequence[str]) -> None:
        marker = self.marker_path(output)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "format_version": FORMAT_VERSION,
            "sources": list(sources),
            "output": str(output),
            "bytes": output.stat().st_size,
        }, indent=2) + "\n", encoding="utf-8")

    def zstd_test(self, path: Path) -> None:
        subprocess.run(("zstd", "-tq", path), check=True)

    def zstd_header(self, path: Path) -> bytes:
        process = subprocess.Popen(
            ("zstd", "-dcq", path), stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        try:
            return process.stdout.readline().rstrip(b"\r\n")
        finally:
            process.stdout.close()
            process.terminate()
            process.wait()

    def download(self, url: str, destination: Path, compression: str) -> None:
        part = destination.with_name(destination.name + ".part")
        command = (
            "curl", "--fail", "--location", "--retry", "10",
            "--retry-all-errors", "--continue-at", "-", "--output", part, url,
        )
        self.command(command)
        if self.args.dry_run:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            # A resumed request for an already complete .part file can receive
            # HTTP 416. Accept it only if the complete-file test succeeds.
            if not self.compression_valid(part, compression):
                if completed.returncode != 33:  # CURLE_RANGE_ERROR
                    completed.check_returncode()
                # Some publishers do not honor Range requests. Restart an
                # incomplete object once when continuation is unavailable.
                part.unlink(missing_ok=True)
                restart = tuple(item for item in command if item not in ("--continue-at", "-"))
                self.command(restart)
                subprocess.run(restart, check=True)
        if not self.compression_valid(part, compression):
            raise RuntimeError(f"downloaded file failed {compression} validation: {part}")
        os.replace(part, destination)

    def compression_valid(self, path: Path, compression: str) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        command = ("zstd", "-tq", path) if compression == "zstd" else ("gzip", "-t", path)
        return subprocess.run(command, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0

    def direct_trace_complete(self, spec: DirectTrace, output: Path) -> bool:
        if self.args.force or not output.is_file():
            return False
        if self.marker_matches(output, (spec.url,)):
            return True
        # Schema plus frame magic is stable across harmless publisher-side
        # recompression, unlike compressed byte size or multipart ETags.
        with output.open("rb") as stream:
            magic = stream.read(4)
        if magic != ZSTD_MAGIC or self.zstd_header(output) != spec.expected_header:
            return False
        if self.args.verify_existing:
            self.zstd_test(output)
        self.mark_complete(output, (spec.url,))
        return True

    def fetch_direct(self, spec: DirectTrace) -> None:
        output = self.trace_root / spec.relative_output
        if self.args.dry_run:
            self.announce(f"prepare {spec.dataset}: {spec.url} -> {output}")
            self.download(spec.url, output, "zstd")
            return
        if self.direct_trace_complete(spec, output):
            self.announce("already complete", output)
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = self.stage / "direct" / Path(spec.relative_output).name
        self.download(spec.url, staged, "zstd")
        if self.zstd_header(staged) != spec.expected_header:
            raise RuntimeError(f"unexpected CSV header in {spec.url}")
        os.replace(staged, output)
        self.mark_complete(output, (spec.url,))
        self.announce("completed", output)

    def wm_url(self, spec: WikimediaTrace, day: int) -> str:
        return f"{WM_BASE}/{spec.directory}/cache-{spec.name}-{day:02d}.gz"

    def format_wikimedia_day(
        self, raw: Path, fragment: Path, spec: WikimediaTrace,
    ) -> None:
        with gzip.open(raw, "rb") as stream:
            header = stream.readline().rstrip(b"\r\n")
        if header != spec.source_header:
            raise RuntimeError(f"unexpected Wikimedia header in {raw}: {header!r}")

        part = fragment.with_name(fragment.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        commands = (
            ("gzip", "-dc", raw),
            ("tail", "-n", "+2"),
            ("cut", "-f", spec.fields, "--output-delimiter=,"),
            ("zstd", "-T0", "-q", "-c"),
        )
        self.command((*commands[0], "|", *commands[1], "|", *commands[2], "|",
                      *commands[3], ">", part))
        with part.open("wb") as output:
            first = subprocess.Popen(commands[0], stdout=subprocess.PIPE)
            assert first.stdout is not None
            second = subprocess.Popen(commands[1], stdin=first.stdout, stdout=subprocess.PIPE)
            first.stdout.close()
            assert second.stdout is not None
            third = subprocess.Popen(commands[2], stdin=second.stdout, stdout=subprocess.PIPE)
            second.stdout.close()
            assert third.stdout is not None
            fourth = subprocess.Popen(commands[3], stdin=third.stdout, stdout=output)
            third.stdout.close()
            statuses = (fourth.wait(), third.wait(), second.wait(), first.wait())
        if any(statuses):
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Wikimedia conversion failed for {raw}: {statuses}")
        self.zstd_test(part)
        os.replace(part, fragment)

    def assemble_wikimedia(
        self, output: Path, fragments: Sequence[Path], sources: Sequence[str],
    ) -> None:
        part = output.with_name(output.name + ".part")
        output.parent.mkdir(parents=True, exist_ok=True)
        with part.open("wb") as destination:
            subprocess.run(
                ("zstd", "-q", "-c"), input=b"timestamp,key,size\n",
                stdout=destination, check=True,
            )
            for fragment in fragments:
                with fragment.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        self.zstd_test(part)
        if self.zstd_header(part) != b"timestamp,key,size":
            part.unlink(missing_ok=True)
            raise RuntimeError(f"failed to assemble {output}")
        os.replace(part, output)
        self.mark_complete(output, sources)

    def fetch_wikimedia(self, spec: WikimediaTrace) -> None:
        output = self.trace_root / "cdn_wm19_csv" / spec.output_name
        sources = tuple(self.wm_url(spec, day) for day in spec.days)
        if self.args.dry_run:
            self.announce(
                f"prepare wm_{spec.name}: {len(spec.days)} daily TSV archives -> {output}"
            )
            for day, url in zip(spec.days, sources, strict=True):
                raw = self.stage / "wikimedia" / f"cache-{spec.name}-{day:02d}.gz"
                self.download(url, raw, "gzip")
            return
        if not self.args.force and self.marker_matches(output, sources):
            self.announce("already complete", output)
            return
        if not self.args.force and output.is_file():
            with output.open("rb") as stream:
                magic = stream.read(4)
            known_size = WM_EXISTING_SIZES.get(output.name)
            if (
                output.stat().st_size == known_size
                and magic == ZSTD_MAGIC
                and self.zstd_header(output) == b"timestamp,key,size"
            ):
                if self.args.verify_existing:
                    self.zstd_test(output)
                self.mark_complete(output, sources)
                self.announce("already complete", output)
                return
        self.announce(
            f"prepare wm_{spec.name}: {len(spec.days)} daily TSV archives -> {output}"
        )
        fragments = []
        for day, url in zip(spec.days, sources, strict=True):
            stem = f"cache-{spec.name}-{day:02d}"
            raw = self.stage / "wikimedia" / f"{stem}.gz"
            fragment = self.stage / "wikimedia" / f"{stem}.csv.zst"
            fragments.append(fragment)
            if not self.compression_valid(fragment, "zstd"):
                if not self.compression_valid(raw, "gzip"):
                    self.download(url, raw, "gzip")
                self.format_wikimedia_day(raw, fragment, spec)
            else:
                self.announce("reuse formatted day", fragment)
            if not self.args.keep_downloads:
                raw.unlink(missing_ok=True)

        self.assemble_wikimedia(output, fragments, sources)
        if not self.args.keep_downloads:
            for fragment in fragments:
                fragment.unlink(missing_ok=True)
        self.announce("completed", output)

    def run(self) -> None:
        selected = set(self.args.dataset or ("cf", "fb", "wm"))
        if not self.args.dry_run:
            self.require_tools(selected)
        for spec in DIRECT_TRACES:
            if spec.dataset in selected:
                self.fetch_direct(spec)
        if "wm" in selected:
            for spec in WM_TRACES:
                self.fetch_wikimedia(spec)
        self.announce("trace preparation complete for", ", ".join(sorted(selected)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root", type=Path, default=REPO_ROOT / "traces",
        help="destination trace root (default: repository traces/)",
    )
    parser.add_argument(
        "--dataset", choices=("cf", "fb", "wm"), action="append",
        help="prepare only this dataset; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--keep-downloads", action="store_true",
        help="retain Wikimedia gzip archives and formatted daily fragments",
    )
    parser.add_argument(
        "--verify-existing", action="store_true",
        help="fully decompress-test existing direct-download traces",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace final trace files even when completion metadata matches",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print URLs, destinations, and commands without downloading or writing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        Downloader(parse_args(argv)).run()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
