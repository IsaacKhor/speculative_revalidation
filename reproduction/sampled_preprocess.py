#!/usr/bin/env python3
"""Preprocess a full CSV trace while retaining one spatial key sample.

The simulator normally filters keys only after a complete binary trace has
been materialized.  For validation runs this wrapper applies the identical
``key % N == 0`` predicate before packing, while retaining every request for
each selected key.  The reverse pass therefore still computes exact
next-access timestamps for the sampled keys.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))
# The repository preprocessor uses the third-party ``regex`` package only for
# basic ``compile`` and ``match`` calls.  The stdlib engine supports its pattern
# and keeps this streaming wrapper runnable before the uv environment exists.
sys.modules.setdefault("regex", re)
import preprocess as source  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("cf", "fb23", "wm"), required=True)
    parser.add_argument("--key-sample-ratio", type=int, required=True)
    parser.add_argument("--count-file", type=Path, required=True)
    args = parser.parse_args()
    if args.key_sample_ratio < 1:
        parser.error("--key-sample-ratio must be at least 1")
    return args


def quick_key(line: str, input_format: str) -> int:
    try:
        key_text = line.split(",", 2)[1].strip()
    except IndexError as error:
        raise ValueError(f"input row has no key column: {line!r}") from error
    if input_format == "cf":
        return int(key_text, 16)
    digest = hashlib.sha256(key_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & source._MASK64


def row_timestamp(line: str, input_format: str) -> int:
    timestamp = int(line.split(",", 1)[0].strip()) & source._MASK64
    return timestamp // 1000 if input_format == "fb23" else timestamp


def parse_selected(line: str, input_format: str):
    if input_format == "cf":
        return source._parse_cf(line.split(",", 7))
    if input_format == "wm":
        return source._parse_wm(line.split(",", 2))
    return source._parse_fb23(line.split(","))


def main() -> int:
    args = parse_args()
    output = sys.stdout.buffer
    started = time.time()
    source_rows = 0
    selected_rows = 0
    start_ts: int | None = None
    header_skipped = False

    for raw in sys.stdin:
        line = raw.rstrip("\r\n")
        if not header_skipped:
            header_skipped = True
            continue
        if not line:
            continue
        source_rows += 1
        if start_ts is None:
            start_ts = row_timestamp(line, args.format)
        if quick_key(line, args.format) % args.key_sample_ratio:
            continue

        parsed = parse_selected(line, args.format)
        if not parsed:
            continue
        ts, key, zone, size, ttl, ttstale, mime, is_purge = parsed
        ts -= start_ts - 1
        ts = min(ts, source.MAX_UINT32)
        ttl = min(ttl, source.MAX_UINT32)
        ttstale = min(ttstale, source.MAX_UINT32)
        output.write(source._PACKER.pack(
            key, zone, size, ts, 0, ttl, ttstale, mime, is_purge
        ))
        selected_rows += 1

        if selected_rows % 100_000 == 0:
            print(".", end="", flush=True, file=sys.stderr)

    if start_ts is None:
        raise RuntimeError("trace contained no data rows")
    args.count_file.write_text(f"{selected_rows}\n", encoding="utf-8")
    elapsed = time.time() - started
    print(
        f"\nScanned {source_rows} rows; retained {selected_rows} "
        f"(1/{args.key_sample_ratio}) in {elapsed:.2f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
