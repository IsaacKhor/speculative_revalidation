#!/usr/bin/env python3

import argparse
import hashlib
import struct
import sys
from typing import List
import time

# matches simc/preprocess.hpp CdnRequest layout
_PACKER = struct.Struct('<QQQQQQ?7x')
_MASK64 = 0x7fffffffffffffff
# Wikimedia traces do not ship TTL metadata, so default to 24h TTL.
_WM_DEFAULT_TTL = 24 * 60 * 60


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert trace CSV rows to binary format expected by the simulator.')
    parser.add_argument('--format', choices=('cf', 'fb23', 'wm'),
                        default='cf', help='Input trace format (default: cf).')
    return parser.parse_args()


def _parse_cf(parts: List[str]) -> tuple[int, int, int, int, int, int, bool]:
    if len(parts) < 8:
        raise ValueError(f'invalid CF row: {parts!r}')
    ts = int(parts[0].strip()) & _MASK64
    key = int(parts[1].strip(), 16)
    zone = int(parts[2].strip(), 16)
    size = int(parts[3].strip(), 10) & _MASK64
    ttl = int(parts[4].strip(), 10) & _MASK64
    ttstale = int(parts[5].strip(), 10) & _MASK64
    is_purge = parts[6].strip() == '1'
    return ts, key, zone, size, ttl, ttstale, is_purge


def _parse_fb23(parts: List[str]) -> tuple[int, int, int, int, int, int, bool] | None:
    if len(parts) < 15:
        raise ValueError(f'invalid FB23 row: {parts!r}')
    ts = int(parts[0].strip()) & _MASK64
    cache_key = parts[1].strip()
    digest = hashlib.sha256(cache_key.encode('utf-8')).digest()
    # Map long cache keys into the simulator's 64-bit key space.
    key = int.from_bytes(digest[:8], 'little') & _MASK64
    objsize = int(parts[3].strip())
    respsize = int(parts[4].strip())
    if objsize < 0 and respsize < 0:
        return None
    size = objsize if objsize > 0 else respsize
    ttl = int(float(parts[8].strip())) & _MASK64
    zone = 0
    ttstale = 0
    is_purge = False
    return ts, key, zone, size, ttl, ttstale, is_purge


def _parse_wm(parts: List[str]) -> tuple[int, int, int, int, int, int, bool]:
    if len(parts) < 3:
        raise ValueError(f'invalid WM row: {parts!r}')
    ts = int(parts[0].strip()) & _MASK64
    cache_key = parts[1].strip()
    digest = hashlib.sha256(cache_key.encode('utf-8')).digest()
    # Map long cache keys into the simulator's 64-bit key space.
    key = int.from_bytes(digest[:8], 'little') & _MASK64
    size = int(parts[2].strip()) & _MASK64
    zone = 0
    ttl = _WM_DEFAULT_TTL & _MASK64
    ttstale = 0
    is_purge = False
    return ts, key, zone, size, ttl, ttstale, is_purge


def main() -> None:
    args = _parse_args()
    out = sys.stdout.buffer

    tstart = time.time()
    header_skipped = False
    processed_rows = 0
    for raw in sys.stdin:
        line = raw.rstrip('\r\n')
        if not header_skipped:
            header_skipped = True
            continue
        if not line:
            continue
        if args.format == 'cf':
            parts = line.split(',', 7)
            parsed = _parse_cf(parts)
        elif args.format == 'wm':
            parts = line.split(',', 2)
            parsed = _parse_wm(parts)
        else:
            parts = line.split(',')
            parsed = _parse_fb23(parts)
        processed_rows += 1

        if not parsed:
            continue

        ts, key, zone, size, ttl, ttstale, is_purge = parsed

        if processed_rows < 5:
            print(f'Row {processed_rows}: ts={ts}, key={key:x}, '
                  f'zone={zone:x}, size={size}, ttl={ttl}, '
                  f'ttstale={ttstale}, is_purge={is_purge}', file=sys.stderr)

        if processed_rows % 1_000_000 == 0:
            print('.', end='', flush=True, file=sys.stderr)
        if processed_rows % 50_000_000 == 0:
            print(f'={processed_rows/1_000_000}m rows', file=sys.stderr)

        try:
            packed = _PACKER.pack(ts, key, zone, size, ttl, ttstale, is_purge)
        except struct.error as e:
            print(
                f'\nError packing row {processed_rows} ({raw}): {e}', file=sys.stderr)
            continue

        out.write(packed)

    print(f'\nProcessed {processed_rows/1_000_000}m rows', file=sys.stderr)
    elapsed = time.time() - tstart
    print(f'Time elapsed: {elapsed:.2f} seconds', file=sys.stderr)
    print(f'Rows per second: {processed_rows/elapsed:.2f}', file=sys.stderr)


if __name__ == '__main__':
    main()
