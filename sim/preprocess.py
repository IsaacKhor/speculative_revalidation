#!/usr/bin/env python3

import regex
import argparse
import hashlib
import struct
import sys
from typing import List
import time

# matches simc/preprocess.hpp CdnRequest layout
_PACKER = struct.Struct('@QQQIIIII?3x')
_MASK64 = 0x7fffffffffffffff
# Wikimedia traces do not ship TTL metadata, so default to 24h TTL.
_WM_DEFAULT_TTL = 24 * 60 * 60

MAX_UINT32 = 0xffffffff


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert trace CSV rows to binary format expected by the simulator.')
    parser.add_argument('--format', choices=('cf', 'fb23', 'wm'),
                        default='cf', help='Input trace format (default: cf).')
    parser.add_argument('--progress', action='store_true', default=False,
                        help='Show progress output.')
    return parser.parse_args()


# based on manual analysis of top cf trace mime types
cf_mime_mapping = [
    [1, 'application/font-woff2'],
    [1, 'font/otf'],
    [1, 'font/ttf'],
    [1, 'font/woff'],
    [1, 'font/woff2'],

    [2, 'application/javascript'],
    [2, 'application/x-javascript'],
    [2, 'text/javascript'],

    [3, 'application/json'],
    [4, 'application/ocsp-response'],
    [5, 'application/octet-stream'],

    [6, 'application/vnd.apple.mpegurl'],
    [6, 'application/x-mpegurl'],

    [7, 'application/vnd.npm.install-v1+json'],
    [8, 'application/x-binary'],
    [9, 'application/xml'],
    [10, 'application/zip'],

    [11, 'audio/x-m4a'],
    [12, 'binary/octet-stream'],
    [13, 'image/avif'],
    [14, 'image/gif'],

    [15, 'image/jpeg'],
    [15, 'image/jpg'],
    [15, 'image/jxl'],

    [16, 'image/png'],
    [17, 'image/svg+xml'],

    [18, 'image/vnd.microsoft.icon'],
    [18, 'image/x-icon'],

    [19, 'image/webp'],
    [20, 'image/x-etc2'],
    [21, 'multipart/byteranges'],
    [22, 'text/calendar'],
    [23, 'text/css'],
    [24, 'text/html'],
    [25, 'text/plain'],
    [26, 'text/vtt'],

    [27, 'video/webm'],
    [28, 'video/m2ts'],
    [28, 'video/mp2t'],
    [28, 'text/vnd.trolltech.linguist'],
    [28, 'text/vnd.qt.linguist'],

    [29, 'video/mp4'],
    [30, 'video/quicktime'],
    [31, 'video/webm'],
]

cf_mime_generic = [
    [1, 'font'],
    [3, 'json'],
    [11, 'audio'],
    [15, 'image'],
    [18, 'icon'],
    [29, 'video'],
]

cf_mime_map = {mime: idx for idx, mime in cf_mime_mapping}
pattern = regex.compile(r'^[^a-z]*([^;]+)')

def cf_content_type_map(mime: str) -> int:
    if match := regex.match(pattern, mime):
        return cf_mime_map.get(match.group(1).strip(), 0)
    for idx, prefix in cf_mime_generic:
        if mime.startswith(prefix):
            return idx
    return 0


def _parse_cf(parts: List[str]) -> tuple[int, int, int, int, int, int, int, bool]:
    if len(parts) < 8:
        raise ValueError(f'invalid CF row: {parts!r}')
    ts = int(parts[0].strip()) & _MASK64
    key = int(parts[1].strip(), 16)
    zone = int(parts[2].strip(), 16)
    size = int(parts[3].strip(), 10) & _MASK64
    ttl = int(parts[4].strip(), 10) & _MASK64
    ttstale = int(parts[5].strip(), 10) & _MASK64
    mime = cf_content_type_map(parts[7])
    is_purge = parts[6].strip() == '1'
    return ts, key, zone, size, ttl, ttstale, mime, is_purge


def _parse_fb23(parts: List[str]) -> tuple[int, int, int, int, int, int, int, bool] | None:
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

    # see content_type.ipynb for analysis of FB23 mime types, there's only a
    # handful of >30s
    mime = 30
    try:
        mime = int(parts[-2].strip())
    except ValueError:
        pass

    if mime > 30:
        mime = 30

    return ts, key, zone, size, ttl, ttstale, mime, is_purge


def _parse_wm(parts: List[str]) -> tuple[int, int, int, int, int, int, int, bool]:
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
    return ts, key, zone, size, ttl, ttstale, 0, is_purge


def main() -> None:
    args = _parse_args()
    out = sys.stdout.buffer

    tstart = time.time()
    header_skipped = False
    processed_rows = 0
    start_ts = 0
    ts_clamp_count = 0
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

        ts, key, zone, size, ttl, ttstale, mime, is_purge = parsed
        if mime < 0 or mime > 40:
            raise ValueError(f'invalid mime type {mime} in row: {raw}')

        if processed_rows == 1:
            start_ts = ts

        ts -= start_ts - 1  # start at 1 to avoid zero timestamps
        if ts > MAX_UINT32:
            ts_clamp_count += 1
            ts = MAX_UINT32

        if ttl > MAX_UINT32:
            ttl = MAX_UINT32

        if ttstale > MAX_UINT32:
            ttstale = MAX_UINT32

        if processed_rows < 5:
            print(f'Row {processed_rows}: key={key:x}, zone={zone:x}, size={size}, ts={ts}, ttl={ttl}, ttstale={ttstale}, mime={mime}, is_purge={is_purge}', file=sys.stderr)

        if args.progress and processed_rows % 1_000_000 == 0:
            print('.', end='', flush=True, file=sys.stderr)
        if args.progress and processed_rows % 50_000_000 == 0:
            print(f'={processed_rows/1_000_000}m rows', file=sys.stderr)

        try:
            packed = _PACKER.pack(key, zone, size, ts, 0,
                                  ttl, ttstale, mime, is_purge)
        except struct.error as e:
            raise RuntimeError(
                f'Error packing row {processed_rows} ({raw}): {e}')

        out.write(packed)

    print(f'\nProcessed {processed_rows/1_000_000}m rows', file=sys.stderr)
    elapsed = time.time() - tstart
    print(f'Time elapsed: {elapsed:.2f} seconds', file=sys.stderr)
    print(f'Rows per second: {processed_rows/elapsed:.2f}', file=sys.stderr)
    print(f'TS clamp count: {ts_clamp_count}', file=sys.stderr)


if __name__ == '__main__':
    main()
