#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
xmake

mkdir -p traces/sim

# turn traces/wm/t-all.csv.zst into traces/sim/wm_t.bin.zst
# zstdcat traces/wm/t-all.csv.zst | python sim/preprocess.py --format wm > /tmp/trace.bin
# xmake r oracle_backpass /tmp/trace.bin
# zstd --rm -o traces/sim/wm_t.bin.zst /tmp/trace.bin

# turn traces/cf/csv/*.csv.zst into traces/sim/cf_*.csv
for f in traces/cf/csv/*.csv.zst; do
    outf="traces/sim/$(basename "$f" .csv.zst | sed 's/^/cf_/')".bin.zst
    echo "Processing $f -> $outf"
    zstdcat "$f" | python sim/preprocess.py --format cf > /tmp/trace.bin
    xmake r oracle_backpass /tmp/trace.bin
    rm "$outf"
    zstd --rm -o "$outf" /tmp/trace.bin
done

# turn traces/fb23/*.csv.zst into traces/sim/fb23_*.csv
for f in traces/fb23/*.csv.zst; do
    outf="traces/sim/$(basename "$f" .csv.zst | sed 's/^/fb23_/')".bin.zst
    echo "Processing $f -> $outf"
    zstdcat "$f" | python sim/preprocess.py --format fb23 > /tmp/trace.bin
    xmake r oracle_backpass /tmp/trace.bin
    zstd --rm -o "$outf" /tmp/trace.bin
done
