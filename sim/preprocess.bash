#!/usr/bin/env bash

source .venv/bin/activate

# turn traces/wm/t-all.csv.zst into traces/sim/wm_t.bin.zst
zstdcat traces/wm/t-all.csv.zst | python sim/preprocess.py --format wm | zstd > traces/sim/wm_t.bin.zst

# turn traces/cf/csv/*.csv.zst into traces/sim/cf_*.csv
for f in traces/cf/csv/*.csv.zst; do
    outf="traces/sim/$(basename "$f" .csv.zst | sed 's/^/cf_/')".bin.zst
    echo "Processing $f -> $outf"
    zstdcat "$f" | python sim/preprocess.py --format cf | zstd > "$outf"
done

# turn traces/fb23/*.csv.zst into traces/sim/fb23_*.csv
for f in traces/fb23/*.csv.zst; do
    outf="traces/sim/$(basename "$f" .csv.zst | sed 's/^/fb23_/')".bin.zst
    echo "Processing $f -> $outf"
    zstdcat "$f" | python sim/preprocess.py --format fb23 | zstd > "$outf"
done
