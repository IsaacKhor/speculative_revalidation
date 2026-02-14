#!/usr/bin/env bash
set -xeuo pipefail

source .venv/bin/activate
xmake

iter=$1
runnum=9
njobs=8
ksr=1

# get baseline expiry traces
# xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
#     -itraces/sim/cf_{a..h}.bin.zst \
#     -itraces/sim/fb_{a..c}.bin.zst \
#     -itraces/sim/wm_t.bin.zst \
#     --trace-expiry=true \
#     --rv-mode=oracle \
#     --ml-conf-thres=1 \
#     --csvout=/dev/null
# for f in traces/expiry/*.csv; do
#     zstd --rm -o traces/expiry/r${runnum}$(basename "$f").zst "$f"
# done

# python3 sim/train_ml.py --inputs traces/expiry/r${runnum}cf_*.zst --output models/v${runnum}_cf_rfc --sample-frac 0.1
# python3 sim/train_ml.py --inputs traces/expiry/r${runnum}fb_*.zst --output models/v${runnum}_fb_rfc
# python3 sim/train_ml.py --inputs traces/expiry/r${runnum}wm_*.zst --output models/v${runnum}_wm_rfc

# # baseline never, evict expired
# xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
#     -itraces/sim/cf_{a..h}.bin.zst \
#     -itraces/sim/fb_{a..c}.bin.zst \
#     -itraces/sim/wm_t.bin.zst \
#     -itraces/sim/wm_u.bin.zst \
#     --rv-mode=never \
#     --cache-type={lru,gdsf,sieve,arc,fifo} \
#     --csvout=results/r${runnum}i${iter}_baseline.csv

# # baseline never, keep expired
# xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
#     -itraces/sim/cf_{a..h}.bin.zst \
#     -itraces/sim/fb_{a..c}.bin.zst \
#     -itraces/sim/wm_t.bin.zst \
#     -itraces/sim/wm_u.bin.zst \
#     --rv-mode=never \
#     --evict-expired=false \
#     --cache-type={lru,gdsf,sieve,arc,fifo} \
#     --csvout=results/r${runnum}i${iter}_baseline_keep.csv

# # collect zone stats for some traces
# xmake r sim -s$ksr -p$njobs -c2048 \
#     -itraces/sim/cf_{a,c,e,g}.bin.zst \
#     --rv-mode=ml \
#     --cache-type=lru \
#     --dump-zonestats=true \
#     --ml-model-path=models/v${runnum}_cf_rfc.onnx \
#     --ml-conf-thres={0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1} \
#     --csvout=results/r${runnum}i${iter}_cfzs.csv

# # baseline oracle
# xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
#     -itraces/sim/cf_{a..h}.bin.zst \
#     -itraces/sim/fb_{a..c}.bin.zst \
#     -itraces/sim/wm_t.bin.zst \
#     -itraces/sim/wm_u.bin.zst \
#     --rv-mode=oracle \
#     --cache-type={lru,gdsf,sieve,arc,fifo} \
#     --ml-conf-thres={1,2,3,4} \
#     --csvout=results/r${runnum}i${iter}_oracle.csv

# # ml models, cf
# xmake r sim -s$ksr -p$njobs -c2048 \
#     -itraces/sim/cf_{a..h}.bin.zst \
#     --rv-mode=ml \
#     --cache-type={lru,gdsf,sieve,arc,fifo} \
#     --ml-model-path=models/v${runnum}_cf_rfc.onnx \
#     --ml-conf-thres={1,0.95,0.9,0.85,0.8,0.75,0.7,0.68,0.66} \
#     --csvout=results/r${runnum}i${iter}_ml_cf_rfc.csv

# # ml models, fb
# xmake r sim -s$ksr -p$njobs -c2048 \
#     -itraces/sim/fb_{a..c}.bin.zst \
#     --rv-mode=ml \
#     --cache-type={lru,gdsf,sieve,arc,fifo} \
#     --ml-model-path=models/v${runnum}_fb_rfc.onnx \
#     --ml-conf-thres={1,0.95,0.9,0.85,0.8,0.75,0.7,0.68,0.66} \
#     --csvout=results/r${runnum}i${iter}_ml_fb_rfc.csv

# # ml models, wm
# xmake r sim -s$ksr -p$njobs -c2048 \
#     -itraces/sim/wm_t.bin.zst \
#     -itraces/sim/wm_u.bin.zst \
#     --rv-mode=ml \
#     --cache-type={lru,gdsf,sieve,arc,fifo} \
#     --ml-model-path=models/v${runnum}_wm_rfc.onnx \
#     --ml-conf-thres={1,0.95,0.9,0.85,0.8,0.75,0.7,0.68,0.66} \
#     --csvout=results/r${runnum}i${iter}_ml_wm_rfc.csv

# vary za
xmake r sim -s2 -p$njobs -c2048 \
    -itraces/sim/cf_{a..h}.bin.zst \
    --rv-mode=ml \
    --cache-type=lru \
    --dump-zonestats=true \
    --ml-model-path=models/v${runnum}_cf_rfc.onnx \
    --ml-conf-thres={1,0.95,0.9,0.85,0.8,0.75,0.7,0.68,0.66} \
    --rv-ma-za={0.02,0.03,0.04,0.06,0.08,0.1,0.12,0.14,0.16} \
    --csvout=results/r${runnum}i${iter}_ml_cf_rfc_varyza.csv

