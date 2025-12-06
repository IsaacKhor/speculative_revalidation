#!/usr/bin/env bash
set -xeuo pipefail

xmake

iter=$1
runnum=4
njobs=16
ksr=1

# get baseline expiry traces
# rm -f traces/expiry/*.csv.zst
# xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
#     -itraces/sim/cf_{a..h}.bin.zst \
#     -itraces/sim/fb_{a..c}.bin.zst \
#     -itraces/sim/wm_t.bin.zst \
#     --trace-expiry=true \
#     --rv-mode=oracle \
#     --csvout=/dev/null
# for f in traces/expiry/*.csv; do
#     zstd --rm -o traces/expiry/$(basename "$f").zst "$f"
# done

# baseline
xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
    -itraces/sim/cf_{a..h}.bin.zst \
    -itraces/sim/fb_{a..c}.bin.zst \
    -itraces/sim/wm_t.bin.zst \
    --rv-mode={never,oracle} \
    --csvout=results/r${runnum}i${iter}_baseline.csv

# ml models, cf
xmake r sim -s$ksr -p$njobs -c2048 \
    -itraces/sim/cf_{a,c,e,g}.bin.zst \
    -itraces/sim/wm_t.bin.zst \
    --rv-mode=ml \
    --ml-model-path=models/rfc_cf_all_v4.onnx \
    --ml-conf-thres={0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1} \
    --csvout=results/r${runnum}i${iter}_ml_cf_rfc.csv

# ml models, fb
xmake r sim -s$ksr -p$njobs -c2048 \
    -itraces/sim/fb_{a,b,c}.bin.zst \
    -itraces/sim/wm_t.bin.zst \
    --rv-mode=ml \
    --ml-model-path=models/rfc_fb_all_v4.onnx \
    --ml-conf-thres={0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1} \
    --csvout=results/r${runnum}i${iter}_ml_fb_rfc.csv

# cf traces heuristics
# xmake r sim -s$ksr --parallel=8 --capacity=2048 \
#     -itraces/sim/cf_{a,c,e,g}.bin.zst \
#     -itraces/sim/fb_{a,b,c}.bin.zst \
#     --rv-min-ttl={1,5,15,30,45,60,120} \
#     --rv-min-freq={1,2,3,4,5,6,7,8} \
#     --csvout=results/r0.csv

# xmake r sim -s$ksr --parallel=8 --capacity=2048 \
#     -itraces/sim/wm_a.bin.zst \
#     --rv-min-ttl={1,5,15,30,45,60,120} \
#     --rv-min-freq={1,2,3,4,5,6,7,8} \
#     --csvout=results/r1_wm.csv