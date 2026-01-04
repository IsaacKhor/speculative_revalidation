#!/usr/bin/env bash
set -xeuo pipefail

source .venv/bin/activate
xmake

iter=$1
runnum=8
njobs=16
ksr=1

# # get baseline expiry traces
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

# baseline never/always
xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
    -itraces/sim/cf_{a..h}.bin.zst \
    -itraces/sim/fb_{a..c}.bin.zst \
    -itraces/sim/wm_t.bin.zst \
    --rv-mode=never \
    --csvout=results/r${runnum}i${iter}_baseline.csv

# baseline oracle
xmake r sim -s$ksr --parallel=$njobs --capacity=2048 \
    -itraces/sim/cf_{a..h}.bin.zst \
    -itraces/sim/fb_{a..c}.bin.zst \
    -itraces/sim/wm_t.bin.zst \
    --rv-mode=oracle \
    --ml-conf-thres={1,2,3,4,6,8,12,16} \
    --csvout=results/r${runnum}i${iter}_oracle.csv

# ml models, cf
xmake r sim -s$ksr -p$njobs -c2048 \
    -itraces/sim/cf_{a,c,e,g}.bin.zst \
    --rv-mode=ml \
    --ml-model-path=models/v${runnum}_cf_rfc.onnx \
    --ml-conf-thres={0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1} \
    --csvout=results/r${runnum}i${iter}_ml_cf_rfc.csv

# ml models, fb
xmake r sim -s$ksr -p$njobs -c2048 \
    -itraces/sim/fb_{a,b,c}.bin.zst \
    --rv-mode=ml \
    --ml-model-path=models/v${runnum}_fb_rfc.onnx \
    --ml-conf-thres={0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1} \
    --csvout=results/r${runnum}i${iter}_ml_fb_rfc.csv

# ml models, wm
xmake r sim -s$ksr -p$njobs -c2048 \
    -itraces/sim/wm_t.bin.zst \
    --rv-mode=ml \
    --ml-model-path=models/v${runnum}_wm_rfc.onnx \
    --ml-conf-thres={0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1} \
    --csvout=results/r${runnum}i${iter}_ml_wm_rfc.csv
