#!/usr/bin/env bash

xmake

# baseline
xmake r sim -s1 --parallel=16 --capacity=2048 \
    -itraces/sim/cf_{a,c,e,g}.bin.zst \
    -itraces/sim/fb_{a,b,c}.bin.zst \
    -itraces/sim/wm_t.bin.zst \
    --rv-mode={never,always,oracle} \
    --csvout=results/baseline.csv

# cf traces
# xmake r sim -s1 --parallel=16 --capacity=2048 \
#     -itraces/sim/cf_{a,c,e,g}.bin.zst \
#     -itraces/sim/fb23_{a,b,c}.bin.zst \
#     --rv-min-ttl={1,5,15,30,45,60,120} \
#     --rv-min-freq={1,2,3,4,5,6,7,8} \
#     --csvout=results/run_0.csv

xmake r sim -s1 --parallel=16 --capacity=2048 \
    -itraces/sim/wm_a.bin.zst \
    --rv-min-ttl={1,5,15,30,45,60,120} \
    --rv-min-freq={1,2,3,4,5,6,7,8} \
    --csvout=results/run_1_wm.csv

xmake r sim -s1 --parallel=16 --capacity=2048 \
    -itraces/sim/wm_a.bin.zst \
     -itraces/sim/cf_{a,c,e,g}.bin.zst \
     -itraces/sim/fb23_{a,b,c}.bin.zst \
    --rv-min-ttl=-1 \
    --rv-min-freq=-1 \
    --csvout=results/baseline.csv
