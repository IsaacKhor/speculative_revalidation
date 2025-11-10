#!/usr/bin/env bash

xmake

# cf traces
xmake r sim -s1 --parallel=16 --capacity=2048 \
    -itraces/sim/cf_{a,c,e,g}.bin.zst \
    -itraces/sim/fb23_{a,b,c}.bin.zst \
    --rv-min-ttl={1,5,15,30,45,60,120} \
    --rv-min-freq={1,2,3,4,5,6,7,8}
