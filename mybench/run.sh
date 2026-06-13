#!/bin/bash 
set -euo pipefail

# data is generated using libCacheSim/scripts/data_gen.py
# python3 libCacheSim/scripts/data_gen.py -m 1000000 -n 20000000 --alpha 1.0 --bin-output zipf1.0_1_100.oracleGeneral.bin

cd /mydata/tardis/sosp23-s3fifo/cachelib-sosp23/mybench

usage() {
   echo "$0 <algo> <cache size in MB> <hashpower>"
   exit
}

algo=${1:-"lru"} # Algorithm (duh)
sz_base=${2:-"4000"}
hp_base=${3:-"21"}

# Ensure turbo boost is disabled (you tried, but add check)
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    current=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "unknown")
    if [ "$current" != "1" ]; then
        echo "Warning: Turbo boost may still be enabled (value=$current)"
    fi
fi


for nThread in 1 2 4 8 16; do
    sz=$(echo "${sz_base} * ${nThread}" | bc)
    hp=$(echo "${hp_base} + l(${nThread})/l(2)" | bc -l | cut -d'.' -f1)
    # echo "############## ${algo} ${nThread} threads, cache size $sz MB, hashpower $hp"
    # numactl --membind=0 ./_build/${algo} zipf1.0_1_100.oracleGeneral.bin $sz $hp ${nThread} | tail -n 1

    echo "############## lru ${nThread} threads, cache size $sz MB, hashpower $hp"
    numactl --cpunodebind=0 --membind=0 ./_build/lru zipf1.0_1_100.oracleGeneral.bin $sz $hp ${nThread} | tail -n 1

    echo "############## lruforgive ${nThread} threads, cache size $sz MB, hashpower $hp"
    numactl --cpunodebind=0 --membind=0 ./_build/lruforgive zipf1.0_1_100.oracleGeneral.bin $sz $hp ${nThread} | tail -n 1
    # echo -e "thread results ${nThread} | tail -n 1
    "
done

