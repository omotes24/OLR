#!/bin/bash
# run_gxc.sh <gpu> <seed> <tag> ... -- PROTOCOL_IDO.md I61 part II: the per-channel statistics of block A (code/gx_extract_c.py) for base tags:
#   the val2 sets with their degraded copies, and the identical-history versions (tag <id>_eq50).  No score is computed here.
cd ~/olr; PY=${PY:-python}; G=$1; S=$2; shift; shift; L=logs/ido/gx; mkdir -p $L
for T in "$@"; do
  $PY code/gx_extract_c.py --tag $T --gpu $G --seed $S --sets runs/ido/gx_$T.json --label c > $L/c_${T}_s$S.log 2>&1; r1=$?
  $PY code/rfx_run.py code/gx_extract_c.py --tag ${T}_eq50 --gpu $G --seed $S --sets runs/ido/gx_${T}_eq50.json --label c --prefix devall-,fresh- > $L/c_${T}_eq50_s$S.log 2>&1; r2=$?
  echo "GXC $T s$S rc=$r1$r2 $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
done
echo "GXC_LANE_DONE gpu$G s$S $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
