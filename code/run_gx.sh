#!/bin/bash
# run_gx.sh <gpu> <seed> <tag> ... -- PROTOCOL_IDO.md I59: the statistics of the exploration on val2 (code/gx_extract.py) for base tags:
#   the val2 sets and their degraded copies, the identical-history versions (tag <id>_eq50), and the untrained twin (tag <id>_rinit, fp32;
#   noise seed 0 only).  No score is computed here.
cd ~/olr; PY=${PY:-python}; G=$1; S=$2; shift; shift; L=logs/ido/gx; mkdir -p $L
for T in "$@"; do
  $PY code/gx_extract.py --tag $T --gpu $G --seed $S --sets runs/ido/gx_$T.json --label v > $L/${T}_s$S.log 2>&1; r1=$?
  $PY code/rfx_run.py code/gx_extract.py --tag ${T}_eq50 --gpu $G --seed $S --sets runs/ido/gx_${T}_eq50.json --label v --prefix devall-,fresh- > $L/${T}_eq50_s$S.log 2>&1; r2=$?
  r3=-; if [ "$S" = "0" ]; then DG_FP16=0 $PY code/rfx_run.py code/gx_extract.py --tag ${T}_rinit --gpu $G --seed 0 --sets runs/ido/gx_$T.json --label v --prefix devall-,fresh- > $L/${T}_rinit_s0.log 2>&1; r3=$?; fi
  echo "GX $T s$S rc=$r1$r2$r3 $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
done
echo "GX_LANE_DONE gpu$G s$S $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
