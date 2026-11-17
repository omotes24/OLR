#!/bin/bash
# run_gxn.sh <gpu> <seed> <tag> ... -- PROTOCOL_IDO.md I64: the block statistics (code/gx_extract.py) of the rearranged copies of the
#   calibration images (code/mk_gx_neg.py), for base tags and their identical-history versions.  No score is computed here.
cd ~/olr; PY=${PY:-python}; G=$1; S=$2; shift; shift; L=logs/ido/gx; mkdir -p $L
for T in "$@"; do
  $PY code/gx_extract.py --tag $T --gpu $G --seed $S --sets runs/ido/gxn_$T.json --label n --prefix devall- > $L/${T}_n_s$S.log 2>&1; r1=$?
  $PY code/rfx_run.py code/gx_extract.py --tag ${T}_eq50 --gpu $G --seed $S --sets runs/ido/gxn_${T}_eq50.json --label n --prefix devall- > $L/${T}_eq50_n_s$S.log 2>&1; r2=$?
  echo "GXN $T s$S rc=$r1$r2 $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
done
echo "GXN_LANE_DONE gpu$G s$S $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
