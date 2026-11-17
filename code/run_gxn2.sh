#!/bin/bash
# run_gxn2.sh <gpu> <seed> <set-prefix> <label> <tag> ... -- PROTOCOL_IDO.md I66: the block statistics (code/gx_extract.py) of copies of the
#   calibration images listed in runs/ido/<set-prefix>_<tag>[_eq50].json (gxn: code/mk_gx_neg.py, label n; gxn2: code/mk_gx_neg2.py, label n2),
#   for base tags and their identical-history versions.  No score is computed here.
cd ~/olr; PY=${PY:-python}; G=$1; S=$2; P=$3; B=$4; shift; shift; shift; shift; L=logs/ido/gx; mkdir -p $L
for T in "$@"; do
  $PY code/gx_extract.py --tag $T --gpu $G --seed $S --sets runs/ido/${P}_$T.json --label $B --prefix devall- > $L/${T}_${B}_s$S.log 2>&1; r1=$?
  $PY code/rfx_run.py code/gx_extract.py --tag ${T}_eq50 --gpu $G --seed $S --sets runs/ido/${P}_${T}_eq50.json --label $B --prefix devall- > $L/${T}_eq50_${B}_s$S.log 2>&1; r2=$?
  echo "GXN2 $P $T s$S rc=$r1$r2 $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
done
echo "GXN2_LANE_DONE $P gpu$G s$S $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/gx_all.log
