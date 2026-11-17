#!/bin/bash
# run_ol.sh <gpu> <mode> <tag>:<seed> ... -- PROTOCOL_IDO.md I68: the frozen code/ol_score.py on one GPU, for a base tag and then for its
#   identical-history version (<tag>_eq50, through code/rfx_run.py).  No AUROC is computed here.
#   mode dev    the val2 sets (prefix fresh-, label dev): the check against the development numbers
#   mode final  the six test splits (prefix test2- ... test7-, label final): the single evaluation
cd ~/olr; PY=${PY:-python}; G=$1; M=$2; shift; shift; L=logs/ido/ol; mkdir -p $L
case $M in dev) P=fresh-;; final) P=test2-,test3-,test4-,test5-,test6-,test7-;; *) exit 9;; esac
for J in "$@"; do T=${J%%:*}; s=${J##*:}; X=""; [ $M = dev ] && X="--sets runs/ido/gx_${T}_eq50.json"
  $PY code/ol_score.py --tag $T --gpu $G --seed $s --prefix $P --label $M > $L/${M}_${T}_s$s.log 2>&1; r1=$?
  $PY code/rfx_run.py code/ol_score.py --tag ${T}_eq50 --gpu $G --seed $s --prefix $P --label $M $X > $L/${M}_${T}_eq50_s$s.log 2>&1; r2=$?
  echo "OL $M $J rc=$r1$r2 $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/ol_all.log
done
echo "OL_LANE_DONE $M gpu$G $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/ol_all.log
