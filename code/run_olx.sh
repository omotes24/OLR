#!/bin/bash
# run_olx.sh <gpu> <job> ... -- PROTOCOL_IDO.md I70: diagnostic runs of the frozen code/ol_score.py on one GPU.  No AUROC is computed here.
#   dq:<tag>:<seed>   the degraded copies of the ID images of test7 (and the ID set itself) of a base tag                       (label dq)
#   var:<vtag>        a variant tag of code/rfx_run.py (<id>_rinit, <id>_rshuf in fp32; <id>_by_<model>) on test7, noise seed 0  (label x)
#   hr:<tier>:<seed>  the 256x256 tiers, batch 4: in256c (code/e11hr_run.py), in256d = the 112-px control (code/rfx_run.py)      (label hr)
cd ~/olr; PY=${PY:-python}; G=$1; shift; L=logs/ido/olx; mkdir -p $L
idname() { case $1 in edmc10) echo cifar10_test;; l6svhn) echo svhn_test;; dpceleba) echo celeba_test;; in64) echo in64_imagenet_test;; fmnist) echo g32_fmnist_test;; esac; }
for J in "$@"; do
  case $J in
    dq:*) T=$(echo $J | cut -d: -f2); s=$(echo $J | cut -d: -f3)
      $PY code/ol_score.py --tag $T --gpu $G --seed $s --prefix dq-,test7-$(idname $T) --label dq > $L/dq_${T}_s$s.log 2>&1; rc=$?;;
    var:*) V=${J#*:}; F=1; case $V in *_rinit|*_rshuf) F=0;; esac
      DG_FP16=$F $PY code/rfx_run.py code/ol_score.py --tag $V --gpu $G --seed 0 --prefix test7- --label x > $L/var_$V.log 2>&1; rc=$?;;
    hr:*) V=$(echo $J | cut -d: -f2); s=$(echo $J | cut -d: -f3); case $V in in256c) RUN=code/e11hr_run.py; P=q256;; in256d) RUN=code/rfx_run.py; P=q256d;; *) RUN=none;; esac
      HR_BS=4 $PY $RUN code/ol_score.py --tag $V --gpu $G --seed $s --prefix $P-imagenet_test,$P-ssb_hard,$P-ninco,$P-inaturalist,$P-textures,$P-openimage_o --label hr > $L/hr_${V}_s$s.log 2>&1; rc=$?;;
    *) rc=99;;
  esac
  echo "OLX $J rc=$rc $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/olx_all.log
done
echo "OLX_LANE_DONE gpu$G $(date -u +%m-%dT%H:%M:%S)" >> logs/ido/olx_all.log
