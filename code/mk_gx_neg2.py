# mk_gx_neg2.py -- PROTOCOL_IDO.md I66: the mirrored copies (left-right) of the CALIBRATION images (val1 ID, devall-) only; with the
#   upside-down pass of code/gx_extract.py they also give the half turn.  No score is computed, no val2 / test / sel- image is read.
#   per base tag <id>: data/npy/gxn_<id>_hf.npy, runs/ido/gxn2_<id>.json (the calibration row and devall-<ID>__hf); the same from the
#   identical-history calibration images (I49): data/npy/gxn_<id>_eq50_hf.npy, runs/ido/gxn2_<id>_eq50.json
import os, sys, json
import numpy as np
R = os.path.expanduser('~/olr'); NPY = f'{R}/data/npy'; BASE = sys.argv[1:] or ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
for t in BASE:
    for suf, src in (('', f'{R}/runs/ido/gx_{t}.json'), ('_eq50', f'{R}/runs/ido/gx_{t}_eq50.json')):
        cal = [r for r in json.load(open(src)) if r[0] == f'devall-{IDD[t]}']; assert len(cal) == 1 and cal[0][3] == 'none'
        X = np.ascontiguousarray(np.load(f'{NPY}/{cal[0][1]}.npy', mmap_mode='r')[np.asarray(cal[0][2])]); assert X.ndim in (3, 4) and X.shape[1] == X.shape[2] and X.dtype == np.uint8
        Y = np.ascontiguousarray(X[:, :, ::-1]); assert Y.shape == X.shape and np.array_equal(Y[:, :, ::-1], X); p = f'{NPY}/gxn_{t}{suf}_hf.npy'; assert not os.path.exists(p), p; np.save(p, Y)
        o = f'{R}/runs/ido/gxn2_{t}{suf}.json'; assert not os.path.exists(o), o; json.dump(list(cal) + [[f'devall-{IDD[t]}__hf', f'gxn_{t}{suf}_hf', list(range(len(Y))), 'none']], open(o, 'w')); print(t + suf, X.shape, flush=True)
print('MK_GX_NEG2_DONE')
