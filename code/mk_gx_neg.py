# mk_gx_neg.py -- PROTOCOL_IDO.md I64: rearranged copies of the CALIBRATION images (val1 ID, devall-) only; no score is computed, no val2 /
#   test / sel- image is read.  Each copy keeps the pixels and loses the familiar arrangement:
#     r90, r270  rotated by 90 / 270 degrees            j2  the four quadrants exchanged diagonally (a cyclic shift by half the side)
#     j4         the 4 x 4 patches permuted by one fixed permutation without fixed points (the same for every image and every ID)
#     ch         the left half of image i joined to the right half of image i + 1
#   per base tag <id>: data/npy/gxn_<id>_<copy>.npy and runs/ido/gxn_<id>.json (the calibration row and the five copies, devall-<ID>__<copy>);
#   the same from the identical-history calibration images (I49): data/npy/gxn_<id>_eq50_<copy>.npy, runs/ido/gxn_<id>_eq50.json
import os, sys, json
import numpy as np
R = os.path.expanduser('~/olr'); NPY = f'{R}/data/npy'; BASE = sys.argv[1:] or ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']      # (a tag list may be given: the grey arrays are [N, H, W])
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
rs = np.random.RandomState(0); PERM = rs.permutation(16)
while np.any(PERM == np.arange(16)): PERM = rs.permutation(16)
def j4(X):
    n = X.shape[1]; s = n // 4; assert n % 4 == 0; Y = np.empty_like(X)
    for k in range(16): i, j, u, v = k // 4, k % 4, PERM[k] // 4, PERM[k] % 4; Y[:, i * s:(i + 1) * s, j * s:(j + 1) * s] = X[:, u * s:(u + 1) * s, v * s:(v + 1) * s]
    return Y
def ch(X): n = X.shape[2]; Y = X.copy(); Y[:, :, n // 2:] = np.roll(X, -1, axis=0)[:, :, n // 2:]; return Y
COPY = {'r90': lambda X: np.rot90(X, 1, axes=(1, 2)), 'r270': lambda X: np.rot90(X, 3, axes=(1, 2)), 'j2': lambda X: np.roll(X, (X.shape[1] // 2, X.shape[2] // 2), axis=(1, 2)), 'j4': j4, 'ch': ch}
for t in BASE:
    for suf, src in (('', f'{R}/runs/ido/gx_{t}.json'), ('_eq50', f'{R}/runs/ido/gx_{t}_eq50.json')):
        cal = [r for r in json.load(open(src)) if r[0] == f'devall-{IDD[t]}']; assert len(cal) == 1 and cal[0][3] == 'none'
        X = np.ascontiguousarray(np.load(f'{NPY}/{cal[0][1]}.npy', mmap_mode='r')[np.asarray(cal[0][2])]); assert X.ndim in (3, 4) and X.shape[1] == X.shape[2] and X.dtype == np.uint8; rows = list(cal)
        for c, f in COPY.items():
            Y = np.ascontiguousarray(f(X)); assert Y.shape == X.shape and Y.dtype == np.uint8 and (c == 'ch' or np.array_equal(np.sort(Y.reshape(len(Y), -1), 1), np.sort(X.reshape(len(X), -1), 1)))
            p = f'{NPY}/gxn_{t}{suf}_{c}.npy'; assert not os.path.exists(p), p; np.save(p, Y); rows.append([f'devall-{IDD[t]}__{c}', f'gxn_{t}{suf}_{c}', list(range(len(Y))), 'none'])
        o = f'{R}/runs/ido/gxn_{t}{suf}.json'; assert not os.path.exists(o), o; json.dump(rows, open(o, 'w')); print(t + suf, X.shape, 'copies', len(COPY), flush=True)
print('PERM', PERM.tolist()); print('MK_GX_NEG_DONE')
