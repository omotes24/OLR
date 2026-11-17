# mk_gx_sets.py -- PROTOCOL_IDO.md I59: the sets of the exploration of a simpler score.  Development data only (val1 ID = devall-, val2 =
#   fresh-); the sel- rows of runs/hx/sets1_<tag>.json are not used and no test split is touched.  No score is computed.  Per base tag <id>:
#     runs/ido/gx_<id>.json       the calibration set, the val2 sets, and dqv-<ID>__<cond>: the seven degraded copies of I47 (the functions of
#                                 code/mk_rfx_files.py) made from the val2 ID images (data/npy/dqv_<id>_<cond>.npy)
#     runs/ido/gx_<id>_eq50.json  the identical-history versions (2x BOX reduction and bicubic back, I49): the calibration set of
#                                 runs/ok/sets_<id>_eq50.json and the val2 sets (data/npy/eq50_<id>_<set>.npy)
import os, io, json
import numpy as np
from PIL import Image, ImageFilter
R = os.path.expanduser('~/olr'); NPY = f'{R}/data/npy'; BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
def put(path, obj):
    assert not os.path.exists(path), path; json.dump(obj, open(path, 'w'))
def pil(a): a = np.asarray(a); return Image.fromarray(a[..., 0] if (a.ndim == 3 and a.shape[2] == 1) else a)
def back(im, like): b = np.asarray(im, np.uint8); return b[..., None] if (like.ndim == 3 and like.shape[2] == 1) else b
def jpeg(im, q): buf = io.BytesIO(); im.save(buf, format='JPEG', quality=q); buf.seek(0); return Image.open(buf).copy()
def down(im, f):
    n = im.size[0]; m = int(round(n * f)); return im.resize((m, m), Image.BOX if f == 0.5 else Image.BICUBIC).resize((n, n), Image.BICUBIC)
COND = {'blur05': lambda im: im.filter(ImageFilter.GaussianBlur(0.5)), 'blur10': lambda im: im.filter(ImageFilter.GaussianBlur(1.0)), 'jpeg90': lambda im: jpeg(im, 90),
        'jpeg75': lambda im: jpeg(im, 75), 'jpeg50': lambda im: jpeg(im, 50), 'down75': lambda im: down(im, 0.75), 'down50': lambda im: down(im, 0.5)}
def save(name, Y):
    p = f'{NPY}/{name}.npy'; assert not os.path.exists(p), p; np.save(p, Y)
for t in BASE:
    S = [r for r in json.load(open(f'{R}/runs/hx/sets1_{t}.json')) if r[0].startswith(('devall-', 'fresh-'))]; assert not any(r[0].startswith('sel-') for r in S)
    cal = [r for r in S if r[0] == f'devall-{IDD[t]}']; fr = [r for r in S if r[0].startswith('fresh-')]; idr = [r for r in fr if r[0] == f'fresh-{IDD[t]}']; assert len(cal) == 1 and len(idr) == 1 and idr[0][3] == 'none'
    X = np.ascontiguousarray(np.load(f'{NPY}/{idr[0][1]}.npy', mmap_mode='r')[np.asarray(idr[0][2])]); rows = cal + fr
    for c, f in COND.items():
        Y = np.stack([back(f(pil(x)), x) for x in X]); assert Y.shape == X.shape and Y.dtype == np.uint8; save(f'dqv_{t}_{c}', Y); rows.append([f'dqv-{IDD[t]}__{c}', f'dqv_{t}_{c}', list(range(len(Y))), 'none'])
    put(f'{R}/runs/ido/gx_{t}.json', rows)
    ce = [r for r in json.load(open(f'{R}/runs/ok/sets_{t}_eq50.json')) if r[0] == f'devall-{IDD[t]}']; assert len(ce) == 1 and len(ce[0][2]) == len(cal[0][2]); rows, done = list(ce), {}
    for name, data, idx, p in fr:
        if p == 'none':
            Xs = np.ascontiguousarray(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx)]); Y = np.stack([back(down(pil(x), 0.5), x) for x in Xs]); assert Y.shape == Xs.shape
            save(f'eq50_{t}_{name}', Y); done[(data, tuple(idx))] = (f'eq50_{t}_{name}', len(Y)); rows.append([name, f'eq50_{t}_{name}', list(range(len(Y))), 'none'])
    for name, data, idx, p in fr:
        if p != 'none': d, m = done[(data, tuple(idx))]; rows.append([name, d, list(range(m)), p])
    assert len(rows) == 1 + len(fr); put(f'{R}/runs/ido/gx_{t}_eq50.json', rows); print(t, 'val2 sets', len(fr), 'degraded', len(COND), flush=True)
print('MK_GX_SETS_DONE')
