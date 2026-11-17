# mk_rfx_files.py -- PROTOCOL_IDO.md I47: the files of the diagnostic experiments (degraded copies, model variants).  No score is computed.
#   (1a) in256d (112-px control of PROTOCOL_HIRES2.md): runs/ido/sets8_in256d.json (the same 1,000 calibration images and the mirrored ID set,
#        from hc112_imagenet_test), runs/ido/probe_in256d.json
#   (1b) degraded copies of the ID images of test7, per ID: data/npy/dq_<tag>_<cond>.npy and runs/ido/sets9_<tag>.json (dq-<ID>__<cond>):
#        blur05, blur10 (Gaussian blur, sigma 0.5 / 1.0 px), jpeg90, jpeg75, jpeg50 (JPEG quality), down75 (bicubic to 3/4 of the side and
#        back), down50 (2x box reduction and bicubic back, the operation of prep_c112.py)
#   (2)  variant tags <id>_rinit, <id>_rshuf, <id>_by_<model>: links runs/ok/sets_<vtag>.json, runs/ido/sets7_<vtag>.json to the files of
#        <id>, and runs/ido/probe_<vtag>.json (four training images, so that hx_extract.py writes the hooks' meta.json)
import os, io, json
import numpy as np
from PIL import Image, ImageFilter
R = os.path.expanduser('~/olr'); NPY = f'{R}/data/npy'; BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']
BANK = {'edmc10': 'cifar10_train', 'l6svhn': 'svhn_train', 'dpceleba': 'celeba_train', 'in64': 'in64_imagenet_train10', 'fmnist': 'g32_fmnist_train'}
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
# ---- (1a)
cal = [int(i) for i in np.load(f'{R}/runs/base/F_in256c/hc_imagenet_test__cal.npz', allow_pickle=True)['idx']]
S = {r[0]: r for r in json.load(open(f'{R}/runs/ok/sets_in256d.json'))}; ev = S['q256d-imagenet_test'][2]
assert S['q256d-imagenet_test'][1] == 'hc112_imagenet_test' and not (set(cal) & set(ev)) and ev == {r[0]: r for r in json.load(open(f'{R}/runs/ok/sets_in256c.json'))}['q256-imagenet_test'][2]
def put(path, obj):
    assert not os.path.exists(path), path; json.dump(obj, open(path, 'w'))
put(f'{R}/runs/ido/sets8_in256d.json', [['devall-hc112_imagenet_test', 'hc112_imagenet_test', cal, 'none'], ['q256d-imagenet_test__hflip', 'hc112_imagenet_test', ev, 'hflip']])
put(f'{R}/runs/ido/probe_in256d.json', [['probe-bank4', 'hc112_imagenet_bank', [0, 1, 2, 3], 'none']])
# ---- (1b)
def pil(a): a = np.asarray(a); return Image.fromarray(a[..., 0] if (a.ndim == 3 and a.shape[2] == 1) else a)
def back(im, like): b = np.asarray(im, np.uint8); return b[..., None] if (like.ndim == 3 and like.shape[2] == 1) else b
def jpeg(im, q): buf = io.BytesIO(); im.save(buf, format='JPEG', quality=q); buf.seek(0); return Image.open(buf).copy()
def down(im, f):
    n = im.size[0]; m = int(round(n * f)); return im.resize((m, m), Image.BOX if f == 0.5 else Image.BICUBIC).resize((n, n), Image.BICUBIC)
COND = {'blur05': lambda im: im.filter(ImageFilter.GaussianBlur(0.5)), 'blur10': lambda im: im.filter(ImageFilter.GaussianBlur(1.0)), 'jpeg90': lambda im: jpeg(im, 90),
        'jpeg75': lambda im: jpeg(im, 75), 'jpeg50': lambda im: jpeg(im, 50), 'down75': lambda im: down(im, 0.75), 'down50': lambda im: down(im, 0.5)}
for t in BASE:
    e = [r for r in json.load(open(f'{R}/runs/ido/sets7_{t}.json')) if r[0] == f'test7-{IDD[t]}']; assert len(e) == 1 and e[0][3] == 'none'
    X = np.ascontiguousarray(np.load(f'{NPY}/{e[0][1]}.npy', mmap_mode='r')[np.asarray(e[0][2])]); rows = []
    for c, f in COND.items():
        Y = np.stack([back(f(pil(x)), x) for x in X]); assert Y.shape == X.shape and Y.dtype == np.uint8, (t, c, Y.shape, X.shape)
        p = f'{NPY}/dq_{t}_{c}.npy'; assert not os.path.exists(p), p; np.save(p, Y); rows.append([f'dq-{IDD[t]}__{c}', f'dq_{t}_{c}', list(range(len(Y))), 'none'])
        print(t, c, Y.shape, 'mean abs change', round(float(np.abs(Y.astype(np.int16) - X.astype(np.int16)).mean()), 2), flush=True)
    put(f'{R}/runs/ido/sets9_{t}.json', rows)
# ---- (2)
V = [f'{b}_{v}' for b in BASE for v in ('rinit', 'rshuf')] + [f'{b}_by_{m}' for b in ('edmc10', 'l6svhn', 'dpceleba') for m in ('edmc10', 'l6svhn', 'dpceleba') if m != b]
for vt in V:
    b = vt.split('_by_')[0] if '_by_' in vt else vt.rsplit('_', 1)[0]
    for d, f in (('runs/ok', 'sets'), ('runs/ido', 'sets7')):
        dst = f'{R}/{d}/{f}_{vt}.json'; assert not os.path.lexists(dst), dst; os.symlink(f'{R}/{d}/{f}_{b}.json', dst)
    put(f'{R}/runs/ido/probe_{vt}.json', [['probe-bank4', BANK[b], [0, 1, 2, 3], 'none']])
open(f'{R}/runs/ido/rfx_variants.txt', 'w').write('\n'.join(V) + '\n'); print(len(V), 'variant tags:', ' '.join(V)); print('MK_RFX_DONE')
