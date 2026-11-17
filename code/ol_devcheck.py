# ol_devcheck.py -- PROTOCOL_IDO.md I68: check of code/ol_score.py against the development numbers, on val2 only (no test split).
#   runs/ido/ol/<tag>_s<seed>_dev.npz and <tag>_eq50_s<seed>_dev.npz (prefix fresh-) against row O8s of runs/ido/final/gx_val2h_s<seed>.json:
#   the natural-OOD mean AUROC per ID with the original images and under the identical history, and the upside-down ID set.
#   usage: python code/ol_devcheck.py [seed]
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
J = json.load(open(f'{ROOT}/runs/ido/final/gx_val2h_s{SEED}.json'))['rows']; worst = 0.0
for key, row in (('score', 'O8s'), ('sa', 'O8a')):
    for suf, kn, kv in (('', 'nat', 'vflip'), ('_eq50', 'eq_nat', 'eq_vflip')):
        got = []
        for t in BASE:
            F = np.load(f'{ROOT}/runs/ido/ol/{t}{suf}_s{SEED}_dev.npz'); names = [str(n) for n in F['names']]; assert not any(n.startswith(('test', 'sel-')) for n in names)
            idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]
            g = (float(np.mean([auc(F[f'{key}__{idn}'], F[f'{key}__{o}']) for o in nat])), float(auc(F[f'{key}__{idn}'], F[f'{key}__{vf}']))); w = (J[row][t][kn], J[row][t][kv]); got.append(g[0])
            worst = max(worst, abs(g[0] - w[0]), abs(g[1] - w[1])); print(f'  {row} {t + suf:14s} natural {g[0]:6.2f} (development {w[0]:6.2f})   upside-down {g[1]:6.2f} ({w[1]:6.2f})   blocks {int(F["n_blocks_used"])}')
        print(f'{row}{suf or "     "} five-ID natural mean {np.mean(got):6.2f} (development {J[row]["mean"][kn]:6.2f})')
print('largest difference from the development numbers:', round(worst, 3), 'OK' if worst < 0.05 else 'MISMATCH'); print('OL_DEVCHECK_DONE')
