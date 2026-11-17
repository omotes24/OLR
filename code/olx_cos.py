# olx_cos.py -- PROTOCOL_IDO.md I72 (X8): a diagnostic on val2 (noise seed 0), not a method.  The cosine, in the metric of the calibration
#   covariance S, between the direction of OL, P (m - m_o) on the blocks up to A (m_o = mean over the seven other orientations), and the
#   linear discriminant direction between the val2 ID images and the pooled val2 OOD sets of the ID (the diagnostic of I66), per ID; also the
#   mean over the OOD sets of the cosine with the direction fitted on each set alone.  Stored statistics only (runs/ido/gx/*_s0_{v,n,n2}.npz).
#   usage: python code/olx_cos.py   -> runs/ido/final/olx_cos.json
import os, json
import numpy as np
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; O7 = ['v', 'h', 'r180', 'r90', 'r270', 't1', 't2']
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
def rcov(C): cov = np.atleast_2d(np.cov(C.T)); return cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(C.shape[1])
OUT = {}
for t in BASE:
    ld = lambda lab: np.load(f'{ROOT}/runs/ido/gx/{t}_s0_{lab}.npz'); Z, N, N2 = ld('v'), ld('n'), ld('n2')
    iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part']); ca = Z[f'V__{ref}'][:, 0, 0, 1:]; ok = ca.std(0) > 1e-9; cs = [h for h in range(L) if ok[h] and h <= iA]
    K = {'v': Z[f'V__{ref}'][:, 1, 0, 1:], 'h': N2[f'V__{ref}__hf'][:, 0, 0, 1:], 'r180': N2[f'V__{ref}__hf'][:, 1, 0, 1:], 'r90': N[f'V__{ref}__r90'][:, 0, 0, 1:],
         'r270': N[f'V__{ref}__r270'][:, 0, 0, 1:], 't1': N[f'V__{ref}__r90'][:, 1, 0, 1:], 't2': N[f'V__{ref}__r270'][:, 1, 0, 1:]}
    U = ca[:, cs]; mu = U.mean(0); Sg = rcov(U); Ci = np.linalg.inv(Sg); mo = np.mean([K[k][:, cs].mean(0) for k in O7], 0); w = Ci @ (mu - mo)
    names = [str(n) for n in Z['names']]; idn = f'fresh-{IDD[t]}'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, idn + '__vflip', idn + '__hflip') and 'synth' not in n]
    assert not any(n.startswith(('sel-', 'test')) for n in names)
    q = lambda n: Z[f'V__{n}'][:, 0, 0, 1:][:, cs]; A = q(idn)
    lda = lambda B: np.linalg.solve(0.5 * (rcov(A) + rcov(B)), A.mean(0) - B.mean(0)); cz = lambda x, y: float((x @ Sg @ y) / np.sqrt((x @ Sg @ x) * (y @ Sg @ y)))
    wo = lda(np.concatenate([q(o) for o in nat])); per = {o: cz(lda(q(o)), w) for o in nat}
    OUT[t] = dict(cos_pooled=cz(wo, w), cos_per_set_mean=float(np.mean(list(per.values()))), cos_per_set=per, p=len(cs))
    print(t, 'p', len(cs), 'cos(OL direction, pooled OOD direction) %.3f' % OUT[t]['cos_pooled'], 'mean over the sets %.3f' % OUT[t]['cos_per_set_mean'], flush=True)
json.dump(OUT, open(f'{ROOT}/runs/ido/final/olx_cos.json', 'w'), indent=1); print('OLX_COS_DONE')
