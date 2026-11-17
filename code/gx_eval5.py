# gx_eval5.py -- PROTOCOL_IDO.md I63: round 4 of the exploration on val2 (stored statistics of I59, runs/ido/gx/*_v.npz).
#   A(c) = -(a - mu*)^T Sigma^-1 (a - mu*),  mu* = mu + c (mu - mu_f): the Mahalanobis distance of the amplitude profile to the ID mean moved
#   c steps away from the mean profile of the upside-down calibration images (c = 0: T; large c: the Fisher direction L1).
#   A1s = the candidate (c = 1, blocks up to A, one stopped pass); the other rows are for understanding.  Development data only.
#   usage: python code/gx_eval5.py [seed]   -> runs/ido/final/gx_val2e_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']; ROWS = ['A1s', 'A1', 'Ts', 'L1s', 'A0.5s', 'A2s', 'A4s', 'A1sp']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def rcov(C): cov = np.atleast_2d(np.cov(C.T)); return cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(C.shape[1])
def maha(V, mu, cov): d = V - mu; return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)
def logp(v, ref): r = np.sort(ref); return np.log((1 + np.searchsorted(r, v, side='right')) / (len(r) + 1))


def scorers(Z):
    iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part'])
    def q(n): V = Z[f'V__{n}']; return V[:, 0, 0, 1:], V[:, 1, 0, 1:]
    ca, cb = q(ref); ok = ca.std(0) > 1e-9; idx = [h for h in range(L) if ok[h]]; cs = [h for h in idx if h <= iA]; S = {}
    muT, covT = ca[:, idx].mean(0), rcov(ca[:, idx])
    def anti(cols, c, pooled=False):
        U, F = ca[:, cols], cb[:, cols]; mu, mf = U.mean(0), F.mean(0); cov = rcov(U) if not pooled else 0.5 * (rcov(U) + rcov(F)); return lambda n: -maha(q(n)[0][:, cols], mu + c * (mu - mf), cov)
    S['A1s'] = anti(cs, 1.0); S['A1'] = anti(idx, 1.0); S['Ts'] = anti(cs, 0.0); S['A0.5s'] = anti(cs, 0.5); S['A2s'] = anti(cs, 2.0); S['A4s'] = anti(cs, 4.0); S['A1sp'] = anti(cs, 1.0, True)
    U, F = ca[:, cs], cb[:, cs]; w = np.linalg.solve(0.5 * (rcov(U) + rcov(F)), U.mean(0) - F.mean(0)); S['L1s'] = lambda n: q(n)[0][:, cs] @ w
    d = U.mean(0) - F.mean(0); return S, dict(n_blocks=len(cs), delta=float(np.sqrt(d @ np.linalg.solve(rcov(U), d))))


OUT = {'seed': SEED, 'rows': {r: {} for r in ROWS}, 'info': {}}
for t in BASE:
    Z = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_v.npz'); Ze = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_v.npz'); names = [str(n) for n in Z['names']]
    idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]; assert not any(n.startswith(('sel-', 'test')) for n in names)
    (S, G), (Se, _) = scorers(Z), scorers(Ze); OUT['info'][t] = G
    for r in ROWS:
        OUT['rows'][r][t] = dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), hflip=float(auc(S[r](idn), S[r](hf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])),
                                 eq_vflip=float(auc(Se[r](idn), Se[r](vf))), degr_mean=float(np.mean([auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}')) for c in COND])), per={o: float(auc(S[r](idn), S[r](o))) for o in nat},
                                 eq_per={o: float(auc(Se[r](idn), Se[r](o))) for o in nat})
for r in ROWS: OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'hflip', 'eq_nat', 'eq_vflip', 'degr_mean')}
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2e_s{SEED}.json', 'w'), indent=1)
print(f'== noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down || degraded copies')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:5s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f} || {m['degr_mean']:5.1f}")
print('blocks up to A, and the Mahalanobis distance between the upright and the upside-down calibration means:', {t: (OUT['info'][t]['n_blocks'], round(OUT['info'][t]['delta'], 2)) for t in BASE}); print('GX_EVAL5_DONE')
