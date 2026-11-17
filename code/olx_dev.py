# olx_dev.py -- PROTOCOL_IDO.md I70 (X6): development analyses of the orientation likelihood-ratio score on val2, from the stored block
#   statistics (runs/ido/gx/<tag>_s<seed>_{v,n,n2}.npz; noise seeds 0-2).  val2 and the val1 ID images only; no test split.
#   score = -d2 + 2 c lbar;  d2 = (a - m)^T C^-1 (a - m);  lbar = (m - m_o)^T C^-1 (a - m);  m_o = mean over a set of orientations.
#   rows: c;  the set of orientations (c = 8);  all blocks;  the log std in place of the amplitude;  the number of calibration images;
#   the versions without a constant (posterior of "upright" PK, PK - d2/2, PK - d2 in the span / 2).
#   reported: natural-OOD mean per ID and five-ID mean, the upside-down ID set, the same under the identical history.  -> runs/ido/final/olx_dev.json
import os, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEEDS = (0, 1, 2)
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
O7 = ['v', 'h', 'r180', 'r90', 'r270', 't1', 't2']; CS = [0, 1, 2, 4, 8, 16, 32, 64]; NCAL = [50, 100, 200, 500, 1000]
OSETS = {'v': ['v'], 'flips': ['v', 'h', 'r180'], 'quarter': ['r90', 'r270', 't1', 't2'], 'no_mirror': ['v', 'r180', 'r90', 'r270', 't1', 't2']}
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def rcov(C): cov = np.atleast_2d(np.cov(C.T)); return cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(C.shape[1])


def scorers(Z, N, N2):
    iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part']); C = {}
    def q(n, kind=0):
        if (n, kind) not in C: C[(n, kind)] = Z[f'V__{n}'][:, 0, kind, 1:]
        return C[(n, kind)]
    def cal(kind):
        K = {'id': Z[f'V__{ref}'][:, 0, kind, 1:], 'v': Z[f'V__{ref}'][:, 1, kind, 1:], 'h': N2[f'V__{ref}__hf'][:, 0, kind, 1:], 'r180': N2[f'V__{ref}__hf'][:, 1, kind, 1:], 'r90': N[f'V__{ref}__r90'][:, 0, kind, 1:],
             'r270': N[f'V__{ref}__r270'][:, 0, kind, 1:], 't1': N[f'V__{ref}__r90'][:, 1, kind, 1:], 't2': N[f'V__{ref}__r270'][:, 1, kind, 1:]}; return K
    K0, K1 = cal(0), cal(1); ok = K0['id'].std(0) > 1e-9; idx = [h for h in range(L) if ok[h]]; cs = [h for h in idx if h <= iA]; ok1 = K1['id'].std(0) > 1e-9; cs1 = [h for h in range(L) if ok1[h] and h <= iA]; S = {}
    def fam(K, cols, kind, ors, n=None):
        U = K['id'][:n, cols]; mu = U.mean(0); Ci = np.linalg.inv(rcov(U)); DL = {k: mu - K[k][:n, cols].mean(0) for k in ors}; W = {k: Ci @ DL[k] for k in ors}; D2 = {k: float(DL[k] @ W[k]) for k in ors}; wb = np.mean([W[k] for k in ors], 0)
        d2 = lambda m: np.einsum('ij,jk,ik->i', q(m, kind)[:, cols] - mu, Ci, q(m, kind)[:, cols] - mu); lb = lambda m: (q(m, kind)[:, cols] - mu) @ wb; el = lambda m, k: (q(m, kind)[:, cols] - mu) @ W[k]
        return d2, lb, el, D2, W, rcov(U)
    d2, lb, el, D2, W, Sg = fam(K0, cs, 0, O7)
    for c in CS: S[f'c{c}'] = lambda m, c=c: -d2(m) + 2.0 * c * lb(m)
    S['cinf'] = lb
    for nm, ors in OSETS.items(): a2, al = fam(K0, cs, 0, ors)[:2]; S['o_' + nm] = lambda m, a2=a2, al=al: -a2(m) + 16.0 * al(m)
    b2, bl = fam(K0, idx, 0, O7)[:2]; S['allblocks'] = lambda m: -b2(m) + 16.0 * bl(m)
    e2, e1 = fam(K1, cs1, 1, O7)[:2]; S['logstd'] = lambda m: -e2(m) + 16.0 * e1(m)
    for n in NCAL: f2, fl = fam(K0, cs, 0, O7, n)[:2]; S[f'n{n}'] = lambda m, f2=f2, fl=fl: -f2(m) + 16.0 * fl(m)
    pk = lambda m: -np.logaddexp.reduce(np.c_[np.zeros(len(q(m))), np.stack([-(el(m, k) + 0.5 * D2[k]) for k in O7], 1)], axis=1)
    Wm = np.stack([W[k] for k in O7], 1); Gi = np.linalg.pinv(Wm.T @ Sg @ Wm, rcond=1e-6, hermitian=True); mu0 = K0['id'][:, cs].mean(0)
    def d2sub(m): y = (q(m)[:, cs] - mu0) @ Wm; return np.einsum('ij,jk,ik->i', y, Gi, y)
    S['pk'] = pk; S['pk_d2'] = lambda m: pk(m) - 0.5 * d2(m); S['pk_d2sub'] = lambda m: pk(m) - 0.5 * d2sub(m)
    return S, dict(p=len(cs), Delta={k: float(np.sqrt(D2[k])) for k in O7}, n_cal=int(len(K0['id'])))


ROWS = None; ACC = {}; INFO = {}
for sd in SEEDS:
    for t in BASE:
        ld = lambda tag, lab: np.load(f'{ROOT}/runs/ido/gx/{tag}_s{sd}_{lab}.npz'); Z, Ze = ld(t, 'v'), ld(t + '_eq50', 'v'); names = [str(n) for n in Z['names']]; assert not any(n.startswith(('sel-', 'test')) for n in names)
        idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]
        (S, G), (Se, _) = scorers(Z, ld(t, 'n'), ld(t, 'n2')), scorers(Ze, ld(t + '_eq50', 'n'), ld(t + '_eq50', 'n2')); ROWS = list(S); INFO[t] = G
        for r in ROWS:
            ACC.setdefault(r, {}).setdefault(t, []).append(dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])), eq_vflip=float(auc(Se[r](idn), Se[r](vf)))))
    print('seed', sd, 'done', flush=True)
KS = ('nat', 'vflip', 'eq_nat', 'eq_vflip'); OUT = {'rows': {}, 'info': INFO, 'seeds': list(SEEDS)}
for r in ROWS:
    OUT['rows'][r] = {t: {k: float(np.mean([e[k] for e in ACC[r][t]])) for k in KS} for t in BASE}; OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in KS}
    OUT['rows'][r]['seed_means'] = [float(np.mean([ACC[r][t][i]['nat'] for t in BASE])) for i in range(len(SEEDS))]
json.dump(OUT, open(f'{ROOT}/runs/ido/final/olx_dev.json', 'w'), indent=1)
print('== val2, mean of noise seeds 0-2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:10s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f}")
print('OLX_DEV_DONE')
