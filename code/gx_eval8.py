# gx_eval8.py -- PROTOCOL_IDO.md I66: round 7 of the exploration on val2.  The response direction from ALL seven other orientations of the
#   square (v upside-down, h mirrored, r180, r90, r270, t1, t2 diagonal reflections) of the calibration images.
#   d2 = (a - mu)^T Sigma^-1 (a - mu);  l_k = (mu - mu_k)^T Sigma^-1 (a - mu);  Delta_k^2 = (mu - mu_k)^T Sigma^-1 (mu - mu_k);  lbar = mean_k l_k.
#   candidates: O8 = -d2 + 16 lbar;  PKo = -log(1 + sum_k exp(-(l_k + Delta_k^2 / 2)));  JFo = PKo - d2 / 2;  JRo = PKo - (d2 within the span of the seven directions) / 2.
#   usage: python code/gx_eval8.py [seed]   -> runs/ido/final/gx_val2h_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']; O7 = ['v', 'h', 'r180', 'r90', 'r270', 't1', 't2']; O5 = ['v', 'r90', 'r270', 't1', 't2']
ROWS = ['O8s', 'O8a', 'PKo', 'PKoa', 'JFo', 'JRo', 'LGo', 'O4s', 'O16s', 'O8s5', 'MG8s', 'Lh', 'Lr180']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def rcov(C): cov = np.atleast_2d(np.cov(C.T)); return cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(C.shape[1])
def maha(V, mu, cov): d = V - mu; return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)
def logp(v, ref): r = np.sort(ref); return np.log((1 + np.searchsorted(r, v, side='right')) / (len(r) + 1))


def scorers(Z, N, N2):
    iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part']); C = {}; assert str(N['ref']) == ref and str(N2['ref']) == ref
    def q(n):
        if n not in C: V = Z[f'V__{n}']; C[n] = (V[:, 0, 0, 1:], V[:, 1, 0, 1:])
        return C[n]
    ca, cb = q(ref); dev = max(float(np.abs(M[f'V__{ref}'][:, 0, 0, 1:] - ca).max()) for M in (N, N2)); ok = ca.std(0) > 1e-9; idx = [h for h in range(L) if ok[h]]; cs = [h for h in idx if h <= iA]; S = {}; G = {'recheck': dev}
    K = {'v': cb, 'h': N2[f'V__{ref}__hf'][:, 0, 0, 1:], 'r180': N2[f'V__{ref}__hf'][:, 1, 0, 1:], 'r90': N[f'V__{ref}__r90'][:, 0, 0, 1:], 'r270': N[f'V__{ref}__r270'][:, 0, 0, 1:], 't1': N[f'V__{ref}__r90'][:, 1, 0, 1:], 't2': N[f'V__{ref}__r270'][:, 1, 0, 1:]}
    muT, covT = ca[:, idx].mean(0), rcov(ca[:, idx])
    for tg, cols in (('s', cs), ('a', idx)):
        U = ca[:, cols]; mu = U.mean(0); Sg = rcov(U); Ci = np.linalg.inv(Sg); DL = {k: mu - K[k][:, cols].mean(0) for k in O7}; W = {k: Ci @ DL[k] for k in O7}; D2 = {k: float(DL[k] @ W[k]) for k in O7}
        d2 = lambda n, cols=cols, mu=mu, Ci=Ci: np.einsum('ij,jk,ik->i', q(n)[0][:, cols] - mu, Ci, q(n)[0][:, cols] - mu); el = lambda n, k, cols=cols, mu=mu, W=W: (q(n)[0][:, cols] - mu) @ W[k]
        bar = lambda n, ks=tuple(O7), el=el: np.mean([el(n, k) for k in ks], 0)
        pk = lambda n, el=el, D2=D2: -np.logaddexp.reduce(np.c_[np.zeros(len(q(n)[0])), np.stack([-(el(n, k) + 0.5 * D2[k]) for k in O7], 1)], axis=1)
        S['O8' + tg] = lambda n, d2=d2, bar=bar: -d2(n) + 16.0 * bar(n); S['PKo' + ('' if tg == 's' else 'a')] = pk; G[tg] = dict(p=len(cols), Delta={k: float(np.sqrt(D2[k])) for k in O7})
        if tg == 's':
            Wm = np.stack([W[k] for k in O7], 1); Gi = np.linalg.pinv(Wm.T @ Sg @ Wm, rcond=1e-6, hermitian=True); G['rank'] = int(np.linalg.matrix_rank(Wm.T @ Sg @ Wm, tol=1e-6 * np.linalg.norm(Wm.T @ Sg @ Wm, 2), hermitian=True))
            def d2sub(n, cols=cols, mu=mu, Wm=Wm, Gi=Gi): y = (q(n)[0][:, cols] - mu) @ Wm; return np.einsum('ij,jk,ik->i', y, Gi, y)
            S['JFo'] = lambda n, pk=pk, d2=d2: pk(n) - 0.5 * d2(n); S['JRo'] = lambda n, pk=pk, d2sub=d2sub: pk(n) - 0.5 * d2sub(n); S['LGo'] = lambda n, bar=bar: bar(n)
            S['O4s'] = lambda n, d2=d2, bar=bar: -d2(n) + 8.0 * bar(n); S['O16s'] = lambda n, d2=d2, bar=bar: -d2(n) + 32.0 * bar(n); S['O8s5'] = lambda n, d2=d2, bar=bar: -d2(n) + 16.0 * bar(n, tuple(O5))
            S['MG8s'] = lambda n, d2=d2, bar=bar: -d2(n) + 16.0 * bar(n, ('v', 'r90', 'r270')); S['Lh'] = lambda n, el=el: el(n, 'h'); S['Lr180'] = lambda n, el=el: el(n, 'r180')
    return S, G


OUT = {'seed': SEED, 'rows': {r: {} for r in ROWS}, 'info': {}}
for t in BASE:
    ld = lambda tag, lab: np.load(f'{ROOT}/runs/ido/gx/{tag}_s{SEED}_{lab}.npz'); Z, Ze, N, Ne, N2, N2e = ld(t, 'v'), ld(t + '_eq50', 'v'), ld(t, 'n'), ld(t + '_eq50', 'n'), ld(t, 'n2'), ld(t + '_eq50', 'n2')
    names = [str(n) for n in Z['names']]; idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]
    assert not any(str(x).startswith(('sel-', 'test', 'fresh-')) for M in (N, N2, Ne, N2e) for x in M['names']) and not any(n.startswith(('sel-', 'test')) for n in names)
    (S, G), (Se, Ge) = scorers(Z, N, N2), scorers(Ze, Ne, N2e); OUT['info'][t] = dict(G, recheck_eq=Ge['recheck'])
    for r in ROWS:
        OUT['rows'][r][t] = dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), hflip=float(auc(S[r](idn), S[r](hf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])),
                                 eq_vflip=float(auc(Se[r](idn), Se[r](vf))), degr_mean=float(np.mean([auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}')) for c in COND])), per={o: float(auc(S[r](idn), S[r](o))) for o in nat},
                                 eq_per={o: float(auc(Se[r](idn), Se[r](o))) for o in nat})
    print(t, 'done; recheck', round(G['recheck'], 5), round(Ge['recheck'], 5), 'rank', G['rank'], flush=True)
for r in ROWS: OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'hflip', 'eq_nat', 'eq_vflip', 'degr_mean')}
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2h_s{SEED}.json', 'w'), indent=1)
print(f'== noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down, mirrored || identical history: mean | upside-down || degraded copies')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:6s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} {m['hflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f} || {m['degr_mean']:5.1f}")
for t in BASE: g = OUT['info'][t]; print(f'  {t:9s} Delta(s): ' + ' '.join(f"{k} {g['s']['Delta'][k]:.2f}" for k in O7) + f"  rank {g['rank']}")
print('GX_EVAL8_DONE')
