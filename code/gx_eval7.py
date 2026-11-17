# gx_eval7.py -- PROTOCOL_IDO.md I65: round 6 of the exploration on val2.  The response direction from several kinds of rearranged copies of
#   the calibration images (runs/ido/gx/<tag>_s<seed>_n.npz of code/mk_gx_neg.py; the val2 statistics from runs/ido/gx/<tag>_s<seed>_v.npz).
#   d2 = (a - mu)^T Sigma^-1 (a - mu);  l_k = (mu - mu_k)^T Sigma^-1 (a - mu), k in v r90 r270 j2 j4 ch;  lbar = mean of the six.
#   candidates: M8 = -d2 + 16 lbar, LM = lbar, Q8 = -d2 + 16 min_k l_k, PK = log posterior of "untouched" among the seven Gaussians.
#   usage: python code/gx_eval7.py [seed]   -> runs/ido/final/gx_val2g_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']; SIX = ['v', 'r90', 'r270', 'j2', 'j4', 'ch']; ALLK = SIX + ['t1', 't2']
ROWS = ['M8s', 'M8a', 'LMs', 'LMa', 'Q8s', 'PKs', 'A8s', 'MG8s', 'MJ8s', 'M4s', 'M16s'] + ['L' + k for k in ALLK]
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def rcov(C): cov = np.atleast_2d(np.cov(C.T)); return cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(C.shape[1])
def maha(V, mu, cov): d = V - mu; return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)
def logp(v, ref): r = np.sort(ref); return np.log((1 + np.searchsorted(r, v, side='right')) / (len(r) + 1))


def scorers(Z, N, idn=None, nat=None):
    iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part']); C = {}; assert str(N['ref']) == ref
    def q(n):
        if n not in C: V = Z[f'V__{n}']; C[n] = (V[:, 0, 0, 1:], V[:, 1, 0, 1:])
        return C[n]
    ca, cb = q(ref); na = N[f'V__{ref}']; dev = float(np.abs(na[:, 0, 0, 1:] - ca).max()); ok = ca.std(0) > 1e-9; idx = [h for h in range(L) if ok[h]]; cs = [h for h in idx if h <= iA]; S = {}; G = {'recheck': dev}
    K = {'v': cb, 't1': N[f'V__{ref}__r90'][:, 1, 0, 1:], 't2': N[f'V__{ref}__r270'][:, 1, 0, 1:]}; K.update({k: N[f'V__{ref}__{k}'][:, 0, 0, 1:] for k in SIX[1:]})
    muT, covT = ca[:, idx].mean(0), rcov(ca[:, idx])
    for tg, cols in (('s', cs), ('a', idx)):
        U = ca[:, cols]; mu = U.mean(0); Sg = rcov(U); Ci = np.linalg.inv(Sg); W = {k: Ci @ (mu - K[k][:, cols].mean(0)) for k in ALLK}; D2 = {k: float((mu - K[k][:, cols].mean(0)) @ W[k]) for k in ALLK}
        d2 = lambda n, cols=cols, mu=mu, Ci=Ci: np.einsum('ij,jk,ik->i', q(n)[0][:, cols] - mu, Ci, q(n)[0][:, cols] - mu); el = lambda n, k, cols=cols, mu=mu, W=W: (q(n)[0][:, cols] - mu) @ W[k]
        bar = lambda n, ks=tuple(SIX), el=el: np.mean([el(n, k) for k in ks], 0)
        S['M8' + tg] = lambda n, d2=d2, bar=bar: -d2(n) + 16.0 * bar(n); S['LM' + tg] = lambda n, bar=bar: bar(n); G[tg] = dict(p=len(cols), Delta={k: float(np.sqrt(D2[k])) for k in ALLK})
        if tg == 's':
            S['Q8s'] = lambda n, d2=d2, el=el: -d2(n) + 16.0 * np.min([el(n, k) for k in SIX], 0)
            S['PKs'] = lambda n, el=el, D2=D2: -np.logaddexp.reduce(np.c_[np.zeros(len(q(n)[0])), np.stack([-(el(n, k) + 0.5 * D2[k]) for k in SIX], 1)], axis=1)
            S['A8s'] = lambda n, d2=d2, el=el: -d2(n) + 16.0 * el(n, 'v'); S['MG8s'] = lambda n, d2=d2, bar=bar: -d2(n) + 16.0 * bar(n, ('v', 'r90', 'r270')); S['MJ8s'] = lambda n, d2=d2, bar=bar: -d2(n) + 16.0 * bar(n, ('j2', 'j4', 'ch'))
            S['M4s'] = lambda n, d2=d2, bar=bar: -d2(n) + 8.0 * bar(n); S['M16s'] = lambda n, d2=d2, bar=bar: -d2(n) + 32.0 * bar(n)
            for k in ALLK: S['L' + k] = lambda n, k=k, el=el: el(n, k)
            if nat is not None:      # diagnostic, not a method: the direction between the val2 ID images and the pooled val2 OOD sets, and its cosine with each copy direction
                A = q(idn)[0][:, cols]; B = np.concatenate([q(o)[0][:, cols] for o in nat]); wo = np.linalg.solve(0.5 * (rcov(A) + rcov(B)), A.mean(0) - B.mean(0)); cz = lambda x, y: float((x @ Sg @ y) / np.sqrt((x @ Sg @ x) * (y @ Sg @ y)))
                G['cos_oracle'] = {k: cz(wo, W[k]) for k in ALLK}; G['cos_oracle']['six'] = cz(wo, np.mean([W[k] for k in SIX], 0)); G['cos_v'] = {k: cz(W['v'], W[k]) for k in ALLK}
    return S, G


OUT = {'seed': SEED, 'rows': {r: {} for r in ROWS}, 'info': {}}
for t in BASE:
    Z = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_v.npz'); Ze = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_v.npz'); N = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_n.npz'); Ne = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_n.npz')
    names = [str(n) for n in Z['names']]; idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]
    assert not any(n.startswith(('sel-', 'test', 'fresh-')) for n in [str(x) for x in N['names']]) and not any(n.startswith(('sel-', 'test')) for n in names)
    (S, G), (Se, Ge) = scorers(Z, N, idn, nat), scorers(Ze, Ne); OUT['info'][t] = dict(G, recheck_eq=Ge['recheck'])
    for r in ROWS:
        OUT['rows'][r][t] = dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])),
                                 eq_vflip=float(auc(Se[r](idn), Se[r](vf))), degr_mean=float(np.mean([auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}')) for c in COND])), per={o: float(auc(S[r](idn), S[r](o))) for o in nat},
                                 eq_per={o: float(auc(Se[r](idn), Se[r](o))) for o in nat})
    print(t, 'done; largest difference of the calibration statistics between the two extractions', round(G['recheck'], 5), round(Ge['recheck'], 5), flush=True)
for r in ROWS: OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'eq_nat', 'eq_vflip', 'degr_mean')}
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2g_s{SEED}.json', 'w'), indent=1)
print(f'== noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down || degraded copies')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:6s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f} || {m['degr_mean']:5.1f}")
for t in BASE:
    g = OUT['info'][t]; print(f'  {t:9s} Delta(s): ' + ' '.join(f"{k} {g['s']['Delta'][k]:.2f}" for k in ALLK) + ' | cos with the oracle direction: ' + ' '.join(f"{k} {v:+.2f}" for k, v in g['cos_oracle'].items()) + ' | cos with v: ' + ' '.join(f"{k} {v:+.2f}" for k, v in g['cos_v'].items()))
print('GX_EVAL7_DONE')
