# gx_eval4.py -- PROTOCOL_IDO.md I62: round 3 of the exploration on val2 with the stored statistics of I59 (runs/ido/gx/*_v.npz).
#   Fewer numbers in one metric (N2, N3, ST, STs) and the direction of the loss of response learned from the upside-down copies of the
#   calibration images (L1, L1s, LQ, L1d, LN).  Development data only; ID = positive.   usage: python code/gx_eval4.py [seed] [out label]
#   -> runs/ido/final/gx_val2d_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']; ROWS = ['N2', 'N3', 'ST', 'STs', 'L1', 'L1s', 'LQ', 'L1d', 'LN']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def maha_fit(C):
    C = C.reshape(len(C), -1); mu = C.mean(0); cov = np.atleast_2d(np.cov(C.T)); cov = cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(len(mu)); return mu, cov
def maha(V, mu, cov): d = V.reshape(len(V), -1) - mu; return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)
def logp(v, ref): r = np.sort(ref); return np.log((1 + np.searchsorted(r, v, side='right')) / (len(r) + 1))


def scorers(Z):
    part = [str(p) for p in Z['part']]; res = [int(r) for r in Z['res']]; iA = int(Z['iA']); ref = str(Z['ref']); L = len(part)
    def q(n): V = Z[f'V__{n}']; return V[:, 0, 0, 1:], V[:, 1, 0, 1:]                                # amplitude of the image a and of its upside-down copy b
    ca, cb = q(ref); ok = ca.std(0) > 1e-9; idx = [h for h in range(L) if ok[h]]; i0, i1 = idx[0], idx[-1]; S = {}
    muT, covT = maha_fit(ca[:, ok])
    def mk(cols):
        f = lambda n: q(n)[0][:, cols]; mu, cov = maha_fit(f(ref)); return lambda n: -maha(f(n), mu, cov)
    S['N2'] = mk([i0, iA]); S['N3'] = mk([i0, iA, i1])
    stages, cur = [], None
    for h in idx:
        key = (part[h], res[h])
        if key != cur: stages.append([]); cur = key
        stages[-1].append(h)
    def stg(groups):
        f = lambda n: np.stack([q(n)[0][:, g].mean(1) for g in groups], 1); mu, cov = maha_fit(f(ref)); return lambda n: -maha(f(n), mu, cov)
    S['ST'] = stg(stages); S['STs'] = stg([g for g in stages if g[-1] < iA] + [[iA]])
    def fisher(cols):
        U, F = ca[:, cols], cb[:, cols]; Sw = 0.5 * (np.cov(U.T) + np.cov(F.T)); Sw = Sw + 1e-3 * np.mean(np.diag(Sw)) * np.eye(len(cols)); return np.linalg.solve(Sw, U.mean(0) - F.mean(0))
    w = fisher(idx); S['L1'] = lambda n: q(n)[0][:, idx] @ w; cs = [h for h in idx if h <= iA]; ws = fisher(cs); S['L1s'] = lambda n: q(n)[0][:, cs] @ ws
    muF, covF = maha_fit(cb[:, ok]); S['LQ'] = lambda n: -maha(q(n)[0][:, ok], muT, covT) + maha(q(n)[0][:, ok], muF, covF)
    S['L1d'] = lambda n: (q(n)[0][:, idx] - q(n)[1][:, idx]) @ w
    g = lambda n: np.c_[q(n)[0][:, i0], q(n)[0][:, idx] @ w]; muG, covG = maha_fit(g(ref)); S['LN'] = lambda n: -maha(g(n), muG, covG)
    return S, dict(stages=[f'{part[g[0]][:1]}{res[g[0]]}x{len(g)}' for g in stages], w=[float(v) for v in w], first=i0, last=i1, A=iA)


OUT = {'seed': SEED, 'rows': {r: {} for r in ROWS}, 'info': {}}
for t in BASE:
    Z = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_v.npz'); Ze = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_v.npz'); names = [str(n) for n in Z['names']]
    idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]; assert not any(n.startswith(('sel-', 'test')) for n in names)
    (S, G), (Se, _) = scorers(Z), scorers(Ze); OUT['info'][t] = G
    for r in ROWS:
        OUT['rows'][r][t] = dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])), eq_vflip=float(auc(Se[r](idn), Se[r](vf))),
                                 degr_mean=float(np.mean([auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}')) for c in COND])), per={o: float(auc(S[r](idn), S[r](o))) for o in nat})
for r in ROWS: OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'eq_nat', 'eq_vflip', 'degr_mean')}
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2d_s{SEED}.json', 'w'), indent=1)
print(f'== noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down || degraded copies')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:4s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f} || {m['degr_mean']:5.1f}")
print('stages:', {t: OUT['info'][t]['stages'] for t in BASE}); print('GX_EVAL4_DONE')
