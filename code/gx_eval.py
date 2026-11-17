# gx_eval.py -- PROTOCOL_IDO.md I59: the candidates of a simpler score on val2 (development; no test split, no sel- set is read).
#   Inputs: runs/ido/gx/<tag>_s<seed>_v.npz of code/gx_extract.py (<id>, <id>_eq50, <id>_rinit).  Every mean, covariance, regression and
#   rank comes from the calibration images (devall-<ID>) of the same condition.  AUROC with ID = positive (a higher score = more ID).
#   Rows: the candidates G1-G6, C1-C3, S1-S3 of I59, for the amplitude and for the log std.  (Public version: the yardstick rows
#   of the original run are left out; see protocols/PROTOCOL_OLR.md.)
#   Columns: natural-OOD mean per ID and the five-ID mean, the upside-down ID set (five-ID mean), the same under the identical history,
#   and the AUROC of the val2 ID images against their seven degraded copies (mean).   usage: python code/gx_eval.py [seed]
#   -> runs/ido/final/gx_val2_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']; KINDS = ['amp', 'lstd']
ROWS = ['G1', 'G2', 'G3', 'G4', 'C1', 'C2', 'C3', 'S1', 'S2', 'S3', 'G5', 'G6']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def maha_fit(C):
    mu = C.mean(0); cov = np.atleast_2d(np.cov(C.T)); cov = cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(len(mu)); return mu, cov
def maha(V, mu, cov): d = V - mu; return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)
def reg_fit(X, y):                                                     # ridge regression on standardised predictors; returns a function giving the residual
    m, s = X.mean(0), X.std(0) + 1e-12; Xs = (X - m) / s; b = np.linalg.solve(Xs.T @ Xs + 1e-3 * len(X) * np.eye(X.shape[1]), Xs.T @ (y - y.mean())); y0 = y.mean()
    return lambda Xn, yn: yn - (y0 + ((Xn - m) / s) @ b)
def logp(v, ref): r = np.sort(ref); return np.log((1 + np.searchsorted(r, v, side='right')) / (len(r) + 1))


def scorers(Z, k, Zt=None):
    """dict row -> function(set name) -> scores, for statistic kind k of one extraction (and the untrained twin Zt)"""
    part = [str(p) for p in Z['part']]; iA = int(Z['iA']); ref = str(Z['ref']); L = len(part)
    def q(n, F=Z):
        V = F[f'V__{n}']; return V[:, 0, k, 0], V[:, 0, k, 1:], V[:, 1, k, 1:]                      # a0 [N], a [N, L], b [N, L]
    c0, ca, cb = q(ref); ok = ca.std(0) > 1e-9; enc = np.array([p == 'enc' for p in part]) & ok; rest = (~np.array([p == 'enc' for p in part])) & ok; dec = np.array([p == 'dec' for p in part]) & ok
    muT, covT = maha_fit(ca[:, ok]); S = {}
    S['R'] = lambda n: q(n)[1][:, iA]; S['T'] = lambda n: -maha(q(n)[1][:, ok], muT, covT); S['D'] = lambda n: q(n)[1][:, iA] - q(n)[2][:, iA]
    S['G1'] = lambda n: q(n)[1][:, iA] - q(n)[0]
    f2 = reg_fit(c0[:, None], ca[:, iA]); S['G2'] = lambda n: f2(q(n)[0][:, None], q(n)[1][:, iA])
    P = lambda a0, a: np.c_[a0, a[:, enc]]; f3 = reg_fit(P(c0, ca), ca[:, iA]); S['G3'] = lambda n: f3(P(q(n)[0], q(n)[1]), q(n)[1][:, iA])
    sd3 = f3(P(c0, ca), ca[:, iA]).std(); S['G4'] = lambda n: -(S['G3'](n) / sd3) ** 2
    J = lambda a0, a: np.c_[a0, a[:, enc], a[:, rest]]; muJ, covJ = maha_fit(J(c0, ca)); m = 1 + int(enc.sum())
    S['C2'] = lambda n: -maha(J(q(n)[0], q(n)[1])[:, :m], muJ[:m], covJ[:m, :m]); S['C1'] = lambda n: -maha(J(q(n)[0], q(n)[1]), muJ, covJ) - S['C2'](n)
    fc = reg_fit(P(c0, ca), ca[:, dec].mean(1)); S['C3'] = lambda n: fc(P(q(n)[0], q(n)[1]), q(n)[1][:, dec].mean(1))
    K = lambda a0, a, b: np.c_[a0, a[:, iA], b[:, iA]]; muK, covK = maha_fit(K(c0, ca, cb))
    S['S1'] = lambda n: -(maha(K(*q(n)), muK, covK) - (q(n)[0] - muK[0]) ** 2 / covK[0, 0])
    z = lambda f: (lambda n, mu=f(ref).mean(), sd=f(ref).std() + 1e-12: (f(n) - mu) / sd); zD, z2, z3 = z(S['D']), z(S['G2']), z(S['G3'])
    S['S2'] = lambda n: z2(n) + zD(n); S['S3'] = lambda n: z3(n) + zD(n)
    if Zt is not None:
        assert int(Zt['iA']) == iA and [str(h) for h in Zt['hooks']] == [str(h) for h in Z['hooks']]
        tw = lambda n: q(n, Zt)[1][:, iA]; S['G5'] = lambda n: q(n)[1][:, iA] - tw(n); f6 = reg_fit(tw(ref)[:, None], ca[:, iA]); S['G6'] = lambda n: f6(tw(n)[:, None], q(n)[1][:, iA])
    return S


OUT = {'seed': SEED, 'rows': ROWS, 'kinds': {}}
for k, kn in enumerate(KINDS):
    RES = {r: {} for r in ROWS}
    for t in BASE:
        Z = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_v.npz'); Ze = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_v.npz'); pt = f'{ROOT}/runs/ido/gx/{t}_rinit_s0_v.npz'; Zt = np.load(pt) if (SEED == 0 and os.path.exists(pt)) else None
        names = [str(n) for n in Z['names']]; idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]
        assert idn in names and vf in names and not any(n.startswith(('sel-', 'test')) for n in names) and [str(n) for n in Ze['names']][1:] != []
        S, Se = scorers(Z, k, Zt), scorers(Ze, k)
        for r in ROWS:
            if r not in S: RES[r][t] = None; continue
            d = dict(per={o: float(auc(S[r](idn), S[r](o))) for o in nat}, vflip=float(auc(S[r](idn), S[r](vf)))); d['nat'] = float(np.mean(list(d['per'].values())))
            if r in Se: d['eq_nat'] = float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])); d['eq_vflip'] = float(auc(Se[r](idn), Se[r](vf)))
            if r not in ('G5', 'G6'): d['degr'] = {c: float(auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}'))) for c in COND}; d['degr_mean'] = float(np.mean(list(d['degr'].values())))
            RES[r][t] = d
    for r in ROWS:
        v = [RES[r][t] for t in BASE]
        if any(x is None for x in v): RES[r]['mean'] = None; continue
        RES[r]['mean'] = {key: float(np.mean([x[key] for x in v])) for key in ('nat', 'vflip', 'eq_nat', 'eq_vflip', 'degr_mean') if all(key in x for x in v)}
    OUT['kinds'][kn] = RES
    print(f'== {kn}, noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down || degraded copies (mean of 7)')
    for r in ROWS:
        if RES[r]['mean'] is None: print(f'  {r:4s} n/a'); continue
        m = RES[r]['mean']; print(f'  {r:4s} ' + ' '.join(f"{RES[r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || " + (f"{m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f}" if 'eq_nat' in m else '   n/a        ')
                             + ' || ' + (f"{m['degr_mean']:5.1f}" if 'degr_mean' in m else ' n/a'))
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2_s{SEED}.json', 'w'), indent=1); print('GX_EVAL_DONE')
