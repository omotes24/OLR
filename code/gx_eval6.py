# gx_eval6.py -- PROTOCOL_IDO.md I64: round 5 of the exploration on val2 (stored statistics of I59, runs/ido/gx/*_v.npz).
#   d2 = (a - mu)^T Sigma^-1 (a - mu),  l = (mu - mu_f)^T Sigma^-1 (a - mu),  Delta^2 = (mu - mu_f)^T Sigma^-1 (mu - mu_f),  p = number of blocks.
#   candidates: Z1 = (p - d2) / sqrt(2p) + l / Delta;  TP = -(d2 - max(l, 0)^2 / Delta^2)   (s = blocks up to A, a = all blocks).
#   for understanding: l alone, A(c) = -d2 + 2 c l, RK = log rank of -d2 + log rank of l.
#   ORACLE (not a method: reads the val2 OOD images with their set labels, two folds): linear / quadratic discriminants on the profile.
#   usage: python code/gx_eval6.py [seed]   -> runs/ido/final/gx_val2f_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']
ROWS = ['Z1s', 'Z1a', 'TPs', 'TPa', 'Ls', 'La', 'A2a', 'A4a', 'A8a', 'A16a', 'A8s', 'A16s', 'A64s', 'RKs', 'RKa']; FEAT = ['P', 'PL', 'PB']; ORA = ['lda_set', 'qda_set', 'lda_pool', 'qda_pool']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def rcov(C): cov = np.atleast_2d(np.cov(C.T)); return cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(C.shape[1])
def maha(V, mu, cov): d = V - mu; return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)
def logp(v, ref): r = np.sort(ref); return np.log((1 + np.searchsorted(r, v, side='right')) / (len(r) + 1))
def lda(A, B): w = np.linalg.solve(0.5 * (rcov(A) + rcov(B)), A.mean(0) - B.mean(0)); return lambda V: V @ w
def qda(A, B):
    (ma, Ca), (mb, Cb) = (A.mean(0), rcov(A)), (B.mean(0), rcov(B)); la, lb = np.linalg.slogdet(Ca)[1], np.linalg.slogdet(Cb)[1]; return lambda V: -0.5 * (maha(V, ma, Ca) + la) + 0.5 * (maha(V, mb, Cb) + lb)


def scorers(Z):
    iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part']); C = {}
    def q(n):
        if n not in C: V = Z[f'V__{n}']; C[n] = (V[:, 0, 0, 1:], V[:, 1, 0, 1:], V[:, 0, 1, 1:])
        return C[n]
    ca, cb, _ = q(ref); ok = ca.std(0) > 1e-9; idx = [h for h in range(L) if ok[h]]; cs = [h for h in idx if h <= iA]; S = {}; G = {}
    muT, covT = ca[:, idx].mean(0), rcov(ca[:, idx])
    for tg, cols in (('s', cs), ('a', idx)):
        U, F = ca[:, cols], cb[:, cols]; mu, mf = U.mean(0), F.mean(0); Ci = np.linalg.inv(rcov(U)); w = Ci @ (mu - mf); D = float(np.sqrt((mu - mf) @ w)); p = len(cols)
        d2 = lambda n, cols=cols, mu=mu, Ci=Ci: np.einsum('ij,jk,ik->i', q(n)[0][:, cols] - mu, Ci, q(n)[0][:, cols] - mu); el = lambda n, cols=cols, mu=mu, w=w: (q(n)[0][:, cols] - mu) @ w
        S['Z1' + tg] = lambda n, d2=d2, el=el, D=D, p=p: -d2(n) / np.sqrt(2.0 * p) + el(n) / D; S['TP' + tg] = lambda n, d2=d2, el=el, D=D: -(d2(n) - np.maximum(el(n), 0.0) ** 2 / D ** 2); S['L' + tg] = el
        for c in (2, 4, 8, 16, 64): S[f'A{c}{tg}'] = lambda n, d2=d2, el=el, c=c: -d2(n) + 2.0 * c * el(n)
        kT, kL = -d2(ref), el(ref); S['RK' + tg] = lambda n, d2=d2, el=el, kT=kT, kL=kL: logp(-d2(n), kT) + logp(el(n), kL); G[tg] = dict(p=p, Delta=D, c_Z1=float(np.sqrt(p / 2.0) / D))
    feat = {'P': lambda n: q(n)[0][:, idx], 'PL': lambda n: np.c_[q(n)[0][:, idx], q(n)[2][:, idx]], 'PB': lambda n: np.c_[q(n)[0][:, idx], q(n)[1][:, idx]]}
    return S, G, feat


def oracle(feat, idn, nat):
    out = {}
    for fn in FEAT:
        Xi = feat[fn](idn); Xo = {o: feat[fn](o) for o in nat}; R = {k: {o: [] for o in nat} for k in ORA}
        for f in (0, 1):
            tri, tei = Xi[f::2], Xi[1 - f::2]; pool = np.concatenate([Xo[o][f::2] for o in nat]); sl, sq = lda(tri, pool), qda(tri, pool)
            for o in nat:
                tro, teo = Xo[o][f::2], Xo[o][1 - f::2]; R['lda_pool'][o].append(auc(sl(tei), sl(teo))); R['qda_pool'][o].append(auc(sq(tei), sq(teo)))
                s1, s2 = lda(tri, tro), qda(tri, tro); R['lda_set'][o].append(auc(s1(tei), s1(teo))); R['qda_set'][o].append(auc(s2(tei), s2(teo)))
        out[fn] = {k: {o: float(np.mean(v)) for o, v in R[k].items()} for k in ORA}
    return out


OUT = {'seed': SEED, 'rows': {r: {} for r in ROWS}, 'info': {}, 'oracle': {}}
for t in BASE:
    Z = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_v.npz'); Ze = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_v.npz'); names = [str(n) for n in Z['names']]
    idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]; assert not any(n.startswith(('sel-', 'test')) for n in names)
    (S, G, feat), (Se, _, _) = scorers(Z), scorers(Ze); OUT['info'][t] = G
    for r in ROWS:
        OUT['rows'][r][t] = dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])),
                                 eq_vflip=float(auc(Se[r](idn), Se[r](vf))), degr_mean=float(np.mean([auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}')) for c in COND])), per={o: float(auc(S[r](idn), S[r](o))) for o in nat},
                                 eq_per={o: float(auc(Se[r](idn), Se[r](o))) for o in nat})
    OUT['oracle'][t] = oracle(feat, idn, nat); print(t, 'done', flush=True)
for r in ROWS: OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'eq_nat', 'eq_vflip', 'degr_mean')}
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2f_s{SEED}.json', 'w'), indent=1)
print(f'== noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down || degraded copies')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:5s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f} || {m['degr_mean']:5.1f}")
print('p, Delta and the c of Z1:', {t: {g: (v['p'], round(v['Delta'], 2), round(v['c_Z1'], 2)) for g, v in OUT['info'][t].items()} for t in BASE})
print('== ORACLE (not a method; val2 OOD images and labels, two folds): natural mean per ID -> five-ID mean')
for fn in FEAT:
    for k in ORA:
        v = [float(np.mean(list(OUT['oracle'][t][fn][k].values()))) for t in BASE]; print(f'  {fn:2s} {k:8s} ' + ' '.join(f'{x:6.2f}' for x in v) + f'  -> {np.mean(v):6.2f}')
for t in ('edmc10', 'in64'):
    for o in OUT['oracle'][t]['P']['lda_set']: print(f'   {t} {o[6:36]:30s} ' + '  '.join(f'{fn}:' + '/'.join(f'{OUT["oracle"][t][fn][k][o]:5.1f}' for k in ORA) for fn in FEAT))
print('GX_EVAL6_DONE')
