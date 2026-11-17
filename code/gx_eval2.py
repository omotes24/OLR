# gx_eval2.py -- PROTOCOL_IDO.md I61 part I: round 2 of the exploration on val2 with the stored statistics of I59 (runs/ido/gx/*_v.npz).
#   One-formula candidates that differ in the METRIC (M1-M5), read the SHAPE of the response instead of its size (K1-K4), or put both
#   orientations in one formula (P1-P3); diagnostics per block and per principal component.  Development data only; ID = positive.
#   usage: python code/gx_eval2.py [seed]   -> runs/ido/final/gx_val2b_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']
ROWS = ['M1', 'M2', 'M3', 'M4', 'M5', 'K1', 'K2', 'K3', 'K4', 'P1', 'P2', 'P3']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def maha_fit(C):
    mu = C.mean(0); cov = np.atleast_2d(np.cov(C.T)); cov = cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(len(mu)); return mu, cov
def maha(V, mu, cov): d = V - mu; return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)
def logp(v, ref): r = np.sort(ref); return np.log((1 + np.searchsorted(r, v, side='right')) / (len(r) + 1))
def zfit(C): m, s = C.mean(0), C.std(0) + 1e-12; return lambda V: (V - m) / s


def scorers(Z):
    part = [str(p) for p in Z['part']]; iA = int(Z['iA']); ref = str(Z['ref'])
    def q(n):
        V = Z[f'V__{n}']; return V[:, 0, 0, 1:], V[:, 1, 0, 1:], V[:, 0, 1, 1:]                     # amplitude upright a, upside-down b, log std upright l
    ca, cb, cl = q(ref); ok = (ca.std(0) > 1e-9) & (cl.std(0) > 1e-9); S = {}
    muT, covT = maha_fit(ca[:, ok]); S['R'] = lambda n: q(n)[0][:, iA]; S['T'] = lambda n: -maha(q(n)[0][:, ok], muT, covT); D = lambda n: q(n)[0][:, iA] - q(n)[1][:, iA]
    za = zfit(ca[:, ok]); S['M1'] = lambda n: -(za(q(n)[0][:, ok]) ** 2).sum(1); S['M2'] = lambda n: -((q(n)[0][:, ok] - ca[:, ok].mean(0)) ** 2).sum(1)
    S['M3'] = lambda n: za(q(n)[0][:, ok]).sum(1); S['M4'] = lambda n: q(n)[0][:, ok].mean(1)
    lam, U = np.linalg.eigh(covT); S['M5'] = lambda n: -(((q(n)[0][:, ok] - muT) @ U) ** 2 / np.sqrt(lam)).sum(1)
    cr = lambda n: q(n)[0] - q(n)[2]; cc = ca - cl; okc = ok & (cc.std(0) > 1e-9); muC, covC = maha_fit(cc[:, okc]); zc = zfit(cc[:, okc])
    S['K1'] = lambda n: cr(n)[:, iA]; S['K2'] = lambda n: -maha(cr(n)[:, okc], muC, covC); S['K4'] = lambda n: -(zc(cr(n)[:, okc]) ** 2).sum(1)
    J = lambda n: np.c_[q(n)[0][:, ok], q(n)[2][:, ok]]; muJ, covJ = maha_fit(J(ref)); S['K3'] = lambda n: -maha(J(n), muJ, covJ)
    P = lambda n: np.c_[q(n)[0][:, ok], q(n)[1][:, ok]]; muP, covP = maha_fit(P(ref)); S['P1'] = lambda n: -maha(P(n), muP, covP)
    Q = lambda n: np.c_[q(n)[0][:, ok], q(n)[0][:, ok] - q(n)[1][:, ok]]; zq = zfit(Q(ref)); S['P2'] = lambda n: -(zq(Q(n)) ** 2).sum(1); S['P3'] = lambda n: zq(Q(n)).sum(1)
    return S, dict(q=q, ok=ok, iA=iA, part=part, res=[int(r) for r in Z['res']], muT=muT, lam=lam, U=U)


OUT = {'seed': SEED, 'rows': {r: {} for r in ROWS}, 'diag': {}}
for t in BASE:
    Z = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_v.npz'); Ze = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_v.npz'); names = [str(n) for n in Z['names']]
    idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]; assert not any(n.startswith(('sel-', 'test')) for n in names)
    (S, G), (Se, _) = scorers(Z), scorers(Ze)
    for r in ROWS:
        d = dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])), eq_vflip=float(auc(Se[r](idn), Se[r](vf))),
                 degr_mean=float(np.mean([auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}')) for c in COND]))); OUT['rows'][r][t] = d
    q, ok, iA = G['q'], G['ok'], G['iA']; L = len(ok); aI, bI, _ = q(idn)
    blk = [float(np.mean([auc(aI[:, h], q(o)[0][:, h]) for o in nat])) for h in range(L)]; av, bv, _ = q(vf); blkD = [float(auc(aI[:, h] - bI[:, h], av[:, h] - bv[:, h])) for h in range(L)]
    Y = lambda n: (q(n)[0][:, ok] - G['muT']) @ G['U'][:, ::-1]; lam = G['lam'][::-1]; yI = Y(idn); pcs = []
    for j in range(min(5, len(lam))): pcs.append(dict(var_share=float(lam[j] / lam.sum()), signed=float(np.mean([auc(yI[:, j], Y(o)[:, j]) for o in nat])), squared=float(np.mean([auc(-yI[:, j] ** 2, -Y(o)[:, j] ** 2) for o in nat]))))
    topk = {str(k): dict(first=float(np.mean([auc(-(yI[:, :k] ** 2 / lam[:k]).sum(1), -(Y(o)[:, :k] ** 2 / lam[:k]).sum(1)) for o in nat])), rest=float(np.mean([auc(-(yI[:, k:] ** 2 / lam[k:]).sum(1), -(Y(o)[:, k:] ** 2 / lam[k:]).sum(1)) for o in nat])))
            for k in (1, 2, 3, 5, 10) if k < len(lam)}
    OUT['diag'][t] = dict(blocks=[f"{G['part'][h][:1]}{G['res'][h]}" for h in range(L)], A=iA, block_auroc=blk, block_flip_auroc=blkD, pcs=pcs, topk=topk, n_blocks_used=int(ok.sum()))
for r in ROWS: OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'eq_nat', 'eq_vflip', 'degr_mean')}
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2b_s{SEED}.json', 'w'), indent=1)
print(f'== noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down || degraded copies')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:4s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f} || {m['degr_mean']:5.1f}")
for t in BASE:
    d = OUT['diag'][t]; print(f"-- {t}: one-sided AUROC of each block alone (natural mean); A = block {d['A']}"); print('   ' + ' '.join(f"{b}:{v:.0f}" for b, v in zip(d['blocks'], d['block_auroc'])))
    print('   upside-down, a_h - b_h: ' + ' '.join(f"{v:.0f}" for v in d['block_flip_auroc']))
    print('   principal components (share of variance | signed | squared): ' + '; '.join(f"{p['var_share']:.2f} | {p['signed']:.0f} | {p['squared']:.0f}" for p in d['pcs']))
    print('   Mahalanobis on the first k components | on the rest: ' + '; '.join(f"k={k}: {v['first']:.1f} | {v['rest']:.1f}" for k, v in d['topk'].items()))
print('GX_EVAL2_DONE')
