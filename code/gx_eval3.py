# gx_eval3.py -- PROTOCOL_IDO.md I61 part II and I63: block A read per channel (runs/ido/gx/*_c.npz of code/gx_extract_c.py) on val2.
#   H1-H7 of I61, and the two ideas of rounds 3-4 on the channel amplitudes: LC (the Fisher direction between the upright and the
#   upside-down calibration images) and AC (the Mahalanobis distance to the ID mean moved one step away from the upside-down mean).
#   LCr / ACr: the same with a stronger ridge (0.1 x mean variance), for understanding only.  Development data only; ID = positive.
#   usage: python code/gx_eval3.py [seed]   -> runs/ido/final/gx_val2c_s<seed>.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']; ROWS = ['H1', 'H2', 'H2b', 'H3', 'H4', 'H5', 'H6', 'H7', 'LC', 'AC', 'LCr', 'ACr']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def rcov(C, r=1e-3): cov = np.atleast_2d(np.cov(C.T)); return cov + r * np.mean(np.diag(cov)) * np.eye(C.shape[1])
def mfit(C, r=1e-3):
    mu = C.mean(0); Ci = np.linalg.inv(rcov(C, r)); return lambda V, m=None: np.einsum('ij,jk,ik->i', V - (mu if m is None else m), Ci, V - (mu if m is None else m))
def zfit(C): m, s = C.mean(0), C.std(0) + 1e-12; return lambda V: (V - m) / s


def scorers(Z):
    ref = str(Z['ref']); S = {}
    def q(n): V = Z[f'V__{n}'].astype(np.float64); return V[:, 0, 0], V[:, 1, 0], V[:, 0, 1], V[:, 0, 2]      # channel amplitudes of the image a, of its upside-down copy b, log std l, mean m
    ca, cb, cl, cm = q(ref); ok = (ca.std(0) > 1e-9) & (cl.std(0) > 1e-9); za = zfit(ca[:, ok]); S['H1'] = lambda n: za(q(n)[0][:, ok]).mean(1); S['H2'] = lambda n: -(za(q(n)[0][:, ok]) ** 2).sum(1)
    ms = lambda n: np.c_[q(n)[3][:, ok], np.exp(q(n)[2][:, ok])]; zm = zfit(ms(ref)); S['H2b'] = lambda n: -(zm(ms(n)) ** 2).sum(1)
    m3 = mfit(ca[:, ok]); S['H3'] = lambda n: -m3(q(n)[0][:, ok]); m4 = mfit(cl[:, ok]); S['H4'] = lambda n: -m4(q(n)[2][:, ok])
    zd = zfit((ca - cb)[:, ok]); S['H5'] = lambda n: zd((q(n)[0] - q(n)[1])[:, ok]).mean(1)
    zz = lambda f: (lambda n, mu=f(ref).mean(), sd=f(ref).std() + 1e-12: (f(n) - mu) / sd); z1, z5 = zz(S['H1']), zz(S['H5']); S['H6'] = lambda n: z1(n) + z5(n)
    pr = lambda n: np.c_[q(n)[0][:, ok], q(n)[1][:, ok]]; m7 = mfit(pr(ref)); S['H7'] = lambda n: -m7(pr(n))
    U, F = ca[:, ok], cb[:, ok]; d = U.mean(0) - F.mean(0)
    for nm, r in (('', 1e-3), ('r', 1e-1)):
        w = np.linalg.solve(0.5 * (rcov(U, r) + rcov(F, r)), d); S['LC' + nm] = lambda n, w=w: q(n)[0][:, ok] @ w
        mm = mfit(U, r); S['AC' + nm] = lambda n, mm=mm: -mm(q(n)[0][:, ok], U.mean(0) + d)
    return S, int(ok.sum())


OUT = {'seed': SEED, 'rows': {r: {} for r in ROWS}, 'channels': {}}
for t in BASE:
    Z = np.load(f'{ROOT}/runs/ido/gx/{t}_s{SEED}_c.npz'); Ze = np.load(f'{ROOT}/runs/ido/gx/{t}_eq50_s{SEED}_c.npz'); names = [str(n) for n in Z['names']]
    idn = f'fresh-{IDD[t]}'; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, vf, hf) and 'synth' not in n]; assert not any(n.startswith(('sel-', 'test')) for n in names)
    (S, nc), (Se, _) = scorers(Z), scorers(Ze); OUT['channels'][t] = nc
    for r in ROWS:
        OUT['rows'][r][t] = dict(nat=float(np.mean([auc(S[r](idn), S[r](o)) for o in nat])), vflip=float(auc(S[r](idn), S[r](vf))), eq_nat=float(np.mean([auc(Se[r](idn), Se[r](o)) for o in nat])), eq_vflip=float(auc(Se[r](idn), Se[r](vf))),
                                 degr_mean=float(np.mean([auc(S[r](idn), S[r](f'dqv-{IDD[t]}__{c}')) for c in COND])), per={o: float(auc(S[r](idn), S[r](o))) for o in nat})
    print(t, 'channels', nc, flush=True)
for r in ROWS: OUT['rows'][r]['mean'] = {k: float(np.mean([OUT['rows'][r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'eq_nat', 'eq_vflip', 'degr_mean')}
json.dump(OUT, open(f'{ROOT}/runs/ido/final/gx_val2c_s{SEED}.json', 'w'), indent=1)
print(f'== block A per channel, noise seed {SEED}, val2: natural mean per ID (CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST) -> five-ID mean | upside-down || identical history: mean | upside-down || degraded copies   (bars: 95.07 on val2, 90.33 under the identical history)')
for r in ROWS:
    m = OUT['rows'][r]['mean']; print(f'  {r:4s} ' + ' '.join(f"{OUT['rows'][r][t]['nat']:6.2f}" for t in BASE) + f"  -> {m['nat']:6.2f} | {m['vflip']:5.1f} || {m['eq_nat']:6.2f} | {m['eq_vflip']:5.1f} || {m['degr_mean']:5.1f}")
print('GX_EVAL3_DONE')
