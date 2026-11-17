# olx_diag.py -- PROTOCOL_IDO.md I73 (X9, X8b, X10): descriptive diagnostics on val2, noise seeds 0-2, from the stored statistics
#   (runs/ido/gx/<tag>_s<seed>_{v,n,n2}.npz).  Not a method; nothing is selected with them.  All on the blocks up to A (the profile of OLR).
#   X9   does the amplitude fall?  The share of blocks where the mean over the seven other orientations of the calibration images is below
#        the mean of the calibration images (m_o,h < m_h); the share where the mean of each val2 OOD set is below m_h (mean over the sets,
#        min, max); the same for the val2 ID images (reference); per block, and at block A.
#   X8b  OLR's direction P (m - m_o) against directions fitted with the val2 OOD sets: cosine (S metric) with the pooled LDA direction and the
#        mean cosine with the per-set LDA directions; AUROC, two folds (even / odd images), of OLR, of its direction term l, of the pooled LDA
#        and of the per-set LDA, per OOD set.
#   X10  the covariance of the re-oriented profiles against that of the calibration images: kappa_tr = tr(P C) / p (P as in OLR) and
#        kappa_det = (det C / det S)^(1/p) (both without ridge), with C = the covariance of the seven orientations pooled around m_o, and the
#        mean within-orientation covariance; the val2 ID images as reference (kappa near 1 expected); the c implied by kappa, 1 / (kappa - 1).
#   usage: python code/olx_diag.py   -> runs/ido/final/olx_diag.json
import os, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEEDS = (0, 1, 2); O7 = ['v', 'h', 'r180', 'r90', 'r270', 't1', 't2']; C8 = 8.0
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def rcov(C): cov = np.atleast_2d(np.cov(C.T)); return cov + 1e-3 * np.mean(np.diag(cov)) * np.eye(C.shape[1])
def mean_of(dicts, key): return float(np.mean([d[key] for d in dicts]))
RAW = {t: [] for t in BASE}
for t in BASE:
    for sd in SEEDS:
        ld = lambda lab: np.load(f'{ROOT}/runs/ido/gx/{t}_s{sd}_{lab}.npz'); Z, N, N2 = ld('v'), ld('n'), ld('n2')
        iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part']); ca = Z[f'V__{ref}'][:, 0, 0, 1:]; ok = ca.std(0) > 1e-9; cs = [h for h in range(L) if ok[h] and h <= iA]; p = len(cs); assert cs[-1] == iA
        K = {'v': Z[f'V__{ref}'][:, 1, 0, 1:], 'h': N2[f'V__{ref}__hf'][:, 0, 0, 1:], 'r180': N2[f'V__{ref}__hf'][:, 1, 0, 1:], 'r90': N[f'V__{ref}__r90'][:, 0, 0, 1:],
             'r270': N[f'V__{ref}__r270'][:, 0, 0, 1:], 't1': N[f'V__{ref}__r90'][:, 1, 0, 1:], 't2': N[f'V__{ref}__r270'][:, 1, 0, 1:]}
        K = {k: v[:, cs] for k, v in K.items()}; U = ca[:, cs]; mu = U.mean(0); Sg = rcov(U); Ci = np.linalg.inv(Sg); muk = {k: K[k].mean(0) for k in O7}; mo = np.mean([muk[k] for k in O7], 0)
        names = [str(n) for n in Z['names']]; idn = f'fresh-{IDD[t]}'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, idn + '__vflip', idn + '__hflip') and 'synth' not in n]
        assert not any(n.startswith(('sel-', 'test')) for n in names)
        q = lambda n: Z[f'V__{n}'][:, 0, 0, 1:][:, cs]; XI = q(idn); XO = {o: q(o) for o in nat}
        # X9
        bo = {o: XO[o].mean(0) < mu for o in nat}
        x9 = dict(p=p, orient=float(np.mean(mo < mu)), orient_each={k: float(np.mean(muk[k] < mu)) for k in O7}, ood={o: float(np.mean(bo[o])) for o in nat},
                  id_val2=float(np.mean(XI.mean(0) < mu)), atA_orient=float(mo[-1] < mu[-1]), atA_ood=float(np.mean([bo[o][-1] for o in nat])), atA_id_val2=float(XI.mean(0)[-1] < mu[-1]),
                  per_block_orient=(mo < mu).astype(float).tolist(), per_block_ood=np.mean([bo[o] for o in nat], 0).tolist(),
                  per_block_part=[str(Z['part'][h]) for h in cs], per_block_res=[int(Z['res'][h]) for h in cs])
        x9['ood_mean'] = float(np.mean(list(x9['ood'].values()))); x9['ood_min'] = float(min(x9['ood'].values())); x9['ood_max'] = float(max(x9['ood'].values()))
        # X8b
        w = Ci @ (mu - mo); ms = mu + C8 * (mu - mo)
        s_ol = lambda X: -np.einsum('ij,jk,ik->i', X - ms, Ci, X - ms); l_ol = lambda X: (X - mu) @ w
        lda = lambda A, B: np.linalg.solve(0.5 * (rcov(A) + rcov(B)), A.mean(0) - B.mean(0)); cz = lambda x, y: float((x @ Sg @ y) / np.sqrt((x @ Sg @ x) * (y @ Sg @ y)))
        R = {k: {o: [] for o in nat} for k in ('OLR', 'dir', 'lda_pool', 'lda_set')}
        for f in (0, 1):
            tri, tei = XI[f::2], XI[1 - f::2]; wp = lda(tri, np.concatenate([XO[o][f::2] for o in nat]))
            for o in nat:
                teo = XO[o][1 - f::2]; ws = lda(tri, XO[o][f::2])
                R['OLR'][o].append(auc(s_ol(tei), s_ol(teo))); R['dir'][o].append(auc(l_ol(tei), l_ol(teo))); R['lda_pool'][o].append(auc(tei @ wp, teo @ wp)); R['lda_set'][o].append(auc(tei @ ws, teo @ ws))
        x8 = {k: {o: float(np.mean(v)) for o, v in R[k].items()} for k in R}
        x8['mean'] = {k: float(np.mean(list(x8[k].values()))) for k in R}
        x8['cos_pool'] = cz(lda(XI, np.concatenate([XO[o] for o in nat])), w); x8['cos_set'] = {o: cz(lda(XI, XO[o]), w) for o in nat}; x8['cos_set_mean'] = float(np.mean(list(x8['cos_set'].values())))
        # X10
        Sraw = np.cov(U.T)
        def kap(C): ev = np.sort(np.linalg.eigvals(np.linalg.solve(Sraw, C)).real); return dict(trace=float(np.trace(Ci @ C) / p), det=float(np.exp((np.linalg.slogdet(C)[1] - np.linalg.slogdet(Sraw)[1]) / p)),
                                                                                          eig_min=float(ev[0]), eig_med=float(np.median(ev)), eig_max=float(ev[-1]))
        x10 = dict(pooled=kap(np.cov(np.concatenate([K[k] for k in O7]).T)), within=kap(np.mean([np.cov(K[k].T) for k in O7], 0)), id_val2=kap(np.cov(XI.T)), each={k: kap(np.cov(K[k].T))['trace'] for k in O7})
        RAW[t].append(dict(X9=x9, X8b=x8, X10=x10)); print(t, 's%d' % sd, 'p', p, '| X9 orient %.2f ood %.2f (%.2f-%.2f) id %.2f | at A: orient %d ood %.2f' % (x9['orient'], x9['ood_mean'], x9['ood_min'], x9['ood_max'], x9['id_val2'], x9['atA_orient'], x9['atA_ood']),
                                                       '| X8b OLR %.2f dir %.2f pool %.2f set %.2f cos %.2f' % (x8['mean']['OLR'], x8['mean']['dir'], x8['mean']['lda_pool'], x8['mean']['lda_set'], x8['cos_pool']),
                                                       '| X10 kappa_tr pooled %.2f within %.2f id %.2f, det pooled %.2f' % (x10['pooled']['trace'], x10['within']['trace'], x10['id_val2']['trace'], x10['pooled']['det']), flush=True)
OUT = {'seeds': list(SEEDS), 'raw': RAW, 'mean': {}}
for t in BASE:
    r = RAW[t]; nat = list(r[0]['X8b']['OLR'])
    OUT['mean'][t] = dict(p=r[0]['X9']['p'], orient=mean_of([x['X9'] for x in r], 'orient'), ood_mean=mean_of([x['X9'] for x in r], 'ood_mean'), ood_min=mean_of([x['X9'] for x in r], 'ood_min'),
                          ood_max=mean_of([x['X9'] for x in r], 'ood_max'), id_val2=mean_of([x['X9'] for x in r], 'id_val2'), atA_orient=mean_of([x['X9'] for x in r], 'atA_orient'), atA_ood=mean_of([x['X9'] for x in r], 'atA_ood'),
                          ood_sets={o: float(np.mean([x['X9']['ood'][o] for x in r])) for o in nat},
                          auroc={k: float(np.mean([x['X8b']['mean'][k] for x in r])) for k in ('OLR', 'dir', 'lda_pool', 'lda_set')},
                          auroc_sets={k: {o: float(np.mean([x['X8b'][k][o] for x in r])) for o in nat} for k in ('OLR', 'dir', 'lda_pool', 'lda_set')},
                          cos_pool=float(np.mean([x['X8b']['cos_pool'] for x in r])), cos_set_mean=float(np.mean([x['X8b']['cos_set_mean'] for x in r])),
                          kappa={c: {m: float(np.mean([x['X10'][c][m] for x in r])) for m in ('trace', 'det', 'eig_min', 'eig_med', 'eig_max')} for c in ('pooled', 'within', 'id_val2')})
json.dump(OUT, open(f'{ROOT}/runs/ido/final/olx_diag.json', 'w'), indent=1)
print('== mean of seeds 0-2 (blocks up to A):  p | share of blocks below the ID mean: orientations, OOD sets (mean, min-max), val2 ID | at A: orientations, OOD | AUROC OLR, direction, pooled LDA, per-set LDA | cos | kappa_tr pooled, within, val2 ID; kappa_det pooled')
for t in BASE:
    m = OUT['mean'][t]; print(f"  {t:9s} {m['p']:2d} | {m['orient']:.2f} {m['ood_mean']:.2f} ({m['ood_min']:.2f}-{m['ood_max']:.2f}) {m['id_val2']:.2f} | {m['atA_orient']:.2f} {m['atA_ood']:.2f} | "
          + ' '.join(f"{m['auroc'][k]:.2f}" for k in ('OLR', 'dir', 'lda_pool', 'lda_set')) + f" | {m['cos_pool']:.2f} | {m['kappa']['pooled']['trace']:.2f} {m['kappa']['within']['trace']:.2f} {m['kappa']['id_val2']['trace']:.2f}; {m['kappa']['pooled']['det']:.2f}")
print('OLX_DIAG_DONE')
