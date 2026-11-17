# olx_eval.py -- PROTOCOL_IDO.md I70: the analyses of the frozen orientation likelihood-ratio score OL made after its single evaluation.
#   usage: python code/olx_eval.py <part>     AUROC in %, ID = positive; noise seeds 0-2 unless stated
#   sets  X7  per-set AUROC and FPR95 of OL on the six splits, original images and identical history                -> runs/ido/final/olx_sets.json
#   cfs   X3  test (server test2): OL, CFS, CFS-kNN with the original images (runs/cfs_l7) and under the identical history (runs/ido/cfseq);
#             paired bootstrap (2,000, rng 20261121, mean of the seeds) of OL - CFS and OL - CFS-kNN                 -> runs/ido/final/olx_cfs.json
#   dq    X1  the ID images of test7 against their seven degraded copies: OL and its two parts, CFS, CFS-kNN (runs/ido/cfsdq)  -> olx_dq.json
#   var   X2  test7, seed 0: the trained model, freshly initialised weights, shuffled weights, the trained model of another dataset -> olx_var.json
#   hr    X4  256x256: in256c and the 112-px control in256d; OL and its parts, CFS, CFS-kNN (stored); bootstrap (rng 20261122)   -> olx_hr.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); sys.path.insert(0, f'{ROOT}/code')
from boot import counts, boot_auroc
PARTN = sys.argv[1]; SEEDS = (0, 1, 2); NB = 2000; BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SPL = ['test7', 'test6', 'test5', 'test4', 'test3', 'test2']
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
FIN = f'{ROOT}/runs/ido/final'; OLD = f'{ROOT}/runs/ido/ol'
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def fpr(a, b): return 100.0 * float(np.mean(b >= np.percentile(a, 5.0)))
def align(v, have, want):
    if len(have) == len(want) and (np.asarray(have) == np.asarray(want)).all(): return np.asarray(v, np.float64)
    pos = {int(i): k for k, i in enumerate(have)}; return np.asarray(v, np.float64)[[pos[int(i)] for i in want]]
def sets_of(names, sp):
    S = [n for n in names if n.startswith(sp + '-')]; idn = [n for n in S if n.endswith('__vflip')][0][:-len('__vflip')]; vf, hf = idn + '__vflip', idn + '__hflip'
    return idn, vf, hf, [n for n in S if n not in (idn, vf, hf) and 'synth' not in n], [n for n in S if 'synth' in n]
def paired(G, SETS, tags, refs, rng, ours='OL'):
    """paired bootstrap of ours - ref on the natural mean (per tag and mean over the tags), and wins / ties / losses over the pairs (natural sets and upside-down)"""
    BT = {c: {} for c in refs}; W = {c: dict(wins=0, ties=0, losses=0) for c in refs}; per = {c: {} for c in refs}; pt = {c: {} for c in refs}
    for t in tags:
        idn, vf, nat = SETS[t]; Wa = counts(rng, len(G[t][ours](idn, 0)), NB); acc = {c: [] for c in refs}; pd = {c: [] for c in refs}
        for o in nat + ([vf] if vf else []):
            Wb = counts(rng, len(G[t][ours](o, 0)), NB); x = np.mean([boot_auroc(G[t][ours](idn, s), G[t][ours](o, s), Wa, Wb) for s in SEEDS], 0); xo = float(np.mean([auc(G[t][ours](idn, s), G[t][ours](o, s)) for s in SEEDS]))
            for c in refs:
                y = np.mean([boot_auroc(G[t][c](idn, s), G[t][c](o, s), Wa, Wb) for s in SEEDS], 0); d = xo - float(np.mean([auc(G[t][c](idn, s), G[t][c](o, s)) for s in SEEDS]))
                lo, hi = np.percentile(x - y, [2.5, 97.5]); v = 'tie' if abs(d) < 0.05 else ('win' if lo > 0 else ('loss' if hi < 0 else 'tie')); W[c][{'win': 'wins', 'tie': 'ties', 'loss': 'losses'}[v]] += 1
                per[c][f'{t}|{o}'] = dict(diff=d, ci=[float(lo), float(hi)], verdict=v)
                if o != vf: acc[c].append(x - y); pd[c].append(d)
        for c in refs: BT[c][t] = np.mean(acc[c], 0); pt[c][t] = float(np.mean(pd[c]))
    boot = {}
    for c in refs:
        boot[c] = {t: dict(diff=pt[c][t], ci=[float(v) for v in np.percentile(BT[c][t], [2.5, 97.5])]) for t in tags}; m = np.mean([BT[c][t] for t in tags], 0)
        boot[c]['mean'] = dict(diff=float(np.mean([pt[c][t] for t in tags])), ci=[float(v) for v in np.percentile(m, [2.5, 97.5])], p_le0=float(np.mean(m <= 0)))
    return boot, W, per


if PARTN == 'sets':
    OUT = {}
    for sp in SPL:
        OUT[sp] = {}
        for cond, suf in (('original', ''), ('eq50', '_eq50')):
            OUT[sp][cond] = {}
            for t in BASE:
                F = {s: np.load(f'{OLD}/{t}{suf}_s{s}_final.npz') for s in SEEDS}; idn, vf, hf, nat, syn = sets_of([str(n) for n in F[0]['names']], sp)
                P = {o[len(sp) + 1:]: dict(auroc=float(np.mean([auc(F[s][f'score__{idn}'], F[s][f'score__{o}']) for s in SEEDS])), fpr95=float(np.mean([fpr(F[s][f'score__{idn}'], F[s][f'score__{o}']) for s in SEEDS])),
                                           n=int(len(F[0][f'idx__{o}'])), kind=('nat' if o in nat else ('synth' if o in syn else o[-5:]))) for o in nat + [vf, hf] + syn}
                OUT[sp][cond][t] = dict(sets=P, n_id=int(len(F[0][f'idx__{idn}'])), nat=float(np.mean([P[o[len(sp) + 1:]]['auroc'] for o in nat])), fpr95=float(np.mean([P[o[len(sp) + 1:]]['fpr95'] for o in nat])),
                                        vflip=P[vf[len(sp) + 1:]]['auroc'], hflip=P[hf[len(sp) + 1:]]['auroc'], blocks=int(F[0]['n_blocks_used']), A=str(F[0]['A']))
            OUT[sp][cond]['mean'] = {k: float(np.mean([OUT[sp][cond][t][k] for t in BASE])) for k in ('nat', 'fpr95', 'vflip', 'hflip')}
            print(sp, cond, ' '.join(f"{OUT[sp][cond][t]['nat']:6.2f}" for t in BASE), '->', ' '.join(f"{k} {v:.2f}" for k, v in OUT[sp][cond]['mean'].items()), flush=True)
    json.dump(OUT, open(f'{FIN}/olx_sets.json', 'w'), indent=1); print('OLX_SETS_DONE')

if PARTN == 'cfs':
    rng = np.random.default_rng(20261121); OUT = {}; ROWS = ['OL', 'CFS', 'CFS-kNN']
    for cond, suf in (('original', ''), ('eq50', '_eq50')):
        G, SETS = {}, {}
        for t in BASE:
            Of = {s: np.load(f'{OLD}/{t}{suf}_s{s}_final.npz') for s in SEEDS}; idn, vf, hf, nat, syn = sets_of([str(n) for n in Of[0]['names']], 'test2'); SETS[t] = (idn, vf, nat)
            if cond == 'original':
                def cfs(h, n, s, t=t, Of=Of): z = np.load(f'{ROOT}/runs/cfs_l7/s{s}/{t}/{n}__scores.npz'); return align(z[h], z['idx'], Of[s][f'idx__{n}'])
            else:
                C = {s: np.load(f'{ROOT}/runs/ido/cfseq/{t}_s{s}.npz') for s in SEEDS}; Ob = {s: np.load(f'{OLD}/{t}_s{s}_final.npz') for s in SEEDS}      # the processed sets keep the order of the original ones
                def cfs(h, n, s, C=C, Of=Of, Ob=Ob): assert len(Of[s][f'score__{n}']) == len(Ob[s][f'idx__{n}']); return align(C[s][f'{h}__{n}'], C[s][f'idx__{n}'], Ob[s][f'idx__{n}'])
            G[t] = {'OL': lambda n, s, Of=Of: np.asarray(Of[s][f'score__{n}'], np.float64), 'CFS': lambda n, s, cfs=cfs: cfs('diag', n, s), 'CFS-kNN': lambda n, s, cfs=cfs: cfs('knn', n, s)}
        R = {r: {t: dict(nat=float(np.mean([np.mean([auc(G[t][r](SETS[t][0], s), G[t][r](o, s)) for o in SETS[t][2]]) for s in SEEDS])), vflip=float(np.mean([auc(G[t][r](SETS[t][0], s), G[t][r](SETS[t][1], s)) for s in SEEDS])),
                         fpr95=float(np.mean([np.mean([fpr(G[t][r](SETS[t][0], s), G[t][r](o, s)) for o in SETS[t][2]]) for s in SEEDS]))) for t in BASE} for r in ROWS}
        for r in ROWS: R[r]['mean'] = {k: float(np.mean([R[r][t][k] for t in BASE])) for k in ('nat', 'vflip', 'fpr95')}
        boot, W, per = paired(G, SETS, BASE, ['CFS', 'CFS-kNN'], rng); OUT[cond] = dict(rows=R, boot=boot, wtl=W, pairs=per, n_pairs=len(per['CFS']))
        for r in ROWS: print(f'{cond:8s} {r:8s} ' + ' '.join(f"{R[r][t]['nat']:6.2f}" for t in BASE) + f" -> {R[r]['mean']['nat']:6.2f} | {R[r]['mean']['vflip']:5.1f} | FPR95 {R[r]['mean']['fpr95']:5.1f}")
        for c in ('CFS', 'CFS-kNN'): b = boot[c]; print(f"   OL - {c}: {b['mean']['diff']:+.2f} [{b['mean']['ci'][0]:+.2f}, {b['mean']['ci'][1]:+.2f}]  W/T/L {W[c]['wins']}/{W[c]['ties']}/{W[c]['losses']}  per ID " + ' '.join(f"{b[t]['diff']:+.2f}" for t in BASE), flush=True)
    json.dump(OUT, open(f'{FIN}/olx_cfs.json', 'w'), indent=1); print('OLX_CFS_DONE')

if PARTN == 'dq':
    COND = ['blur05', 'blur10', 'jpeg90', 'jpeg75', 'jpeg50', 'down75', 'down50']; ROWS = ['OL', 'OL-l', 'OL-d2', 'CFS', 'CFS-kNN']; OUT = {'cond': COND, 'rows': {r: {} for r in ROWS}}; chk = 0.0
    for t in BASE:
        Z = {s: np.load(f'{OLD}/{t}_s{s}_dq.npz') for s in SEEDS}; F = {s: np.load(f'{OLD}/{t}_s{s}_final.npz') for s in SEEDS}; C = {s: np.load(f'{ROOT}/runs/ido/cfsdq/{t}_s{s}.npz') for s in SEEDS}; idn = f'test7-{IDD[t]}'
        for s in SEEDS: chk = max(chk, float(np.abs(Z[s][f'score__{idn}'] - F[s][f'score__{idn}']).max())); assert (Z[s][f'idx__{idn}'] == C[s][f'idx__{idn}']).all()
        for r, (f, sg) in (('OL', ('score', 1.0)), ('OL-l', ('l', 1.0)), ('OL-d2', ('d2', -1.0))): OUT['rows'][r][t] = {c: float(np.mean([auc(sg * Z[s][f'{f}__{idn}'], sg * Z[s][f'{f}__dq-{IDD[t]}__{c}']) for s in SEEDS])) for c in COND}
        for r, h in (('CFS', 'diag'), ('CFS-kNN', 'knn')): OUT['rows'][r][t] = {c: float(np.mean([auc(C[s][f'{h}__{idn}'], C[s][f'{h}__dq-{IDD[t]}__{c}']) for s in SEEDS])) for c in COND}
        OUT.setdefault('n', {})[t] = int(len(Z[0][f'idx__{idn}']))
    for r in ROWS: OUT['rows'][r]['mean'] = {c: float(np.mean([OUT['rows'][r][t][c] for t in BASE])) for c in COND}
    OUT['check_max_abs_diff_of_the_ID_scores_to_the_evaluation'] = chk; json.dump(OUT, open(f'{FIN}/olx_dq.json', 'w'), indent=1)
    print(f'== ID images of test7 against their degraded copies: AUROC (%), original = positive; 50 = blind (ID scores vs the evaluation: max abs diff {chk:.1e})')
    for r in ROWS:
        print(f'-- {r}:   ' + ''.join(c.rjust(9) for c in COND))
        for t in BASE + ['mean']: print(f'  {t:9s}' + ''.join(f"{OUT['rows'][r][t][c]:9.1f}" for c in COND))
    print('OLX_DQ_DONE')

if PARTN == 'var':
    FLD = {'score': ('score', 1.0), 'all': ('sa', 1.0), 'l': ('l', 1.0), 'd2': ('d2', -1.0)}; OUT = {}
    def rows_of(z, sp='test7'):
        idn, vf, hf, nat, syn = sets_of([str(n) for n in z['names']], sp)
        d = {k: dict(nat=float(np.mean([auc(sg * z[f'{f}__{idn}'], sg * z[f'{f}__{o}']) for o in nat])), vflip=float(auc(sg * z[f'{f}__{idn}'], sg * z[f'{f}__{vf}']))) for k, (f, sg) in FLD.items()}
        d['blocks'] = int(z['n_blocks_used']); d['finite'] = bool(all(np.isfinite(z[f'score__{o}']).all() for o in [idn, vf] + nat)); return d
    for b in BASE:
        OUT[b] = {'trained': rows_of(np.load(f'{OLD}/{b}_s0_final.npz'))}
        for v in ['rinit', 'rshuf'] + [f'by_{m}' for m in ('edmc10', 'l6svhn', 'dpceleba') if m != b and b in ('edmc10', 'l6svhn', 'dpceleba')]:
            p = f'{OLD}/{b}_{v}_s0_x.npz'
            if os.path.exists(p): OUT[b][v] = rows_of(np.load(p))
    json.dump(OUT, open(f'{FIN}/olx_var.json', 'w'), indent=1)
    print('== test7, noise seed 0: natural-OOD mean AUROC (%) | upside-down, per ID and model variant (calibration = the val1 ID images of the ID)')
    for k in FLD:
        print(f'-- {k}')
        for b in BASE: print(f'  {b:9s} ' + '   '.join(f"{v} {d[k]['nat']:.1f}|{d[k]['vflip']:.1f}" for v, d in OUT[b].items()))
    print('blocks / finite:', {b: {v: (d['blocks'], d['finite']) for v, d in OUT[b].items()} for b in BASE}); print('OLX_VAR_DONE')

if PARTN == 'hr':
    rng = np.random.default_rng(20261122); OUT = {}; S5 = ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o']; NEAR, FAR = S5[:2], S5[2:]; ROWS = ['OL', 'OLa', 'OLl', 'OLt', 'CFS', 'CFS-kNN']
    for tier, P in (('in256c', 'q256'), ('in256d', 'q256d')):
        Of = {s: np.load(f'{OLD}/{tier}_s{s}_hr.npz') for s in SEEDS}; idn = f'{P}-imagenet_test'; vf, hf = idn + '__vflip', idn + '__hflip'
        def cfs(h, n, s, tier=tier, Of=Of): z = np.load(f'{ROOT}/runs/cfs_l7/s{s}/{tier}/{n}__scores.npz'); return align(z[h], z['idx'], Of[s][f'idx__{n}'])
        def ol(f, sg=1.0, Of=Of): return lambda n, s: sg * np.asarray(Of[s][f'{f}__{n}'], np.float64)
        G = {tier: {'OL': ol('score'), 'OLa': ol('sa'), 'OLl': ol('l'), 'OLt': ol('d2', -1.0), 'CFS': lambda n, s, cfs=cfs: cfs('diag', n, s), 'CFS-kNN': lambda n, s, cfs=cfs: cfs('knn', n, s)}}
        R = {}
        for r in ROWS:
            a = {o: float(np.mean([auc(G[tier][r](idn, s), G[tier][r](f'{P}-{o}', s)) for s in SEEDS])) for o in S5}; has = True
            try: v = float(np.mean([auc(G[tier][r](idn, s), G[tier][r](vf, s)) for s in SEEDS]))
            except (KeyError, FileNotFoundError): v, has = float('nan'), False
            R[r] = dict(auroc=a, near=float(np.mean([a[o] for o in NEAR])), far=float(np.mean([a[o] for o in FAR])), natural=float(np.mean(list(a.values()))), vflip=v, fpr95=float(np.mean([np.mean([fpr(G[tier][r](idn, s), G[tier][r](f'{P}-{o}', s)) for o in S5]) for s in SEEDS])))
        hfl = float(np.mean([auc(G[tier]['OL'](idn, s), G[tier]['OL'](hf, s)) for s in SEEDS]))
        hasv = bool(np.isfinite(R['CFS']['vflip']))
        boot, W, per = paired({tier: {k: G[tier][k] for k in ('OL', 'CFS', 'CFS-kNN')}}, {tier: (idn, vf if hasv else None, [f'{P}-{o}' for o in S5])}, [tier], ['CFS', 'CFS-kNN'], rng)
        OUT[tier] = dict(rows=R, hflip_OL=hfl, boot={c: boot[c][tier] for c in boot}, wtl=W, pairs=per, n={o: int(len(Of[0][f'idx__{P}-{o}'])) for o in ['imagenet_test'] + S5}, blocks=int(Of[0]['n_blocks_used']), A=str(Of[0]['A']))
        print(f'== {tier}: ' + ' '.join(o.rjust(11) for o in S5) + '        near    far natural | upside-down | FPR95')
        for r in ROWS: d = R[r]; print(f'  {r:8s} ' + ' '.join(f"{d['auroc'][o]:11.2f}" for o in S5) + f"   {d['near']:7.2f} {d['far']:6.2f} {d['natural']:7.2f} | {d['vflip']:6.2f} | {d['fpr95']:5.1f}")
        for c in ('CFS', 'CFS-kNN'): b = OUT[tier]['boot'][c]; print(f"   OL - {c} (natural): {b['diff']:+.2f} [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}]  W/T/L {W[c]['wins']}/{W[c]['ties']}/{W[c]['losses']}", flush=True)
    json.dump(OUT, open(f'{FIN}/olx_hr.json', 'w'), indent=1); print('OLX_HR_DONE')

if PARTN == 'info':                                         # the fitted quantities of the frozen score (noise seed 0): p, n, Delta, the largest |m_k - m| is not stored; Delta from m, m*, P
    OUT = {}
    for t in BASE + ['in256c', 'in256d']:
        p = f'{OLD}/{t}_s0_' + ('hr' if t.startswith('in256') else 'final') + '.npz'
        if not os.path.exists(p): continue
        z = np.load(p); m, ms, Ci, c = z['m'], z['m_star'], z['Ci'], float(z['c']); d = (ms - m) / c; ref = str(z['ref'])
        OUT[t] = dict(p=int(len(m)), blocks_up_to_A=int(list(z['hooks']).index(str(z['A'])) + 1), blocks_all=int(len(z['hooks'])), blocks_used_all=int(z['n_blocks_used_all']), A=str(z['A']), c=c, Delta=float(np.sqrt(d @ Ci @ d)), n_cal=int(len(z[f'idx__{ref}'])),
                      sd_ratio=float(2 * c * np.sqrt(d @ Ci @ d) / np.sqrt(2 * len(m))))
        print(t, OUT[t])
    json.dump(OUT, open(f'{FIN}/olx_info.json', 'w'), indent=1); print('OLX_INFO_DONE')
