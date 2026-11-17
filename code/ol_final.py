# ol_final.py -- PROTOCOL_IDO.md I68: the single evaluation of the frozen orientation likelihood-ratio score (code/ol_score.py) on the same
#   images as its references.  Splits test7 ... test2: all have been looked at with other methods, never with this
#   score (disclosed).  Noise seeds 0-2, the mean over the seeds is reported.  Two conditions: the original images, and the identical
#   resampling history (every image reduced to half its side and brought back; tag <id>_eq50).
#   OL = score (blocks up to A, one stopped pass); ablations: OLa (all blocks), OLl (l alone), OLt (-d2 alone).
#   PUBLIC VERSION (omotes24/OLR): the run (md5 4fa29b1f13d1, PROTOCOL_IDO.md I68-I69) also scored registered references, methods outside
#   this paper, with a paired bootstrap against them; those parts are left out here.  What remains computes exactly the OLR rows of
#   runs/ido/final/ol_final.json (OL and the three ablations per split, condition and ID, mean of the seeds) and the statement H4.
#   -> runs/ido/final/ol_final.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/olr'); sys.path.insert(0, f'{ROOT}/code')
from boot import counts, boot_auroc
BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEEDS = (0, 1, 2); NB = 2000; SPL = ['test7', 'test6', 'test5', 'test4', 'test3', 'test2']; ABL = ['OLa', 'OLl', 'OLt']
def auc(a, b):
    r = rankdata(np.r_[a, b]); na, nb = len(a), len(b); return 100.0 * (r[:na].sum() - na * (na + 1) / 2.0) / (na * nb)
def get(F, key, sign=1.0): return lambda n, s: sign * np.asarray(F[s][f'{key}__{n}'], np.float64)
rng = np.random.default_rng(20261120); OUT = {'splits': {}}
for sp in SPL:
    OUT['splits'][sp] = {}
    for cond in ('original', 'eq50'):
        G, SETS = {}, {}
        for t in BASE:
            Of = {s: np.load(f'{ROOT}/runs/ido/ol/{t}_s{s}_final.npz') for s in SEEDS} if cond == 'original' else {s: np.load(f'{ROOT}/runs/ido/ol/{t}_eq50_s{s}_final.npz') for s in SEEDS}
            S = [str(n) for n in Of[0]['names'] if str(n).startswith(sp + '-')]
            idn = [n for n in S if n.endswith('__vflip')][0][:-len('__vflip')]; vf, hf = idn + '__vflip', idn + '__hflip'; nat = [n for n in S if n not in (idn, vf, hf) and 'synth' not in n]; SETS[t] = (idn, vf, nat)
            G[t] = {'OL': get(Of, 'score'), 'OLa': get(Of, 'sa'), 'OLl': get(Of, 'l'), 'OLt': get(Of, 'd2', -1.0)}
        KEYS = ['OL'] + ABL
        R = {r: {t: dict(nat=float(np.mean([np.mean([auc(G[t][r](SETS[t][0], s), G[t][r](o, s)) for o in SETS[t][2]]) for s in SEEDS])),
                         vflip=float(np.mean([auc(G[t][r](SETS[t][0], s), G[t][r](SETS[t][1], s)) for s in SEEDS]))) for t in BASE} for r in KEYS}
        for r in R: R[r]['mean'] = dict(nat=float(np.mean([R[r][t]['nat'] for t in BASE])), vflip=float(np.mean([R[r][t]['vflip'] for t in BASE])))
        OUT['splits'][sp][cond] = dict(rows=R, n_pairs=sum(len(SETS[t][2]) + 1 for t in BASE))
        print(f'{sp} {cond:8s} OL ' + ' '.join(f"{R['OL'][t]['nat']:6.2f}" for t in BASE) + f" -> {R['OL']['mean']['nat']:6.2f} | {R['OL']['mean']['vflip']:5.1f}"
              + '   ablations ' + ' '.join(f"{k} {R[k]['mean']['nat']:.2f}" for k in ABL), flush=True)
SM = {}
for cond in ('original', 'eq50'):
    six = {k: float(np.mean([OUT['splits'][sp][cond]['rows'][k]['mean']['nat'] for sp in SPL])) for k in ['OL'] + ABL}; vfl = {'OL': float(np.mean([OUT['splits'][sp][cond]['rows']['OL']['mean']['vflip'] for sp in SPL]))}
    SM[cond] = dict(six_split_mean=six, six_split_vflip=vfl)
    print(f'{cond}: six-split mean ' + ' '.join(f'{k} {v:.2f}' for k, v in six.items()) + ' | upside-down ' + ' '.join(f'{k} {v:.1f}' for k, v in vfl.items()), flush=True)
OUT['summary'] = SM; OUT['claims'] = dict(H4=SM['original']['six_split_mean']['OL'] >= 95.0)      # H4 of I68: six-split mean of OL >= 95.0
json.dump(OUT, open(f'{ROOT}/runs/ido/final/ol_final.json', 'w'), indent=1); print('claims', json.dumps(OUT['claims'])); print('OL_FINAL_DONE')
