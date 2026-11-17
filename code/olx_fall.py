# olx_fall.py -- PROTOCOL_IDO.md I73, addendum to X9: does the amplitude fall?  With a margin, because the sign alone is decided by shifts far
#   below the spread of the images (the val2 ID images lie below the calibration mean in 94% of the blocks for CIFAR-10 and in 0% for SVHN).
#   val2, noise seeds 0-2, stored statistics, blocks up to A.  z_h = (mean of a set at block h - ID mean at h) / SD of the ID amplitude at h
#   (calibration images).  Shares of blocks with z_h < -0.2 (falls) and z_h > +0.2 (rises), for the mean over the seven re-orientations of the
#   calibration images, for each val2 OOD set (mean, min, max over the sets) and for the val2 ID images (reference); z at block A.
#   usage: python code/olx_fall.py   -> runs/ido/final/olx_fall.json
import os, json
import numpy as np
ROOT = os.path.expanduser('~/olr'); BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEEDS = (0, 1, 2); O7 = ['v', 'h', 'r180', 'r90', 'r270', 't1', 't2']; MG = 0.2
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
OUT = {'margin': MG, 'tags': {}}
for t in BASE:
    acc = []
    for sd in SEEDS:
        ld = lambda lab: np.load(f'{ROOT}/runs/ido/gx/{t}_s{sd}_{lab}.npz'); Z, N, N2 = ld('v'), ld('n'), ld('n2')
        iA = int(Z['iA']); ref = str(Z['ref']); L = len(Z['part']); ca = Z[f'V__{ref}'][:, 0, 0, 1:]; ok = ca.std(0) > 1e-9; cs = [h for h in range(L) if ok[h] and h <= iA]; assert cs[-1] == iA
        K = [Z[f'V__{ref}'][:, 1, 0, 1:], N2[f'V__{ref}__hf'][:, 0, 0, 1:], N2[f'V__{ref}__hf'][:, 1, 0, 1:], N[f'V__{ref}__r90'][:, 0, 0, 1:], N[f'V__{ref}__r270'][:, 0, 0, 1:],
             N[f'V__{ref}__r90'][:, 1, 0, 1:], N[f'V__{ref}__r270'][:, 1, 0, 1:]]
        mu, sdv = ca[:, cs].mean(0), ca[:, cs].std(0); z = lambda m: (m[cs] - mu) / sdv
        names = [str(n) for n in Z['names']]; idn = f'fresh-{IDD[t]}'; nat = [n for n in names if n.startswith('fresh-') and n not in (idn, idn + '__vflip', idn + '__hflip') and 'synth' not in n]
        zo = z(np.mean([k.mean(0) for k in K], 0)); zi = z(Z[f'V__{idn}'][:, 0, 0, 1:].mean(0)); zs = {o: z(Z[f'V__{o}'][:, 0, 0, 1:].mean(0)) for o in nat}
        sh = lambda v: dict(fall=float(np.mean(v < -MG)), rise=float(np.mean(v > MG)), atA=float(v[-1]))
        acc.append(dict(orient=sh(zo), id_val2=sh(zi), ood={o: sh(v) for o, v in zs.items()}, p=len(cs)))
    nat = list(acc[0]['ood']); m3 = lambda f: float(np.mean([f(a) for a in acc]))
    OUT['tags'][t] = dict(p=acc[0]['p'], orient={k: m3(lambda a: a['orient'][k]) for k in ('fall', 'rise', 'atA')}, id_val2={k: m3(lambda a: a['id_val2'][k]) for k in ('fall', 'rise', 'atA')},
                          ood_sets={o: {k: m3(lambda a: a['ood'][o][k]) for k in ('fall', 'rise', 'atA')} for o in nat})
    d = OUT['tags'][t]; d['ood'] = {k: float(np.mean([v[k] for v in d['ood_sets'].values()])) for k in ('fall', 'rise', 'atA')}
    d['ood_fall_min'] = float(min(v['fall'] for v in d['ood_sets'].values())); d['ood_fall_max'] = float(max(v['fall'] for v in d['ood_sets'].values())); d['ood_atA_max'] = float(max(v['atA'] for v in d['ood_sets'].values()))
    print(f"{t:9s} p {d['p']:2d} | re-oriented: falls {d['orient']['fall']:.2f} rises {d['orient']['rise']:.2f} z at A {d['orient']['atA']:+.2f} | OOD: falls {d['ood']['fall']:.2f} ({d['ood_fall_min']:.2f}-{d['ood_fall_max']:.2f}) rises {d['ood']['rise']:.2f} z at A {d['ood']['atA']:+.2f} (max {d['ood_atA_max']:+.2f}) | val2 ID: falls {d['id_val2']['fall']:.2f} rises {d['id_val2']['rise']:.2f} z at A {d['id_val2']['atA']:+.2f}", flush=True)
json.dump(OUT, open(f'{ROOT}/runs/ido/final/olx_fall.json', 'w'), indent=1); print('OLX_FALL_DONE')
