# ol_score.py -- the orientation likelihood-ratio score (PROTOCOL_IDO.md I68; frozen there).  The frozen denoiser, the test image, and the
#   val1 ID images (devall-<ID>).  C1 as reworded in I68: nothing trained by gradient descent, no further network, no human label, no OOD
#   image (two mean vectors and one covariance matrix in closed form; "no trained classifier" is NOT claimed).
#   For an image x (uint8 -> [-1, 1]; the set's own transform first), ONE pass at sigma_1 = 0.01 with the shared noise image
#   dgcore.shared_noise(seed, 0, 0); per block h (the hooks matched by ct_extract_defs.PATS, forward order) the amplitude
#       a_h(x) = mean over channels of log max over positions |h - spatial mean of the channel|.
#   The profile a(x) = (a_h)_h over the blocks up to A, the first decoder block whose output is 16x16 (a deployment stops the pass there).
#   From the val1 ID images:  m, C = mean and covariance of their profiles (C + 1e-3 * (mean variance) * I; blocks with zero variance are
#   left out);  m_o = the mean over the seven other orientations of the square (upside-down, mirrored, half turn, the two quarter turns,
#   the two diagonal reflections) of the mean profile of the same images in that orientation.
#       score(x) = -(a - m*)^T C^-1 (a - m*),     m* = m + c (m - m_o),   c = 8          (larger = more in-distribution)
#   Up to a constant this is -d2 + 2 c l with d2 = (a - m)^T C^-1 (a - m) and l = (m - m_o)^T C^-1 (a - m): the Gaussian log-likelihood
#   ratio between the ID profiles N(m, C) and N(m_o, (1 + 1/c) C).
#   Also stored (registered ablations): d2, l, and sa = the same score on the profile of all blocks (a full pass).
#   usage: python code/ol_score.py --tag edmc10 --gpu 0 --seed 0 --prefix test7-,test6- --label final [--sets runs/ido/gx_edmc10_eq50.json]
#   -> runs/ido/ol/<tag>_s<seed>_<label>.npz        (--sets: a further sets file, read after the standard ones)
import os, sys, re, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, PATS
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--prefix', required=True); p.add_argument('--label', required=True); p.add_argument('--sets', default='')
a = p.parse_args(); PRE = tuple(a.prefix.split(',')); assert not any(q.startswith(('sel-', 'devall-')) for q in PRE)
SIGMA, RIDGE, CC = 0.01, 1e-3, 8.0
spec, bankset, bs = SPEC[a.tag]; dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev); mods = dict(net.named_modules()); cand = [n for n in mods if any(re.search(q, n) for q in PATS)]; store, order = {}, []


def grab(name):
    def f(m, inp, o):
        if isinstance(o, (tuple, list)): o = o[0]
        if torch.is_tensor(o) and o.ndim == 4:
            if name not in store: order.append(name)
            store[name] = o
    return f


for n in cand: mods[n].register_forward_hook(grab(n))
with torch.no_grad():
    den(to_t(np.ascontiguousarray(np.load(f'{NPY}/{bankset}.npy', mmap_mode='r')[:2]), dev), 0.2)
HOOKS = list(order); meta = json.load(open(f'{ROOT}/runs/hx/s0/{a.tag}/meta.json')); assert HOOKS == [h[0] for h in meta['hooks']]
PART = {h[0]: h[1] for h in meta['hooks']}; RES = {h[0]: h[3] for h in meta['hooks']}
iA = [i for i, h in enumerate(HOOKS) if PART[h] == 'dec' and RES[h] == 16][0]; upA = list(range(iA + 1)); FULL = list(range(len(HOOKS)))
ALL = {}
for f_ in [f'runs/ok/sets_{a.tag}.json', f'runs/e11/sets_{a.tag}.json', f'runs/hx/sets1_{a.tag}.json', f'runs/hx/sets3_{a.tag}.json'] + [f'runs/ido/sets{i}_{a.tag}.json' for i in (4, 5, 6, 7, 8, 9)] + ([a.sets] if a.sets else []):
    if os.path.exists(f'{ROOT}/{f_}'):
        for r in json.load(open(f'{ROOT}/{f_}')): ALL.setdefault(r[0], r)
REF = [n for n in ALL if n.startswith('devall-') and '__' not in n]; assert len(REF) == 1; REF = REF[0]
NAMES = [REF] + [n for n in ALL if n.startswith(PRE)]; assert not any(n.startswith('sel-') for n in NAMES)
VF = TF['vflip'][0]; q1 = lambda x: torch.rot90(x, 1, (2, 3)); q3 = lambda x: torch.rot90(x, 3, (2, 3)); mir = lambda x: x.flip(3)
ORI = [VF, mir, lambda x: VF(mir(x)), q1, q3, lambda x: VF(q1(x)), lambda x: VF(q3(x))]      # the seven other orientations of the square


def amp(o):
    o = o.float(); return (o - o.mean((2, 3), keepdim=True)).abs().amax((2, 3)).clamp_min(1e-20).log().mean(1)


def prof(y, eps):
    store.clear(); den(y + SIGMA * eps, SIGMA); return torch.stack([amp(store[h]) for h in HOOKS], 1).double().cpu().numpy()   # a deployment stops this pass after block A


@torch.no_grad()
def measure(name, copies=False):
    _, data, idx, pre = ALL[name]; idx = np.asarray(idx); xu = np.load(f'{NPY}/{data}.npy', mmap_mode='r'); V, O = [], []
    for i in range(0, len(idx), bs):
        x = to_t(np.ascontiguousarray(xu[idx[i:i + bs]]), dev)
        if pre != 'none': x = TF[pre][0](x)
        eps = shared_noise(x.shape[1:], a.seed, 0, 0, dev); V.append(prof(x, eps))
        if copies: assert x.shape[2] == x.shape[3]; O.append(np.stack([prof(f(x), eps) for f in ORI]))      # calibration only: [7, B, blocks]
    return np.concatenate(V), (np.concatenate(O, 1) if copies else None), idx


def fit(Vr, Or, cols):
    ok = Vr[:, cols].std(0) > 1e-9; use = [c for c, k in zip(cols, ok) if k]; m = Vr[:, use].mean(0); C = np.atleast_2d(np.cov(Vr[:, use].T)); C = C + RIDGE * np.mean(np.diag(C)) * np.eye(len(use))
    mo = Or[:, :, use].mean(1).mean(0); Ci = np.linalg.inv(C); return dict(use=use, m=m, Ci=Ci, ms=m + CC * (m - mo), w=Ci @ (m - mo))


def score(V, P): d = V[:, P['use']] - P['ms']; return -np.einsum('ij,jk,ik->i', d, P['Ci'], d)


t0 = time.time(); Vr, Or, ixr = measure(REF, True); PU, PF = fit(Vr, Or, upA), fit(Vr, Or, FULL)
out = dict(names=np.array(NAMES), ref=REF, hooks=np.array(HOOKS), A=HOOKS[iA], n_blocks_used=len(PU['use']), n_blocks_used_all=len(PF['use']), sigma=SIGMA, ridge=RIDGE, c=CC, seed=a.seed, m=PU['m'], m_star=PU['ms'], Ci=PU['Ci'])
for n in NAMES:
    V, idx = (Vr, ixr) if n == REF else measure(n)[::2]; d = V[:, PU['use']] - PU['m']
    out[f'score__{n}'] = score(V, PU); out[f'sa__{n}'] = score(V, PF); out[f'd2__{n}'] = np.einsum('ij,jk,ik->i', d, PU['Ci'], d); out[f'l__{n}'] = d @ PU['w']; out[f'idx__{n}'] = idx
    print(a.tag, n, len(idx), f'{time.time() - t0:.0f}s', flush=True)
os.makedirs(f'{ROOT}/runs/ido/ol', exist_ok=True); np.savez(f'{ROOT}/runs/ido/ol/{a.tag}_s{a.seed}_{a.label}.npz', **out)
print('OL_SCORE_DONE', a.tag, f's{a.seed}', a.label, 'A =', HOOKS[iA], 'blocks in the profile:', len(PU['use']), 'of', iA + 1, f'{time.time() - t0:.0f}s', flush=True)
