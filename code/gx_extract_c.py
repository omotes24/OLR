# gx_extract_c.py -- PROTOCOL_IDO.md I61 part II (exploration on val2): block A (the first decoder block with a 16x16 output) read per
#   CHANNEL: for every channel its amplitude (log max over positions |h - spatial mean|), log std over positions and spatial mean, for the
#   image and for its upside-down copy; the pass, hooks and shared noise image of code/ol_score.py (sigma_1 = 0.01).
#   usage: python [code/rfx_run.py] code/gx_extract_c.py --tag <tag> --gpu 0 --seed 0 --sets runs/ido/gx_edmc10.json --label c
#   -> runs/ido/gx/<tag>_s<seed>_<label>.npz   (V__<set> [N, 2 orientations, 3 kinds (amp, lstd, mean), channels] float32)
import os, sys, re, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, PATS
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--sets', required=True); p.add_argument('--label', required=True); p.add_argument('--prefix', default='devall-,fresh-,dqv-')
a = p.parse_args(); SIGMA = 0.01; PRE = tuple(a.prefix.split(','))
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
PART = {h[0]: h[1] for h in meta['hooks']}; RES = {h[0]: h[3] for h in meta['hooks']}; A = [h for h in HOOKS if PART[h] == 'dec' and RES[h] == 16][0]
ROWS = [r for r in json.load(open(f'{ROOT}/{a.sets}')) if r[0].startswith(PRE)]; assert ROWS and not any(r[0].startswith(('sel-', 'test')) for r in ROWS)
REF = [r[0] for r in ROWS if r[0].startswith('devall-') and '__' not in r[0]]; assert len(REF) == 1


def stats(o):
    o = o.float(); m = o.mean((2, 3)); c = o - m[:, :, None, None]; e = 1e-20
    return torch.stack([c.abs().amax((2, 3)).clamp_min(e).log(), c.pow(2).mean((2, 3)).sqrt().clamp_min(e).log(), m], 1)          # [B, 3, C]


@torch.no_grad()
def measure(row):
    _, data, idx, pre = row; idx = np.asarray(idx); xu = np.load(f'{NPY}/{data}.npy', mmap_mode='r'); V = []
    for i in range(0, len(idx), bs):
        x = to_t(np.ascontiguousarray(xu[idx[i:i + bs]]), dev)
        if pre != 'none': x = TF[pre][0](x)
        eps = shared_noise(x.shape[1:], a.seed, 0, 0, dev); two = []
        for y in (x, TF['vflip'][0](x)):
            store.clear(); den(y + SIGMA * eps, SIGMA); two.append(stats(store[A]))
        V.append(torch.stack(two, 1).float().cpu().numpy())
    return np.concatenate(V), idx


t0 = time.time(); out = dict(A=A, ref=REF[0], names=np.array([r[0] for r in ROWS]), kinds=np.array(['amp', 'lstd', 'mean']), seed=a.seed, sigma=SIGMA)
for r in ROWS:
    out[f'V__{r[0]}'], out[f'idx__{r[0]}'] = measure(r); print(a.tag, r[0], len(r[2]), out[f'V__{r[0]}'].shape, f'{time.time() - t0:.0f}s', flush=True)
np.savez(f'{ROOT}/runs/ido/gx/{a.tag}_s{a.seed}_{a.label}.npz', **out)
print('GX_EXTRACT_C_DONE', a.tag, a.seed, a.label, 'A', A, 'channels', out[f'V__{REF[0]}'].shape[-1], 'sets', len(ROWS), flush=True)
