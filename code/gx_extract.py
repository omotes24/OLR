# gx_extract.py -- PROTOCOL_IDO.md I59 (exploration on val2): per image, two one-number statistics of every U-Net block AND of the network
#   input itself, for the image and for its upside-down copy, with the pass, hooks and shared noise image of code/ol_score.py
#   (sigma_1 = 0.01):  amp = mean over channels of log max over positions |h - spatial mean| (the amplitude of the frozen method),
#   lstd = mean over channels of log std over positions.  Position 0 of the last axis is the input x + sigma eps, positions 1.. the blocks.
#   usage: [DG_FP16=0] python code/rfx_run.py code/gx_extract.py --tag <tag> --gpu 0 --seed 0 --sets runs/ido/gx_edmc10.json --label v
#          (base tags may be run without rfx_run.py)      -> runs/ido/gx/<tag>_s<seed>_<label>.npz   (V__<set> [N, 2 orientations, 2 kinds, 1 + blocks])
import os, sys, re, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, PATS
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--sets', required=True); p.add_argument('--label', required=True); p.add_argument('--prefix', default='devall-,fresh-,dqv-')
a = p.parse_args(); SIGMA = 0.01; KINDS = ['amp', 'lstd']; PRE = tuple(a.prefix.split(','))
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
PART = {h[0]: h[1] for h in meta['hooks']}; RES = {h[0]: h[3] for h in meta['hooks']}; iA = [i for i, h in enumerate(HOOKS) if PART[h] == 'dec' and RES[h] == 16][0]
ROWS = [r for r in json.load(open(f'{ROOT}/{a.sets}')) if r[0].startswith(PRE)]; assert ROWS and not any(r[0].startswith(('sel-', 'test')) for r in ROWS)
REF = [r[0] for r in ROWS if r[0].startswith('devall-') and '__' not in r[0]]; assert len(REF) == 1


def stats(o):
    o = o.float(); c = o - o.mean((2, 3), keepdim=True); e = 1e-20
    return torch.stack([c.abs().amax((2, 3)).clamp_min(e).log().mean(1), c.pow(2).mean((2, 3)).sqrt().clamp_min(e).log().mean(1)], 1)          # [B, 2]


@torch.no_grad()
def measure(row):
    _, data, idx, pre = row; idx = np.asarray(idx); xu = np.load(f'{NPY}/{data}.npy', mmap_mode='r'); V = []
    for i in range(0, len(idx), bs):
        x = to_t(np.ascontiguousarray(xu[idx[i:i + bs]]), dev)
        if pre != 'none': x = TF[pre][0](x)
        eps = shared_noise(x.shape[1:], a.seed, 0, 0, dev); two = []
        for y in (x, TF['vflip'][0](x)):
            xn = y + SIGMA * eps; store.clear(); den(xn, SIGMA); two.append(torch.stack([stats(xn)] + [stats(store[h]) for h in HOOKS], 2))      # [B, 2, 1 + L]
        V.append(torch.stack(two, 1).double().cpu().numpy())
    return np.concatenate(V), idx


t0 = time.time(); out = dict(hooks=np.array(HOOKS), part=np.array([PART[h] for h in HOOKS]), res=np.array([RES[h] for h in HOOKS]), iA=iA, ref=REF[0], names=np.array([r[0] for r in ROWS]), kinds=np.array(KINDS), seed=a.seed, sigma=SIGMA)
for r in ROWS:
    out[f'V__{r[0]}'], out[f'idx__{r[0]}'] = measure(r); print(a.tag, r[0], len(r[2]), f'{time.time() - t0:.0f}s', flush=True)
os.makedirs(f'{ROOT}/runs/ido/gx', exist_ok=True); np.savez(f'{ROOT}/runs/ido/gx/{a.tag}_s{a.seed}_{a.label}.npz', **out)
print('GX_EXTRACT_DONE', a.tag, a.seed, a.label, 'blocks', len(HOOKS), 'A', HOOKS[iA], 'sets', len(ROWS), flush=True)
