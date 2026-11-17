# rfx_run.py -- PROTOCOL_IDO.md I47: runs a script unchanged (here: ol_score.py, gx_extract.py, gx_extract_c.py) with a tag that is not in
#   ct_extract_defs.SPEC.  No frozen file is edited; the tag is added at run time, as run_hires.py / e11hr_run.py do.
#     in256d               the ADM 256 model on the 112-px control images of PROTOCOL_HIRES2.md (bank set hc112_imagenet_bank), batch HR_BS
#     <id>_rinit           the model of <id> with freshly initialised weights (the constructor of its own library, torch seed 0)
#     <id>_rshuf           the model of <id> with the entries of every parameter tensor permuted at random (torch seed 0): the same values,
#                          no learned structure
#     <id>_eq50            the trained model of <id> on the images of mk_rfx_eq.py (every image reduced to half its side and brought back)
#     <id>_by_<model>      the images and calibration of <id> passed through the trained model of <model> (same input size)
#     <id>_eq50k           as <id>_eq50 with the bank set eq50_<id>_bank (I54);  in256e / in256g  the stronger 256x256 controls of I54
#   <id>, <model> in edmc10, l6svhn, dpceleba, in64, fmnist.   usage: [HR_BS=4] python code/rfx_run.py code/<script>.py ... --tag <tag> ...
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import dgcore, ct_extract_defs
SPEC = ct_extract_defs.SPEC; BASE = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; ROOT = dgcore.ROOT
tag = sys.argv[sys.argv.index('--tag') + 1]
if tag in ('in256d', 'in256e', 'in256g'):                    # in256e, in256g: the stronger controls of I54 (their sets files hold the images)
    SPEC[tag] = ('adm:256', 'hc112_imagenet_bank', int(os.environ.get('HR_BS', '4')))
elif tag.endswith('_eq50k'):                                  # I54: as <id>_eq50, with the bank set eq50_<id>_bank (not used by OLR)
    b = tag[:-6]; assert b in BASE, tag; SPEC[tag] = (SPEC[b][0], f'eq50_{b}_bank', SPEC[b][2])
elif tag.endswith('_eq50'):                                   # I49: the trained model of <id>; the sets files of the tag hold the half-resolution control images
    b = tag[:-5]; assert b in BASE, tag; SPEC[tag] = SPEC[b]
elif tag.endswith(('_rinit', '_rshuf')):
    b, v = tag.rsplit('_', 1); assert b in BASE, tag; SPEC[tag] = (f'{v}|{SPEC[b][0]}', SPEC[b][1], min(SPEC[b][2], 100))   # fp32 runs: smaller batches (11-GB GPU)
elif '_by_' in tag:
    b, m = tag.split('_by_'); assert b in BASE and m in BASE and b != m, tag; SPEC[tag] = (SPEC[m][0], SPEC[b][1], SPEC[b][2])
else: raise SystemExit(f'unknown tag {tag}')
_load_model = dgcore.load_model


def fresh(spec, net):
    """a freshly initialised copy of the network of `spec` (the constructor of its library; nothing is loaded)"""
    kind, p = spec.split(':', 1)
    if kind in ('edm', 'edm2'): return type(net)(*net.init_args, **net.init_kwargs)                      # NVlabs persistence: the stored constructor arguments
    if kind == 'iddpm':
        from improved_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
        a = model_and_diffusion_defaults(); a.update(image_size=32, num_channels=128, num_res_blocks=3, learn_sigma=True, diffusion_steps=4000, noise_schedule='cosine', dropout=0.3)
        return create_model_and_diffusion(**a)[0]
    if kind == 'l6':
        from diffusers import UNet2DModel
        return UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2, block_out_channels=[128, 128, 256, 256, 512, 512],
                           down_block_types=('DownBlock2D',) * 4 + ('AttnDownBlock2D', 'DownBlock2D'), up_block_types=('UpBlock2D', 'AttnUpBlock2D') + ('UpBlock2D',) * 4)
    if kind == 'own':
        from dgnet import build_net
        return build_net(torch.load(f'{ROOT}/runs/{p}', map_location='cpu', weights_only=False)['cfg'])
    raise ValueError(spec)


def load_model(spec, device):
    if spec == 'adm:256':
        import admwrap; return admwrap.load_adm(device, fp16=os.environ.get('DG_FP16', '1') != '0')
    if '|' not in spec: return _load_model(spec, device)
    v, base = spec.split('|', 1); den, net = _load_model(base, device); torch.manual_seed(0)
    with torch.no_grad():
        if v == 'rinit':
            sd = fresh(base, net).state_dict(); cur = net.state_dict(); assert set(sd) == set(cur) and all(sd[k].shape == cur[k].shape for k in sd)
            net.load_state_dict(sd)
        elif v == 'rshuf':
            g = torch.Generator().manual_seed(0); n = 0
            for name, p in net.named_parameters():
                if p.numel() > 1: p.copy_(p.detach().flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p)); n += 1
            print(f'[rfx] {n} parameter tensors permuted', flush=True)
        else: raise ValueError(v)
    net.to(device).eval().requires_grad_(False)
    return den, net


dgcore.load_model = load_model
script = sys.argv[1]; sys.argv = [os.path.basename(script)] + sys.argv[2:]
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
