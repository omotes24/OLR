# e11hr_run.py -- PROTOCOL_E11HR.md: code/run_hires2.py (the runner of the 256 / 512 tiers) with an optional batch size HR_BS for the
#   in256c tier, so that the ADM 256 model fits next to paused training processes; per-image features do not depend on the batch size.
#   usage: [HR_BS=4] python code/e11hr_run.py code/<script>.py <its arguments, with --tag in256c>
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import dgcore, ct_extract_defs
BS = int(os.environ.get('HR_BS', '8'))
ct_extract_defs.SPEC['in256c'] = ('adm:256', 'hc_imagenet_bank', BS)
_load_model = dgcore.load_model


def load_model(spec, device):
    if spec == 'adm:256':
        import admwrap; return admwrap.load_adm(device, fp16=os.environ.get('DG_FP16', '1') != '0')
    return _load_model(spec, device)


_to_t = dgcore.to_t


def to_t(x, device):
    if getattr(x, 'dtype', None) == np.float16:
        return torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).float()
    return _to_t(x, device)


dgcore.load_model = load_model; dgcore.to_t = to_t; ct_extract_defs.to_t = to_t
script = sys.argv[1]
sys.argv = [os.path.basename(script)] + sys.argv[2:]
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
