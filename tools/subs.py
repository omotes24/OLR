# subs.py -- the path substitutions applied to the GPU-side files (code/) when they were copied from the GPU server into this
#   repository: the project root and the Python interpreter of the server are replaced by neutral ones.  Every substitution is exact.
#   (The other change, the removal of the lines about methods outside the paper, is described in code/README.md and in
#   protocols/PROTOCOL_OLR.md, I80; it is not invertible.)
SUBS = [
    ('source ~/miniforge3/etc/profile.d/conda.sh && conda activate dg && ', ': && '),     # activate your own environment instead
    ('PY=~/miniforge3/envs/dg/bin/python', 'PY=${PY:-python}'),
    ('PY=$HOME/miniforge3/envs/dg/bin/python', 'PY=${PY:-python}'),
    ('~/dragon_glider', '~/olr'),
    ('$HOME/dragon_glider', '$HOME/olr'),
    ('{HOME}/dragon_glider', '{HOME}/olr'),
]

if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        s = open(p, encoding='utf-8').read()
        for a, b in SUBS: s = s.replace(a, b)
        open(p, 'w', encoding='utf-8').write(s)
