# assemble.py -- main.tex = pre.tex + body0-3.tex, the placeholders \OL... (capitals only) replaced by the texts in texts/*.tex
import os, re
H = os.path.dirname(os.path.abspath(__file__))
src = ''.join(open(f'{H}/{f}', encoding='utf-8').read() for f in ('pre.tex', 'body0.tex', 'body1.tex', 'body2.tex', 'body3.tex'))
for k in re.findall(r'\\(OL[A-Z]+)(?![A-Za-z])', src):
    p = f'{H}/texts/{k}.tex'
    if os.path.exists(p): src = src.replace('\\' + k, open(p, encoding='utf-8').read().strip())
left = re.findall(r'\\(OL[A-Z]+)(?![A-Za-z])', src)
open(f'{H}/main.tex', 'w', encoding='utf-8').write(src)
print('main.tex written;', 'placeholders left:', left)
