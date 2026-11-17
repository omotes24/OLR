# mk_data.py -- writes the TikZ data snippets of the illustrated resume (figs/d_*.tex).
#   d_dots.tex : the schematic clouds of Figures 4 and 6 in the whitened coordinates (one unit = one ID standard deviation):
#                ID profiles ~ N(0, I) (navy), re-oriented profiles ~ N((-Delta, 0), 0.93^2 I) (grey), Delta = 1.77 (CIFAR-10, olx_info.json).
#                The spread 0.93 = sqrt(0.87), the trace ratio of the re-oriented covariance for CIFAR-10 (supplement Table A5).
#   d_cells.tex: one 16x16 channel map (values) and its absolute deviation from the spatial mean (schematic; random values with one bump).
import numpy as np, os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
DELTA = 1.768
rng = np.random.default_rng(20261117)
idp = rng.standard_normal((44, 2))
orp = rng.standard_normal((44, 2)) * 0.93 + np.array([-DELTA, 0.0])
def keep(p, ylim=2.45, xlo=-4.4):
    return [q for q in p if abs(q[1]) < ylim and q[0] > xlo]
with open(os.path.join(out, 'd_dots.tex'), 'w') as f:
    f.write('%% written by gen/mk_data.py: schematic clouds in whitened coordinates (unit = one ID standard deviation), Delta = %.3f\n' % DELTA)
    f.write('\\def\\iddots{' + ', '.join('%.3f/%.3f' % tuple(q) for q in keep(idp)) + '}\n')
    f.write('\\def\\ordots{' + ', '.join('%.3f/%.3f' % tuple(q) for q in keep(orp)) + '}\n')
    f.write('\\def\\mDelta{%.3f}\n' % DELTA)
# channel map: smooth background around a level, one strong local response at (row 4, col 11)
g = np.zeros((16, 16))
yy, xx = np.mgrid[0:16, 0:16]
g += 0.35 + 0.06 * rng.standard_normal((16, 16))
g += 0.10 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
g += 0.55 * np.exp(-((yy - 4) ** 2 + (xx - 11) ** 2) / 1.6)
dev = np.abs(g - g.mean())
imax = np.unravel_index(np.argmax(dev), dev.shape)
def shade(v, lo, hi):
    return int(round(8 + 72 * (v - lo) / (hi - lo)))
with open(os.path.join(out, 'd_cells.tex'), 'w') as f:
    f.write('% written by gen/mk_data.py: schematic 16x16 channel map (row-major from the top left); darker = larger\n')
    f.write('\\def\\cellsraw{%\n')
    for r in range(16):
        f.write(''.join('\\fill[black!%d] (%d,%d) rectangle ++(1,-1);' % (shade(g[r, c], g.min(), g.max()), c, -r) for c in range(16)) + '%\n')
    f.write('}\n\\def\\cellsdev{%\n')
    for r in range(16):
        f.write(''.join('\\fill[black!%d] (%d,%d) rectangle ++(1,-1);' % (shade(dev[r, c], 0, dev.max()), c, -r) for c in range(16)) + '%\n')
    f.write('}\n')
    f.write('\\def\\cellmaxrow{%d}\\def\\cellmaxcol{%d}\n' % imax)
print('max cell', imax, 'id dots', len(keep(idp)), 'or dots', len(keep(orp)))
# schematic per-block mean amplitudes for Figure 3(b) (drawing coordinates in cm; NOT data):
#   ID mean (navy) with a +-1 SD band; seven re-oriented means (grey) that fall in the later blocks (about 1.6 SD at A, as for CIFAR-10);
#   an OOD mean (red) that rises a little in the first blocks and falls more later (about 2.2 SD at A).
P = 23
X0, X1 = 8.35, 16.4
xs = [X0 + (X1 - X0) * k / (P - 1) for k in range(P)]
kk = np.arange(1, P + 1)
m = -2.05 + 0.32 * np.sin(0.42 * kk) + 0.10 * np.cos(1.3 * kk)
SD = 0.24
ramp = np.clip((kk - 5) / 18.0, 0, None)
rng2 = np.random.default_rng(7)
ors = []
for g in range(7):
    amp = 1.6 * (0.75 + 0.5 * g / 6.0)
    ors.append(m - SD * (amp * ramp + 0.12 * rng2.standard_normal(P) * (ramp > 0)))
dro = np.where(kk <= 3, -0.45, 2.2 * np.clip((kk - 3) / 20.0, 0, None)) + 0.15 * rng2.standard_normal(P) * (kk > 3)
ood = m - SD * dro
def path(y):
    return ' -- '.join('(%.3f,%.3f)' % (x, v) for x, v in zip(xs, y))
with open(os.path.join(out, 'd_prof.tex'), 'w') as f:
    f.write('%% written by gen/mk_data.py: schematic per-block mean amplitudes (drawing coordinates, cm), p = %d\n' % P)
    f.write('\\def\\profid{%s}\n' % path(m))
    f.write('\\def\\profidhi{%s}\n' % path(m + SD))
    f.write('\\def\\profidlo{%s}\n' % path(m - SD))
    f.write('\\def\\profors{%s}\n' % ' '.join(path(y) for y in ors))
    f.write('\\def\\profood{%s}\n' % path(ood))
    f.write('\\def\\profxA{%.3f}\\def\\profyA{%.3f}\\def\\profyAor{%.3f}\\def\\profyAood{%.3f}\n' % (xs[-1], m[-1], np.mean([o[-1] for o in ors]), ood[-1]))
    f.write('\\def\\profxone{%.3f}\n' % xs[0])
print('prof ok; at A: id %.2f or %.2f ood %.2f' % (m[-1], np.mean([o[-1] for o in ors]), ood[-1]))
