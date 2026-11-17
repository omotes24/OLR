# mk_tabs.py SRC OUT -- the table images of the OLR slides (2026-11-17).  Every number is read from the rows that the resume's
#   generator wrote (SRC = the resume source: tab_*_ol.tex, supp_*_ol.tex), so the slides and the resume cannot differ.
#   Style of the earlier slides: Times (newtx) and Hiragino Mincho, booktabs, no vertical rules, full-width brackets, the slides' own [n].
#   Marks: navy bold where OLR ties or beats the best prior method (the resume's light-blue cells), dagger for the best prior method,
#   red for values below 70 % on the background slide.  Compiled on the Mac with lualatex (Hiragino), then 600 dpi PNG.
import io, os, re, sys

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
REFS = ['song2021score', 'karras2022edm', 'nalisnick2019deep', 'serra2020input', 'nalisnick2019typ', 'mahmood2021msma', 'heng2024diffpath',
        'alexis2026gepc', 'shoushtari2026eigenscore', 'rouzoumka2026cfs', 'xiang2023ddae', 'ronneberger2015unet', 'lee2018mahalanobis',
        'hendrycks2019oe', 'dhariwal2021adm', 'zhang2023openood', 'ren2019likelihood', 'kamkari2024geometric', 'nichol2021improved',
        'karras2024edm2', 'tack2020csi']
NUM = {k: i + 1 for i, k in enumerate(REFS)}

PRE = r"""\documentclass[border=3pt]{standalone}
\usepackage{luatexja-fontspec}
\setmainjfont[Scale=1,BoldFont=HiraMinProN-W6]{Hiragino Mincho ProN}
\usepackage{newtxtext}
\usepackage{newtxmath}
\ltjsetparameter{xkanjiskip=0pt}
\usepackage{booktabs,xcolor}
\definecolor{tred}{RGB}{192,57,43}
\definecolor{tnavy}{RGB}{31,58,122}
\newcommand{\R}[1]{\textcolor{tred}{#1}}
\newcommand{\B}[1]{\textcolor{tnavy}{\textbf{#1}}}
\newcommand{\U}[1]{\textcolor{tnavy}{\underline{#1}}}
\newcommand{\dg}{\rlap{$^{\dagger}$}}
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{5pt}
\begin{document}%
"""
END = '\\end{document}\n'
WRITTEN = []


def wr(name, body):
    io.open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8').write(PRE + body + END)
    WRITTEN.append(name)


def rows(f):
    out = []
    for l in io.open(os.path.join(SRC, f), encoding='utf-8'):
        l = l.rstrip('\n')
        if l.startswith('%'):
            continue
        s = l.strip()
        if s.endswith('\\bottomrule'):
            s = s[:-len('\\bottomrule')]
        if s == '\\midrule':
            out.append(None)
            continue
        if s.endswith('\\\\'):
            out.append([c.strip() for c in s[:-2].split('&')])
    return out


def name(c):
    """a label cell of the resume -> the slide's label: full-width brackets, the slides' reference numbers, OLR(Ours) bold"""
    c = c.replace('\\cellcolor{firstblue}', '').strip()
    c = re.sub(r'~?\\cite\{([^}]+)\}', lambda m: '[%s]' % ', '.join(str(NUM[k.strip()]) for k in m.group(1).split(',')), c)
    c = c.replace('\\textbf{\\Ours(Ours)}', '\\textbf{OLR（Ours）}').replace('\\Ours{}', 'OLR').replace('\\Ours', 'OLR')
    c = c.replace('(', '（').replace(')', '）').replace('kNN ヘッド', 'kNNヘッド')
    c = c.replace('\\textbf{OLR}', '\\textbf{OLR}')
    return c


def num(c, dagger=True, red_below=None):
    """a value cell: the resume's light-blue bold -> navy bold, its dagger kept (as an rlap), its underline -> navy underline"""
    blue = 'firstblue' in c
    ul = '\\underline' in c
    c = c.replace('\\cellcolor{firstblue}', '').replace('\\textbf{', '').replace('\\boldmath', '').replace('\\underline{', '')
    c = c.replace('}', '') if (blue or ul) else c
    dg = '$^{\\dagger}$' in c
    c = c.replace('$^{\\dagger}$', '')
    c = c.replace('--', '–')
    if red_below is not None:
        m = re.match(r'^\d+\.\d$', c)
        if m and float(c) < red_below:
            c = '\\R{%s}' % c
    if blue:
        c = '\\B{%s}' % c
    elif ul:
        c = '\\U{%s}' % c
    if dg and dagger:
        c += '\\dg'
    return c


IDS5 = 'CIFAR-10 & SVHN & CelebA & ImageNet-64 & FashionMNIST'

# ---------------------------------------------------------------- p.2 background: six prior rows of the main table, red below 70
M = rows('tab_main_ol.tex')
P = [r for r in M if r]
assert len(P) == 11 and 'OLR' in name(P[10][0]), [name(r[0]) for r in P]
keep = {'NLL（尤度）[1]': 'NLL（尤度）[1]', 'IC（PNG）[4]': 'IC（PNG）[4]', 'MSMA型[6]': 'MSMA型[6]', 'DiffPath[7]': 'DiffPath[7]',
        'CFS[10]': 'CFS[10]', 'CFS（kNNヘッド）[10]': 'CFS（kNNヘッド）[10]'}
L = [r'\begin{tabular}{l*{5}{r}rr}%', r'\toprule', r' & \multicolumn{5}{c}{分布内} & & \\', r'\cmidrule(lr){2-6}',
     r'手法 & ' + IDS5 + r' & 5つの平均 & 順伝播\\', r'\midrule']
got = 0
for r in P[:10]:
    n = name(r[0])
    if n not in keep:
        continue
    if n == 'CFS[10]':
        L.append(r'\midrule')
    v = [num(c, dagger=False, red_below=70) for c in r[1:7]] + [re.sub(r'\$\^\{[^}]*\}\$', '', r[9])]
    L.append(n + ' & ' + ' & '.join(v) + r'\\')
    got += 1
assert got == 6
wr('t_bg', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')

# ---------------------------------------------------------------- p.7 the main table
L = [r'\begin{tabular}{l*{5}{r}rrcrr}%', r'\toprule', r' & \multicolumn{5}{c}{分布内（自然画像の分布外集合の平均）} & & & OLRの & & \\', r'\cmidrule(lr){2-6}',
     r'手法 & ' + IDS5 + r' & 5つの平均 & 上下反転 & 勝/分/負 & 順伝播 & 秒/枚\\', r'\midrule']
for i, r in enumerate(P):
    if i in (8, 10):
        L.append(r'\midrule')
    v = [num(c) for c in r[1:8]] + [r[8].replace('--', '–'), r[9].replace('$^{\\S\\dagger}$', '$^{\\S}$\\dg'), r[10].replace('$^{\\dagger\\ddagger}$', '$^{\\ddagger}$\\dg')]
    L.append(name(r[0]) + ' & ' + ' & '.join(v) + r'\\')
wr('t_main', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')

# ---------------------------------------------------------------- p.3 / p.4 does the amplitude fall (val2, blocks up to A)
F = [r for r in rows('tab_fall_ol.tex') if r]
assert len(F) == 5
L = [r'\begin{tabular}{lrrrr}%', r'\toprule', r' & & \multicolumn{3}{c}{分布外の集合（集合ごとの値の平均）}\\', r'\cmidrule(lr){3-5}',
     r'分布内 & $p$ & 下がる & 上がる & $z_{\mathrm{A}}$\\', r'\midrule']
for r in F:
    L.append(' & '.join([r[0], r[1], r[5], r[6], r[7]]) + r'\\')
wr('t_fall_ood', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
L = [r'\begin{tabular}{lrrrrrr}%', r'\toprule', r' & \multicolumn{3}{c}{向きを変えた画像} & \multicolumn{3}{c}{val2の分布内（参考）}\\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
     r'分布内 & 下がる & 上がる & $z_{\mathrm{A}}$ & 下がる & 上がる & $z_{\mathrm{A}}$\\', r'\midrule']
for r in F:
    L.append(' & '.join([r[0], r[2], r[3], r[4], r[8], r[9], r[10]]) + r'\\')
wr('t_fall_or', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
L = [r'\begin{tabular}{lrrrrrrrrrr}%', r'\toprule', r' & & \multicolumn{3}{c}{向きを変えた画像} & \multicolumn{3}{c}{分布外の集合} & \multicolumn{3}{c}{val2の分布内}\\',
     r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}', r'分布内 & $p$ & 下がる & 上がる & $z_{\mathrm{A}}$ & 下がる & 上がる & $z_{\mathrm{A}}$ & 下がる & 上がる & $z_{\mathrm{A}}$\\', r'\midrule']
for r in F:
    L.append(' & '.join(r) + r'\\')
wr('t_fall_full', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
FS = [r for r in rows('supp_fallsign_ol.tex') if r]
L = [r'\begin{tabular}{lrrrrrrrr}%', r'\toprule', r' & & 向きを変えた & \multicolumn{3}{c}{分布外の集合} & val2の & \multicolumn{2}{c}{ブロックA}\\',
     r'\cmidrule(lr){4-6}\cmidrule(lr){8-9}', r'分布内 & $p$ & 画像 & 平均 & 最小 & 最大 & 分布内 & 向き & 分布外\\', r'\midrule']
for r in FS:
    L.append(' & '.join(r) + r'\\')
wr('t_fallsign', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')

# ---------------------------------------------------------------- p.8 the two terms (six-split means) and the constant c (val2)
S = rows('tab_six_ol.tex')
six = {}
for r in S:
    if r and len(r) == 10:
        six[re.sub(r'\\cellcolor\{firstblue\}|\\quad', '', r[0]).strip()] = r
L = [r'\begin{tabular}{l*{5}{r}rr}%', r'\toprule', r' & \multicolumn{5}{c}{分布内} & & \\', r'\cmidrule(lr){2-6}',
     r'6分割の平均 & ' + IDS5 + r' & 5つの平均 & 上下反転\\', r'\midrule']
for k, lab in (('6分割の平均', '\\textbf{OLR（Ours）}'), ('内積の項だけ', '内積の項だけ'), ('距離の項だけ', '距離の項だけ'), ('全ブロック', 'ブロックAで止めない')):
    r = six[k]
    L.append(lab + ' & ' + ' & '.join(c.replace('\\cellcolor{firstblue}', '') for c in r[1:8]) + r'\\')
wr('t_terms', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
DV = rows('tab_dev_ol.tex')
cs = []
for r in DV:
    if r and len(r) == 4 and not r[0].startswith('\\multicolumn'):
        cs.append(r)
cblock = cs[:9]
labs = [re.sub(r'\\cellcolor\{firstblue\}', '', r[0]).split('(')[0].strip() for r in cblock]
assert labs == ['0', '1', '2', '4', '8', '16', '32', '64', '$\\infty$'], labs
L = [r'\begin{tabular}{l*{9}{r}}%', r'\toprule', r'$c$ & ' + ' & '.join(['0', '1', '2', '4', '\\B{8}', '16', '32', '64', '$\\infty$']) + r'\\', r'\midrule']
for j, lab in ((1, '5つの平均'), (2, '縮小の履歴をそろえた対照'), (3, '上下反転')):
    vals = []
    for r in cblock:
        c = r[j]
        vals.append(('\\B{%s}' % c.replace('\\cellcolor{firstblue}', '')) if 'firstblue' in c else c)
    L.append(lab + ' & ' + ' & '.join(vals) + r'\\')
wr('t_c', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')

# ---------------------------------------------------------------- appendix: splits, models, development log, val2 analysis, six splits, cost
L = [r'\begin{tabular}{lll}%', r'\toprule', r'分割 & 画像 & 用途\\', r'\midrule',
     r'train & 分布内の学習画像5万枚（ImageNet-64は1万枚） & 比較手法の当てはめ\\',
     r'val1 & 分布内の画像2,000枚（ImageNet-64は1,500枚） & OLRの$\vec{\mu}$、$\mathbf{S}$、$\vec{\mu}_{o}$。比較手法の当てはめ\\',
     r'val2 & 分布内と各分布外集合から1,000枚ずつ（FashionMNISTは2,000枚） & OLRの開発\\',
     r'test & 500枚ずつ & 比較手法との比較\\',
     r'test2〜6 & 500枚ずつ（test6のCIFAR-10は258枚） & 分割を変えた確認\\',
     r'\bottomrule', r'\end{tabular}%']
wr('t_splits', '\n'.join(L) + '\n')
MO = [r for r in rows('supp_models_ol.tex') if r]
L = [r'\begin{tabular}{lllllrrr}%', r'\toprule', r'分布内 & 解像度 & モデル & ブロック & ブロックA & 画像$n$ & $\Delta$ & 数値\\', r'\midrule']
for r in MO:
    src = name(r[3]) if '\\cite' in r[3] else r[3]
    model = name(r[2]) + ('' if src == '本研究で学習' else src) + ('（本研究で学習）' if src == '本研究で学習' else '')
    L.append(' & '.join([name(r[0]), r[1], model, r[4], r[5], r[6], r[7], r[8]]) + r'\\')
wr('t_models', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
DL = [r for r in rows('supp_devlog_ol.tex') if r]
assert len(DL) == 8
L = [r'\begin{tabular}{cl*{4}{r}*{4}{r}}%', r'\toprule', r' & & \multicolumn{4}{c}{候補} & \multicolumn{4}{c}{変種（候補にしない）}\\', r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
     r'回 & 試した式 & 数 & 最良 & 対照 & 基準 & 数 & 最良 & 対照 & 基準\\', r'\midrule']
for r in DL:
    L.append(' & '.join([r[0], re.sub(r',\s*', '、', name(r[1]))] + [c.replace('--', '–') for c in r[2:]]) + r'\\')
wr('t_devlog', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
DW = [r for r in rows('tab_devw_ol.tex') if r]


def dcell(c):
    return ('\\B{%s}' % c.replace('\\cellcolor{firstblue}', '')) if 'firstblue' in c else c


L = [r'\begin{tabular}{lrrr@{\hspace{18pt}}lrrr@{\hspace{18pt}}lrrr}%', r'\toprule',
     r'\multicolumn{4}{c}{定数$c$（7つの向き）} & \multicolumn{4}{c}{使う向き（$c=8$）} & \multicolumn{4}{c}{分布内の画像の枚数$n$}\\',
     r'\cmidrule(lr){1-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}', r' & 平均 & 対照 & 上下反転 & & 平均 & 対照 & 上下反転 & & 平均 & 対照 & 上下反転\\', r'\midrule']
for r in DW:
    cells = [dcell(c) for c in r]
    cells = [c.replace('(\\textbf{\\Ours})', '（OLR）').replace('(', '（').replace(')', '）') for c in cells]
    L.append(' & '.join(cells) + r'\\')
wr('t_dev', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
L = [r'\begin{tabular}{l*{5}{r}rrrr}%', r'\toprule', r' & \multicolumn{5}{c}{分布内} & & & & \\', r'\cmidrule(lr){2-6}',
     r' & ' + IDS5 + r' & 5つの平均 & 上下反転 & 左右反転 & FPR95\\', r'\midrule']
for r in S:
    if r is None:
        L.append(r'\midrule')
        continue
    if len(r) == 1:
        lab = r[0].replace('\\multicolumn{10}{@{}l}{', '')
        lab = lab[:-1] if lab.endswith('}') else lab
        lab = name(lab).replace('（6分割の平均）', '（6分割の平均）')
        L.append(r'\multicolumn{10}{l}{%s}\\' % lab)
        continue
    lab = re.sub(r'\\cellcolor\{firstblue\}', '', r[0]).replace('\\quad', '\\quad ')
    vals = [c.replace('\\cellcolor{firstblue}', '').replace('--', '–') for c in r[1:]]
    if 'firstblue' in r[0]:
        lab = lab.replace('6分割の平均', '\\textbf{6分割の平均}')
        vals = ['\\textbf{%s}' % v for v in vals]
    L.append(lab + ' & ' + ' & '.join(vals) + r'\\')
wr('t_six', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
C = rows('tab_cost_ol.tex')
L = [r'\begin{tabular}{lcrrrrrr}%', r'\toprule', r'手法 & 止める & 秒/枚 & 比 & 画像 & 数値 & MB & AUROC\\', r'\midrule']
for r in C:
    if r is None:
        L.append(r'\midrule')
        continue
    L.append(' & '.join([name(r[0])] + [c.replace('\\cellcolor{firstblue}', '').replace('1/2', '$\\frac{1}{2}$') for c in r[1:]]) + r'\\')
wr('t_cost', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')

# ---------------------------------------------------------------- appendix: image quality, history, model, oracle, kappa, 256, pairs
G = rows('tab_degr_ol.tex')
L = [r'\begin{tabular}{l*{7}{r}r}%', r'\toprule', r' & \multicolumn{2}{c}{ぼかし} & \multicolumn{3}{c}{JPEG} & \multicolumn{2}{c}{縮小して戻す} & \\',
     r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}', r'手法 & 0.5 & 1.0 & 90 & 75 & 50 & $\frac{3}{4}$ & $\frac{1}{2}$ & 平均\\', r'\midrule']
for r in G:
    if r is None:
        L.append(r'\midrule')
        continue
    L.append(' & '.join([name(r[0]).replace('\\quad', '\\quad ')] + [c.replace('\\cellcolor{firstblue}', '') for c in r[1:]]) + r'\\')
wr('t_degr', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
Hh = rows('tab_hist_ol.tex')
L = [r'\begin{tabular}{ll*{5}{r}rrl}%', r'\toprule', r'画像 & 手法 & ' + IDS5 + r' & 5つの平均 & 上下反転 & OLRとの差\\', r'\midrule']
for r in Hh:
    if r is None:
        L.append(r'\midrule')
        continue
    vals = [c.replace('\\cellcolor{firstblue}', '').replace('--', '–') for c in r[2:]]
    L.append(' & '.join([r[0], name(r[1])] + vals) + r'\\')
wr('t_hist', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
V = [r for r in rows('tab_var_ol.tex') if r]
L = [r'\begin{tabular}{l*{5}{r}rrrr}%', r'\toprule', r'重み & ' + IDS5 + r' & 5つの平均 & 上下反転 & 内積の項 & 距離の項\\', r'\midrule']
for r in V:
    L.append(' & '.join([c.replace('\\cellcolor{firstblue}', '') for c in r]) + r'\\')
wr('t_var', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
VB = [r for r in rows('supp_var_ol.tex') if r]
L = [r'\begin{tabular}{llrrrrr}%', r'\toprule', r'分布内 & モデルの学習データ & 本来 & OLR & 内積の項 & 距離の項 & 上下反転\\', r'\midrule']
for r in VB:
    L.append(' & '.join(r) + r'\\')
wr('t_varby', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
O = rows('supp_oracle_ol.tex')
L = [r'\begin{tabular}{l*{5}{r}r}%', r'\toprule', r' & ' + IDS5 + r' & 5つの平均\\', r'\midrule']
for r in O:
    if r is None:
        L.append(r'\midrule')
        continue
    lab = name(r[0])
    L.append(' & '.join([lab] + r[1:]) + r'\\')
wr('t_oracle', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
K = [r for r in rows('supp_kappa_ol.tex') if r]
L = [r'\begin{tabular}{lr*{5}{r}rr}%', r'\toprule', r' & & \multicolumn{5}{c}{7つをまとめて} & 向きごと & val2の分布内\\', r'\cmidrule(lr){3-7}',
     r'分布内 & $p$ & トレース & 行列式 & 固有値の最小 & 中央 & 最大 & トレース & トレース\\', r'\midrule']
for r in K:
    L.append(' & '.join(r) + r'\\')
wr('t_kappa', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
HR = rows('tab_hr_ol.tex')
L = [r'\begin{tabular}{ll*{5}{r}rrrr}%', r'\toprule', r'画像 & 手法 & SSB-hard & NINCO & iNaturalist & Textures & OpenImage-O & 近い & 遠い & 5つの平均 & 上下反転\\', r'\midrule']
for r in HR:
    if r is None:
        L.append(r'\midrule')
        continue
    L.append(' & '.join([name(r[0]), name(r[1])] + [num(c) for c in r[2:]]) + r'\\')
wr('t_hr', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
HP = rows('supp_hr_ol.tex')
L = [r'\begin{tabular}{ll*{5}{r}rrrrr}%', r'\toprule', r'画像 & 式 & SSB-hard & NINCO & iNaturalist & Textures & OpenImage-O & 近い & 遠い & 5つの平均 & 上下反転 & FPR95\\', r'\midrule']
for r in HP:
    if r is None:
        L.append(r'\midrule')
        continue
    L.append(' & '.join([name(r[0]), name(r[1])] + r[2:]) + r'\\')
wr('t_hrparts', '\n'.join(L + [r'\bottomrule', r'\end{tabular}%']) + '\n')
PA = rows('supp_pairs_ol.tex')


def pcell(c):
    win = 'firstblue' in c
    loss = 'lossred' in c
    m = re.search(r'(\$[+-]\$)?(\d+\.\d)\}?$', c.replace('\\boldmath', ''))
    assert m, c
    s = {'$+$': '+', '$-$': '−', None: ''}[m.group(1)] + m.group(2)
    return ('\\B{%s}' % s) if win else (('\\R{\\textbf{%s}}' % s) if loss else s)


groups = []
for r in PA:
    if r is None:
        continue
    assert len(r) == 8, r
    if r[0]:
        groups.append([r[0], []])
    groups[-1][1].append(r)
assert [g[0] for g in groups] == ['CIFAR-10', 'SVHN', 'CelebA', 'ImageNet-64', 'FashionMNIST'] and sum(len(g[1]) for g in groups) == 51


def ptab(g):
    out = [r'\begin{tabular}[t]{lrrlrrr}', r'\toprule', r'\multicolumn{7}{l}{分布内：%s}\\' % g[0], r'\midrule',
           r'分布外 & OLR & \multicolumn{2}{c}{先行研究の最良} & 差 & CFS-kNN & 差\\', r'\midrule']
    for r in g[1]:
        out.append(' & '.join([r[1], r[2], r[3], r[4].replace('CFS-kNN', 'CFS-kNN').replace('IC-PNG', 'IC（PNG）').replace('IC-JPEG2000', 'IC（JPEG2000）'),
                               pcell(r[5]), r[6], pcell(r[7])]) + r'\\')
    return '\n'.join(out + [r'\bottomrule', r'\end{tabular}'])


wr('t_pairs1', ('\\hspace{14pt}%\n').join(ptab(g) + '%\n' for g in groups[:2]))
wr('t_pairs2', ('\\hspace{14pt}%\n').join(ptab(g) + '%\n' for g in groups[2:4]))
wr('t_pairs3', ptab(groups[4]) + '%\n')
print('wrote', len(WRITTEN), ' '.join(WRITTEN))
