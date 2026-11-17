# mk_deck.py OUT.pptx -- the OLR slides for the 26f WIP midterm talk (2026-11-17).
#   Format of the earlier decks (1114): 1920 x 1080 pt, Hiragino Sans W6, numbered headings 48 pt in 404040, page number "k/10" at the top
#   right, key message 30 pt red underlined, body 30 pt black with "・" bullets, red / navy emphasis, a progress bar of 8 segments
#   (背景 / 観測 / 提案手法 / 結果 / 考察とまとめ) on the content pages, tables typeset in TeX (Times) and pasted as 600 dpi images,
#   every mathematical character an equation object (OMML; Keynote imports it as native equations), punctuation 「、。」, no spaces
#   between Japanese and alphanumerics.  Main part: 10 pages (title, 8 content pages, References); then the supplement.
#   Every number is read from the resume (numbers_ol.tex) or from the table images that mk_tabs.py built from the resume's rows.
import io, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from pptx import Presentation
from pptx.util import Emu
from lxml import etree
from deckxml import *

OUT = sys.argv[1]
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper')   # the resume's source: numbers_ol.tex, main.bbl
B = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(B, '..', 'png')   # the table images (tex/t_*.tex compiled, 600 dpi)

# ------------------------------------------------------------------ numbers of the resume
def macros(fn):
    s = io.open(fn, encoding='utf-8').read()
    out = {}
    for m in re.finditer(r'\\newcommand\{\\ol([A-Za-z]+)\}\{', s):
        i = m.end(); depth = 1; j = i
        while depth:
            if s[j] == '{': depth += 1
            elif s[j] == '}': depth -= 1
            j += 1
        out[m.group(1)] = s[i:j - 1].replace('{,}', ',').replace('\\,', '')
    return out
N = macros(os.path.join(SRC, 'numbers_ol.tex'))
assert N['TestOL'] == '95.6' and N['GapCfsKnn'] == '7.6' and N['NStored'] == '552', (N['TestOL'], N['GapCfsKnn'])

# ------------------------------------------------------------------ references (numbered in order of first appearance in the slides)
REFS = ['song2021score', 'karras2022edm', 'nalisnick2019deep', 'serra2020input', 'nalisnick2019typ', 'mahmood2021msma', 'heng2024diffpath',
        'alexis2026gepc', 'shoushtari2026eigenscore', 'rouzoumka2026cfs', 'xiang2023ddae', 'ronneberger2015unet', 'lee2018mahalanobis',
        'hendrycks2019oe', 'dhariwal2021adm', 'zhang2023openood', 'ren2019likelihood', 'kamkari2024geometric', 'nichol2021improved',
        'karras2024edm2', 'tack2020csi']
NUM = {k: i + 1 for i, k in enumerate(REFS)}


def cite(*keys):
    return '[%s]' % ', '.join(str(NUM[k]) for k in keys)


def bib():
    s = io.open(os.path.join(SRC, 'main.bbl'), encoding='utf-8').read()
    out = {}
    for it in re.split(r'\\bibitem\{', s)[1:]:
        key, body = it.split('}', 1)
        t = ' '.join(body.split())
        for a, b in (("{\\`a}", 'à'), ("{\\'A}", 'Á'), ("{\\'o}", 'ó'), ("{\\'u}", 'ú'), ("{\\~n}", 'ñ'), ("{\\'e}", 'é'), ('{\\"o}', 'ö'), ('{\\"u}', 'ü')):
            t = t.replace(a, b)
        t = t.replace('{\\em ', '{').replace('~', ' ').replace('--', '–')
        t = t.replace('{', '').replace('}', '').replace('\\end thebibliography', '').replace('\\endthebibliography', '').strip()
        t = re.sub(r'\\end.*$', '', t).strip()
        out[key] = t
    return out
BIB = bib()
for k in REFS:
    assert k in BIB, k

# ------------------------------------------------------------------ equation pieces
mu = lambda: vec(I('μ'))
mus = lambda: sup(vec(I('μ')), U('*'))
muo = lambda: sub(vec(I('μ')), I('o'))
xv = lambda: vec(I('x'))
av = lambda: vec(I('a')) + par(xv())
S_ = lambda: Bd('S')
P_ = lambda: Bd('P')
TR = lambda e: sup(e, U('⊤'))


def eq_amp():
    inner = absv(sub(I('f'), I('h,j,u')) + par(xv()) + U('−') + sub(bar(I('f')), I('h,j')) + par(xv()))
    return (sub(I('a'), I('h')) + par(xv()) + U('=') + frac(U('1'), sub(I('C'), I('h')))
            + summ(I('j') + U('=') + U('1'), sub(I('C'), I('h')), U('log') + U('\u2009') + maxu(I('u'), inner)))


def eq_mustar(): return mus() + U('=') + mu() + U('+') + I('c') + par(mu() + U('−') + muo())
def eq_score(): return I('s') + par(xv()) + U('=') + U('−') + TR(brace(av() + U('−') + mus())) + P_() + brace(av() + U('−') + mus())
def eq_prec(): return P_() + U('=') + sup(par(S_() + U('+') + I('λ') + bar(I('s')) + Bd('I')), U('−1'))


def eq_expand():
    return (I('s') + par(xv()) + U('=') + col(NAVY, lambda: U('−') + sup(I('d'), U('2')) + par(xv())) + U('+')
            + col(RED, lambda: U('2') + I('c') + I('ℓ') + par(xv())) + U('−') + sup(I('c'), U('2')) + sup(U('Δ'), U('2')))


def eq_dsq(): return sup(I('d'), U('2')) + par(xv()) + U('=') + TR(brace(av() + U('−') + mu())) + P_() + brace(av() + U('−') + mu())
def eq_ell(): return I('ℓ') + par(xv()) + U('=') + TR(par(mu() + U('−') + muo())) + P_() + brace(av() + U('−') + mu())
def eq_delta(): return sup(U('Δ'), U('2')) + U('=') + TR(par(mu() + U('−') + muo())) + P_() + par(mu() + U('−') + muo())


def eq_llr():
    num_ = I('N') + par(vec(I('a')) + U(';') + mu() + U(',') + sup(P_(), U('−1')))
    den_ = I('N') + brace(vec(I('a')) + U(';') + muo() + U(',') + par(U('1') + U('+') + frac(U('1'), I('c'))) + sup(P_(), U('−1')))
    return U('log') + U('\u2009') + frac(num_, den_) + U('=') + frac(I('s') + par(xv()), U('2') + par(U('1') + U('+') + I('c'))) + U('+') + U('const.')


def eq_muo():
    return muo() + U('=') + frac(U('1'), U('7') + I('n')) + summ(I('k') + U('=') + U('1'), U('7'), summ(I('i') + U('=') + U('1'), I('n'), vec(I('a')) + par(sub(Bd('G'), I('k')) + sub(vec(I('x')), I('i')))))


def m(omml, plain, color=BLACK, sz=30):
    """an inline equation in colour / size"""
    old = (Ctx.color, Ctx.sz)
    Ctx.color, Ctx.sz = color, int(round(sz * 100))
    try:
        x = omml() if callable(omml) else omml
    finally:
        Ctx.color, Ctx.sz = old
    return M(x, plain, color, sz)


def mi(f, plain, color=BLACK, sz=30):
    return m(f, plain, color, sz)


# ------------------------------------------------------------------ deck
prs = Presentation()
prs.slide_width = Emu(1920 * PT)
prs.slide_height = Emu(1080 * PT)
for lay in list(prs.slide_layouts) + [prs.slide_master]:
    for ph in list(lay.placeholders):
        ph._element.getparent().remove(ph._element)
BLANK = prs.slide_layouts[6]
NMAIN = 10
SEG = [(96, 302), (320, 526), (530, 736), (754, 960), (964, 1169), (1187, 1393), (1397, 1603), (1621, 1827)]
GROUPS = [('背景', [0]), ('観測', [1, 2]), ('提案手法', [3, 4]), ('結果', [5, 6]), ('考察とまとめ', [7])]


class Slide:
    def __init__(self):
        self.sl = prs.slides.add_slide(BLANK)
        self.tree = self.sl.shapes._spTree

    def add(self, xml):
        self.tree.append(etree.fromstring(xml))

    def text(self, x, y, w, h, paras, name=None, anchor='t'):
        self.add(textbox(x, y, w, h, paras, name=name, anchor=anchor))

    def pic(self, path, x, y, w=None, h=None, name=None):
        im = Image.open(path)
        if w is None and h is None:
            raise ValueError
        if w is None:
            w = h * im.size[0] / im.size[1]
        if h is None:
            h = w * im.size[1] / im.size[0]
        p = self.sl.shapes.add_picture(path, Emu(int(x * PT)), Emu(int(y * PT)), Emu(int(w * PT)), Emu(int(h * PT)))
        if name:
            p.name = name
        return w, h

    def table(self, name, x, y, scale, center=None):
        """a table image at `scale` times its natural size (the TeX table's 10 pt text becomes 10*scale pt)"""
        path = os.path.join(TAB, name + '.png')
        im = Image.open(path)
        w = im.size[0] * 72 / 600 * scale
        h = im.size[1] * 72 / 600 * scale
        if center is not None:
            x = center - w / 2
        self.pic(path, x, y, w, h, name=name)
        return x, y, w, h

    def line(self, x1, y1, x2, y2, color, width, name=None):
        self.add(line(x1, y1, x2, y2, color, width, name=name))

    def notes(self, t):
        self.sl.notes_slide.notes_text_frame.text = t


def P(runs, **kw):
    return Para(runs, **kw)


def bul(runs, before=14, sz=30):
    return Para(runs, bullet=True, before=before, sz=sz)


def heading(s, title):
    runs = [T(title, HEAD, 48)] if isinstance(title, str) else [T(r, HEAD, 48) if isinstance(r, str) else r for r in title]
    s.text(96, 32, 1500, 66, [P(runs, sz=48)], name='heading')


def pageno(s, k, total=NMAIN):
    s.text(1524, 47, 300, 33, [P([T('%d/%d' % (k, total) if total else '%d' % k, HEAD, 24)], algn='r', sz=24)], name='page')


def keymsg(s, runs, y=108, color=RED, u=True):
    rr = []
    for r in runs:
        if isinstance(r, str):
            rr.append(T(r, color, 30, u=u))
        else:
            rr.append(r)
    s.text(96, y, 1728, 46, [P(rr)], name='key')


def progress(s, seg):
    for i, (x0, x1) in enumerate(SEG):
        if i < seg:
            s.line(x0, 1029.5, x1, 1029.5, NAVY, 5)
        elif i == seg:
            s.line(x0, 1029.5, x1, 1029.5, NAVY, 11)
        else:
            s.line(x0, 1029.5, x1, 1029.5, BAR_OFF, 5)
    for lab, segs in GROUPS:
        x0 = SEG[segs[0]][0]; x1 = SEG[segs[-1]][1]
        on = seg in segs
        s.text(x0, 1037, x1 - x0, 24, [P([T(lab, LAB_ON if on else LAB_OFF, 18)], algn='ctr', sz=18)], name='bar_' + lab)


def caption(s, x, y, w, h, runs, sz=24, algn=None):
    rr = [T(r, HEAD, sz) if isinstance(r, str) else r for r in runs]
    s.text(x, y, w, h, [P(rr, sz=sz, algn=algn)], name='caption')


def eqno(s, y, k, h=50):
    s.text(1704, y, 120, h, [P([T('(%d)' % k, BLACK, 30)], algn='r')], name='eqno%d' % k, anchor='ctr')


def disp(s, x, y, w, h, omml, plain, k=None, algn='ctr'):
    s.text(x, y, w, h, [DISP(omml, plain, algn=algn)], name='eq%s' % (k or ''), anchor='ctr')
    if k:
        eqno(s, y, k, h)


# ================================================================== p.1 title
s = Slide()
s.pic(os.path.join(B, 'logo.png'), 1727, 27, 188, 188, name='logo')
# key visual: the original and its seven re-orientations (the earlier title visual, with eight copies)
sw, sh = s.pic(os.path.join(B, 'strip_title.png'), 960 - 1318 / 2, 186, 1318, 156, name='visual')
x0 = 960 - 1318 / 2 + 8
s.text(x0 - 20, 354, 180, 34, [P([T('そのまま', RED, 26)], algn='ctr', sz=26)], name='vis_lab0')
s.text(x0 + 166, 354, 7 * 140 + 6 * 26, 34, [P([T('向きを変えた7つ（反転と回転）', '9AA1AC', 26)], algn='ctr', sz=26)], name='vis_lab1')
s.text(160, 428, 1600, 130, [P([T('OLR', NAVYT, 100, font='Helvetica Neue', bold=True)], algn='ctr', sz=100)], name='title')
s.text(160, 570, 1600, 64, [P([T('向きを変えた分布内画像による拡散モデルの分布外検知', NAVYT, 44)], algn='ctr', sz=44)], name='subtitle')
s.line(460, 672, 1460, 672, NAVYT, 4.5, name='rule')
s.text(160, 706, 1600, 44, [P([T('26f WIP 中間発表（2026/12/10 4限）', GRAYT, 30)], algn='ctr')], name='info1')
s.text(160, 758, 1600, 44, [P([T('d-hacks B3 omote / Mentor：zackeyさん', GRAYT, 30)], algn='ctr')], name='info2')
s.line(0, 1073, 260, 1073, RED, 14, name='band_red')
s.line(260, 1073, 1920, 1073, NAVY, 14, name='band_navy')

# ================================================================== p.2 background
s = Slide(); heading(s, '1.背景'); pageno(s, 2)
keymsg(s, ['尤度や統計量による手法は評価の回数が多く、中間特徴による手法は学習画像との比較が要る'])
s.text(96, 176, 1728, 300, [
    bul([T('拡散モデル%sは、学習済みのモデルを凍結したまま分布外検知に使える' % cite('song2021score', 'karras2022edm'))], before=0),
    bul([T('ただし尤度は、分布外の画像に分布内より高い値を与えることがある%s。尤度や統計量による手法[1, 4–9]は、1枚あたり10〜数百回のネットワークの評価を要し、分布内データセットによって崩れる' % cite('nalisnick2019deep', 'serra2020input'))]),
    bul([T('CFS%sは、低いノイズでの1回の順伝播から中間特徴%sを読み、これらを上回る。ただし、1,024次元の特徴を学習画像5万枚の統計量や近傍と比べる' % (cite('rouzoumka2026cfs'), cite('xiang2023ddae')))])], name='body')
caption(s, 96, 488, 1728, 34, ['自然画像の分布外集合でのAUROC（%、test、分布内ごとの平均。赤字は70%未満。順伝播は1枚あたりの回数で、+の後は逆伝播）'])
s.table('t_bg', 0, 534, 2.35, center=960)
s.text(60, 900, 1800, 46, [P([T('本研究：ブロックごとに数値を1つだけ読み、向きを変えた分布内画像から分布外の方向を求める手法「OLR」を提案', RED, 30)])], name='honken')
progress(s, 0)

# ================================================================== p.3 observation 1
s = Slide(); heading(s, '2.観測①：ブロックの振幅'); pageno(s, 3)
keymsg(s, ['学習済みのモデルでは、分布外の画像でブロックの振幅が下がる'])
s.text(96, 176, 1000, 230, [
    bul([T('画像にノイズを加えて、ノイズ除去器（U-Net%s）に通す' % cite('ronneberger2015unet'))], before=0),
    bul([T('ブロック'), mi(lambda: I('h'), 'h'), T('の出力の各チャネルで、空間平均から最も離れた値を読む')])], name='body1')
disp(s, 96, 300, 960, 120, eq_amp(), 'a_h(x)', k=None)
s.text(1000, 335, 70, 50, [P([T('(1)', BLACK, 30)], algn='r')], name='eqno1', anchor='ctr')
s.text(96, 450, 1000, 330, [
    bul([T('どこかに強い応答があれば大きい。ブロックごとに数値1つ')], before=0),
    bul([T('入力から、デコーダで'), mi(lambda: U('16') + U('×') + U('16'), '16x16'), T('になる最初のブロックAまでの'), BR(), mi(lambda: I('p'), 'p'),
         T('個を並べる（CIFAR-10で'), mi(lambda: I('p') + U('=') + U('23'), 'p=23'), T('）'), BR(), T('順伝播はブロックAで止めてよい')]),
], name='body2')
caption(s, 1130, 182, 694, 34, ['分布外の集合での振幅（val2、ブロックAまで）'])
tx, ty, tw, th = s.table('t_fall_ood', 1130, 226, 2.5)
s.text(1130, ty + th + 14, 694, 180, [P([mi(lambda: I('z'), 'z', HEAD, 24), T('：集合の平均と分布内の平均の差を、分布内の標準偏差で割った値。下がる、上がる：', HEAD, 24),
                                        mi(lambda: I('z') + U('<') + U('−0.2'), 'z<-0.2', HEAD, 24), T('、', HEAD, 24), mi(lambda: I('z') + U('>') + U('0.2'), 'z>0.2', HEAD, 24),
                                        T('のブロックの割合（%）。', HEAD, 24), mi(lambda: sub(I('z'), U('A')), 'zA', HEAD, 24), T('：ブロックAでの', HEAD, 24), mi(lambda: I('z'), 'z', HEAD, 24)], sz=24)], name='tabnote')
s.text(96, 900, 1728, 46, [P([T('ブロックAでは、38の分布外集合すべてで振幅の平均が分布内より低い（SVHNでは上がるブロックも多い）', NAVY, 30)])], name='bottom')
progress(s, 1)

# ================================================================== p.4 observation 2
s = Slide(); heading(s, '2.観測②：向きを変えた分布内画像'); pageno(s, 4)
keymsg(s, ['分布内の画像の向きを変えるだけでも、振幅は同じように下がる'])
s.pic(os.path.join(B, 'strip_p4.png'), 88, 168, 1746, 192, name='strip')
LAB = ['そのまま', '上下反転', '左右反転', '180度回転', '90度回転', '270度回転', '対角で反転', '対角で反転']
for i, lab in enumerate(LAB):
    cx = 96 + i * (176 + 45.7) + 88
    s.text(cx - 110, 368, 220, 30, [P([T(lab, RED if i == 0 else GRAYT, 22)], algn='ctr', sz=22)], name='lab%d' % i)
s.text(96, 430, 940, 420, [
    bul([T('正方形の画像の向きの変え方は、そのままを除いて7つ'), BR(), T('画素を並べ替えるだけで、画素の値の集まりは変わらない')], before=0),
    bul([T('見慣れた画像が見慣れなくなると、振幅が下がる'), BR(), T('分布外の画像を使わずに「下がる方向」が分かる')]),
    bul([T('分布外の集合で決めた方向とも近い'), BR(), T('（余弦は%s〜%s。補足資料J）' % (N['CosMin'], N['CosMax']))]),
], name='body')
caption(s, 1070, 436, 754, 34, ['向きを変えた画像での振幅（val2、ブロックAまで）'])
tx, ty, tw, th = s.table('t_fall_or', 1070, 480, 2.45)
caption(s, 1070, ty + th + 12, 754, 40, ['値の定義はp.3と同じ。val2の分布内は参考'])
s.text(96, 900, 1728, 46, [P([T('向きを変えた分布内画像が、分布外の画像の代わりに「見慣れない側」を教えてくれる', NAVY, 30)])], name='bottom')
progress(s, 2)

# ================================================================== p.5 method idea
s = Slide(); heading(s, '3.提案手法（発想）'); pageno(s, 5)
keymsg(s, ['着眼：分布内の平均を、向きを変えた画像の反対側へずらしてから距離を測る'])
s.pic(os.path.join(B, 'fig_c.png'), 96, 186, 760, None, name='fig_c')
s.text(920, 186, 904, 150, [
    bul([T('分布内の画像の並び（振幅のベクトル）の平均'), mi(mu, 'mu'), T('と共分散'), mi(S_, 'S'), T('、'), BR(), T('向きを変えた画像の平均'), mi(muo, 'mu_o'), T('を求める')], before=0)], name='b1')
disp(s, 920, 300, 760, 56, eq_mustar(), 'mu*', k=None)
eqno(s, 300, 2, 56)
s.text(920, 372, 904, 100, [
    bul([mi(mus, 'mu*'), T('（'), mi(lambda: I('c') + U('=') + U('8'), 'c=8'), T('）までのマハラノビス距離%sをスコアにする' % cite('lee2018mahalanobis'))], before=0)], name='b2')
disp(s, 920, 476, 760, 60, eq_score(), 's(x)', k=None)
eqno(s, 476, 3, 60)
s.text(920, 548, 904, 50, [P([T('ただし'), mi(eq_prec, 'P=(S+lsI)^-1'), T('。式は1つだけ')], marl=30)], name='b3')
s.text(96, 640, 1728, 46, [P([T('展開すると、2つの項に分かれる', BLACK, 30)])], name='b4')
disp(s, 96, 694, 1580, 60, eq_expand(), 'expand', k=None)
eqno(s, 694, 4, 60)
s.text(96, 772, 1728, 220, [
    bul([mi(lambda: U('−') + sup(I('d'), U('2')), '-d2', NAVY), T('：距離の項。分布内の平均からの外れ（どの方向でも小さくなる）', NAVY)], before=0),
    bul([mi(lambda: I('ℓ'), 'l', RED), T('：内積の項。', RED), mi(lambda: mu() + U('−') + muo(), 'mu-muo', RED), T('との内積（スカラー）で、向きを変えた画像の側へ外れると小さくなる', RED)], before=8),
    bul([T('向きを変えても並びが動かないデータでは'), mi(lambda: U('Δ'), 'Delta'), T('が小さく、内積の項は自動的に軽くなる')], before=8)], name='b5')
progress(s, 3)

# ================================================================== p.6 the method
s = Slide(); heading(s, '3.提案手法「OLR」'); pageno(s, 6)
keymsg(s, ['OLR：テスト時はブロックAまで1回通し、2次形式を1つ計算するだけ'])
s.pic(os.path.join(B, 'fig_a.png'), 96, 172, 1728, None, name='fig_a')
s.pic(os.path.join(B, 'fig_b.png'), 96, 560, 960, None, name='fig_b')
s.text(1100, 566, 724, 440, [
    bul([T('事前の準備（1回だけ）：'), BR(), T('分布内の画像を8つの向きで通し、'), BR(), mi(mu, 'mu'), T('、'), mi(S_, 'S'), T('、'), mi(muo, 'mu_o'), T('を求める')], before=0),
    bul([T('テスト時：ブロックAで止める'), BR(), T('順伝播1回（全体の%s%%）と、' % N['FwdAShare']), BR(), mi(lambda: I('p'), 'p'), T('次元の2次形式1つ')]),
    bul([T('勾配による学習、追加のネットワーク、'), BR(), T('人手のラベル、分布外の画像は使わない')]),
], name='body')
progress(s, 4)

# ================================================================== p.7 results 1
s = Slide(); heading(s, '4.結果①'); pageno(s, 7)
keymsg(s, ['OLRは平均%s%%で、先行研究の最良（CFSのkNNヘッド、%s%%）を%sポイント上回る' % (N['TestOL'], N['BestPriorMean'], N['GapCfsKnn'])])
tx, ty, tw, th = s.table('t_main', 0, 170, 2.6, center=960)
caption(s, 96, ty + th + 10, 1728, 70, ['AUROC（%）、test。左の5列は自然画像の分布外集合の平均。†：先行研究の最良。紺の太字：OLRが最良に並ぶか上回る列。勝/分/負：全51組でのOLRの結果。§：途中で止める。秒/枚：CIFAR-10のモデル、CPU（2スレッド。‡：同じ計算機で続けて測定）'])
s.text(96, 846, 1728, 160, [
    bul([T('差の95%%信頼区間は[%s, %s]。6つのテスト分割では%s〜%s%%（平均%s%%）' % (N['GapCfsKnnLo'], N['GapCfsKnnHi'], N['SixMin'], N['SixMax'], N['SixMean']))], before=0),
    bul([T('及ばないのはSVHN（%s%%。DiffPathは%s%%）。負けた%s組はすべてSVHNで、相手は10回以上の評価を要する手法' % (N['TestSVHN'], N['PriorBestSVHN'], N['NLostPrior']))], before=8)], name='body')
progress(s, 5)

# ================================================================== p.8 results 2
s = Slide(); heading(s, '4.結果②'); pageno(s, 8)
keymsg(s, ['効いているのは内積の項。縮小の履歴をそろえても、CFSとの差は変わらない'])
caption(s, 96, 176, 1200, 34, ['式の一部を変えたもの（AUROC、%。6つのテスト分割の平均）'])
tx, ty, tw, th = s.table('t_terms', 96, 218, 2.3)
s.text(tx + tw + 50, 226, 1824 - (tx + tw + 50), 260, [
    bul([T('内積の項だけで%s%%、距離の項だけでは%s%%' % (N['SixL'], N['SixT']))], before=0),
    bul([T('距離の項はSVHNとImageNet-64で効く')], before=8)], name='b1')
caption(s, 96, 500, 1200, 34, ['定数', mi(lambda: I('c'), 'c', HEAD, 24), 'を変えたとき（AUROC、%。開発用の分割val2）'])
tx, ty, tw, th = s.table('t_c', 96, 542, 2.3)
s.text(tx + tw + 50, 550, 1824 - (tx + tw + 50), 200, [
    bul([T('8と16の間は平ら。'), mi(lambda: I('c'), 'c'), T('は2つの項の重み')], before=0)], name='b2')
s.text(96, 776, 1728, 240, [
    bul([T('縮小の履歴をそろえた対照（すべての画像を半分に縮小して戻す）：OLRは%s%%、CFSのkNNヘッドは%s%%。差は%sポイントで変わらない' % (N['EqOL'], N['EqCfsKnn'], N['EqGapCfsKnn']))], before=0),
    bul([T('ノイズ除去器の重みを初期化したままにすると%s%%（内積の項は%s%%→%s%%）。学習した応答を読んでいる' % (N['VarRinit'], N['VarTrainedL'], N['VarRinitL']))], before=8)], name='b3')
progress(s, 6)

# ================================================================== p.9 discussion and summary
s = Slide(); heading(s, '5.考察とまとめ'); pageno(s, 9)
s.text(96, 108, 600, 46, [P([T('まとめ', NAVY, 30)])], name='sub1')
disp(s, 96, 156, 1728, 60, eq_score(), 's(x)')
s.text(96, 236, 1728, 330, [
    bul([T('凍結した拡散モデルのブロックごとに数値を1つ読み、向きを変えた分布内画像から求めた方向で判定するOLRを提案')], before=0),
    bul([T('5つの分布内データセットで平均AUROC%s%%（6分割で%s〜%s%%）。先行研究の最良を%sポイント上回る' % (N['TestOL'], N['SixMin'], N['SixMax'], N['GapCfsKnn']), RED, u=True)], before=10),
    bul([T('順伝播は途中で止める1回、保存は%s個の数値。勾配による学習、追加のネットワーク、人手のラベル、分布外の画像なし' % N['NStored'])], before=10)], name='b1')
s.text(96, 478, 900, 46, [P([T('考察と残る課題', RED, 30)])], name='sub2')
s.text(96, 530, 1728, 380, [
    bul([T('補助の分布外データ%sの代わりに、分布内の画像の向きを変えた写しを使い、学習の代わりに平均を1つ求める' % cite('hendrycks2019oe'))], before=0),
    bul([mi(lambda: U('256') + U('×') + U('256'), '256x256'), T('のImageNetでは、向きを変えても並びがほとんど動かず（'), mi(lambda: U('Δ') + U('=') + U(N['HrDelta']), 'Delta=0.55'),
         T('）、内積の項が働かない（上下反転%s%%）。' % N['HrOLVf']), mi(lambda: U('Δ'), 'Delta'), T('は分布内の画像だけで求まるので、使えるかは事前に確かめられる')], before=10),
    bul([T('画質と縮小の履歴に反応する（CFSも同じ）。'), mi(lambda: I('c') + U('=') + U('8'), 'c=8'), T('は開発用の分割で決め、テスト分割は以前に他の手法で結果を見ている')], before=10)], name='b2')
s.text(1600, 950, 224, 46, [P([T('終わり', BLACK, 30, u=True)], algn='r')], name='end')
progress(s, 7)

# ================================================================== p.10 references
s = Slide(); s.text(96, 32, 1500, 66, [P([T('References', HEAD, 48)], sz=48)], name='heading'); pageno(s, 10)
half = (len(REFS) + 1) // 2
for col_i, keys in enumerate((REFS[:half], REFS[half:])):
    paras = []
    for k in keys:
        paras.append(Para([T('[%d]\t%s' % (NUM[k], BIB[k]), BLACK, 16)], sz=16, before=6, hang=40))
    s.text(96 + col_i * 880, 130, 840, 900, paras, name='refs%d' % col_i)

# ================================================================== supplement
APP = []


def app_page(title, sub=None):
    s = Slide()
    k = 11 + len(APP)
    APP.append(title)
    heading(s, title)
    pageno(s, k, total=None)
    if sub:
        s.text(96, 108, 1728, 46, [P([T(sub, BLACK, 30)])], name='sub')
    return s


# divider
s = Slide(); APP.append('補足資料')
s.pic(os.path.join(B, 'logo.png'), 1727, 27, 188, 188, name='logo')
s.text(132, 500, 1200, 100, [P([T('補足資料', HEAD, 72)], sz=72)], name='title')

# A. setting
s = app_page('A.実験の設定', '4.実験の補足：画像の分け方とモデル')
s.table('t_splits', 96, 170, 2.3)
s.table('t_models', 96, 440, 2.2)
s.text(96, 735, 1728, 300, [
    bul([T('ブロック：プロファイルに入るブロックの数', BLACK, 26), mi(lambda: I('p'), 'p', BLACK, 26), T('、ブロックAまでの数、全ブロックの数。', BLACK, 26), mi(lambda: U('Δ'), 'Delta', BLACK, 26),
         T('：', BLACK, 26), mi(lambda: sup(U('Δ'), U('2')), 'Delta2', BLACK, 26), T('（式(7)）の平方根。数値：保存する数値の数', BLACK, 26)], before=0, sz=26),
    bul([T('分布外：', BLACK, 26), mi(lambda: U('32') + U('×') + U('32'), '32x32', BLACK, 26), T('の分布内には他の自然画像のデータセット（CIFAR-10、CIFAR-100、SVHN、CelebA、Textures、Tiny-ImageNet、LSUNとImageNet%s、ImageNet32、Places365）。ImageNet-64にはOpenOOD v1.5%sの5集合とPlaces365、CelebA。FashionMNISTには灰色画像の4集合' % (cite('tack2020csi'), cite('zhang2023openood')), BLACK, 26)], before=8, sz=26)], name='body')

# B. development
s = app_page('B.開発の経過（val2）', '3.提案手法の補足：式の決め方')
s.table('t_devlog', 0, 170, 2.3, center=960)
s.text(96, 580, 1728, 420, [
    bul([T('各回の前に、候補の式、候補にしない変種、判断の基準を手順書に書いてから計算した。候補は%s通り、変種は%s通り' % (N['NDev'], N['NVar']))], before=0),
    bul([T('基準：val2の平均が%s%%以上、かつ縮小の履歴をそろえた対照で%s%%以上。8回目の候補（7つの向き、' % (N.get('BarNat', '95.5'), N.get('BarEq', '91.5'))), mi(lambda: I('c') + U('=') + U('8'), 'c=8'), T('）が初めて満たした')]),
    bul([T('6〜8回目には基準を上回る変種があり、次の回の候補はそれを見て決めた（'), mi(lambda: I('c') + U('=') + U('8'), 'c=8'), T('は6回目、7つの向きは7回目の後）')]),
    bul([T('実装を凍結してから、6つのテスト分割で1回だけ評価した')])], name='body')

# C. val2 analysis
s = app_page('C.val2での分析', '3.提案手法の補足：定数、向き、枚数')
s.table('t_dev', 0, 170, 2.45, center=960)
s.text(96, 640, 1728, 360, [
    bul([mi(lambda: I('c'), 'c'), T('を0から大きくすると上がり、8と16で最も高い。'), mi(lambda: I('c') + U('→') + U('∞'), 'c->inf'), T('（内積の項だけ）で下がる')], before=0),
    bul([T('上下反転だけでもほぼ同じ。差が出るのはSVHNで、90度の回転と対角の反転で上がる')]),
    bul([T('分布内の画像を1,000枚に減らしても変わらない')]),
    bul([T('平均：自然画像の分布外集合の5つの平均。対照：縮小の履歴をそろえた対照。紺の太字はOLR', HEAD, 24)], sz=24)], name='body')

# D. six splits
s = app_page('D.6つのテスト分割', '4.結果の補足：分割を変えた確認と、式の一部を変えたもの')
s.table('t_six', 0, 170, 2.6, center=960)
caption(s, 96, 780, 1728, 80, ['AUROC（%）。上下反転、左右反転：分布内の画像とその反転を見分けるAUROC。FPR95：%、低いほどよい。下の3行は評価の前に登録した変種'])

# E. cost
s = app_page('E.時間と用意するもの', '4.結果の補足：CIFAR-10のモデル、CPU（2スレッド）')
s.table('t_cost', 0, 180, 2.9, center=960)
s.text(96, 470, 1728, 400, [
    bul([T('止める：順伝播を止める位置（'), mi(lambda: frac(U('1'), U('2')), '1/2'), T('：デコーダで解像度が半分になるところ）')], before=0),
    bul([T('比：ブロックAまでの順伝播（%s秒）との比' % N['FwdA'])]),
    bul([T('画像：事前に通す分布内の画像。数値、MB：保存する数値の数と16ビットでの大きさ')]),
    bul([T('CFSをOLRと同じval1の画像に当てはめても、testでの値は%s%%（対角ガウス）と%s%%（kNNヘッド）で、差は画像の枚数によらない' % (N['CfsValMean'], N['CfsKnnValMean']))])], name='body')

# F. fall
s = app_page('F.振幅は下がるか（詳細）', '2.観測の補足：val2、ブロックAまで')
caption(s, 96, 170, 1728, 34, [mi(lambda: I('z') + U('<') + U('−0.2'), 'z', HEAD, 24), '、', mi(lambda: I('z') + U('>') + U('0.2'), 'z', HEAD, 24), 'のブロックの割合（%）と、ブロックAでの', mi(lambda: I('z'), 'z', HEAD, 24)])
s.table('t_fall_full', 96, 212, 2.6)
caption(s, 96, 560, 1728, 34, ['符号だけで数えたもの（分布内の平均より低いブロックの割合、%）。val2の分布内でも0%や94%になるので、本編では0.2標準偏差の幅をとった'])
s.table('t_fallsign', 96, 640, 2.4)

# G. image quality
s = app_page('G.画質への感度', '4.結果の補足：分布内の画像と、劣化させた写しを見分けるAUROC（test6、5つの平均）')
s.table('t_degr', 96, 180, 2.8)
s.text(1100, 190, 724, 600, [
    bul([T('内容が同じでも、OLRは劣化させた写しを平均%s%%で見分ける' % N['DqOLMean'])], before=0),
    bul([T('CFSのkNNヘッドも%s%%で、画質への感度は中間特徴を読む検知器に共通' % N['DqCfsKnnMean'])]),
    bul([T('距離の項が内積の項より敏感。ImageNet-64は品質90のJPEGを%s%%で見分ける' % N['DqINJpeg'])]),
    bul([T('50%なら見分けない', HEAD, 24)], sz=24)], name='body')

# H. history
s = app_page('H.縮小の履歴をそろえた対照', '4.結果の補足：test、すべての画像の一辺を半分に縮小して戻す')
s.table('t_hist', 0, 180, 2.55, center=960)
s.text(96, 540, 1728, 400, [
    bul([T('分布外集合は、データセットごとに違う元画像から違う方法で縮小されている。較正とCFSの参照画像も含めてすべてを縮小して戻した')], before=0),
    bul([T('OLRは%sポイント、CFSのkNNヘッドも%sポイント下がり、差は%sポイント（[%s, %s]）で変わらない' % (N['EqDropOL'], N['EqDropCfsKnn'], N['EqGapCfsKnn'], N['EqGapCfsKnnLo'], N['EqGapCfsKnnHi']))]),
    bul([T('低下には、履歴の違いによる分と、細部が失われた分が混ざっている')])], name='body')

# I. model
s = app_page('I.モデルの学習', '4.結果の補足：ノイズ除去器の重みだけを変える（test6、ノイズの種0）')
s.table('t_var', 0, 176, 2.55, center=960)
s.table('t_varby', 96, 440, 2.4)
s.text(1080, 450, 744, 560, [
    bul([T('初期化したまま、並べ替え：学習していない重み。内積の項は%s%%→%s%%' % (N['VarTrainedL'], N['VarRinitL']))], before=0),
    bul([T('距離の項の低下は小さい（%s%%→%s%%）。画像の統計の違いは、学習していないネットワークの振幅にも現れる' % (N['VarTrainedT'], N['VarRinitT']))]),
    bul([T('左下の表：別のデータで学習したモデル。SVHNの画像をCIFAR-10のモデルに通しても%s%%で、SVHNの結果はモデルが数字を学習した証拠にはならない' % N['VarByMax'])])], name='body')

# J. oracle
s = app_page('J.分布外集合で決めた方向との比較', '2.観測の補足：val2、ブロックAまで（偶数番目と奇数番目に分けて交差）')
s.table('t_oracle', 0, 180, 2.6, center=960)
s.text(96, 520, 1728, 480, [
    bul([T('下の2行の方向は、val2の分布外の画像を使って線形判別で決めたもの（手法ではない。並びが持つ情報の目安）')], before=0),
    bul([T('OLRの方向'), mi(lambda: P_() + par(mu() + U('−') + muo()), 'P(mu-muo)'), T('との余弦は%s〜%s。分布外の画像なしで近い方向が得られている' % (N['CosMin'], N['CosMax']))]),
    bul([T('集合ごとに決めた方向は画質や履歴の違いも拾う（ImageNet-64に対するSSB-hardを%s%%で分ける。OLRは%s%%）' % (N['XSsbSet'], N['XSsbOLR']))])], name='body')

# K. likelihood ratio
s = app_page('K.尤度比としての解釈', '3.提案手法の補足：式(4)の2つの項と、正規分布の尤度比')
disp(s, 96, 168, 1580, 60, eq_dsq(), 'd2'); eqno(s, 168, 5, 60)
disp(s, 96, 250, 1580, 60, eq_ell(), 'l'); eqno(s, 250, 6, 60)
disp(s, 96, 332, 1580, 60, eq_delta(), 'Delta2'); eqno(s, 332, 7, 60)
s.text(96, 414, 1728, 80, [P([T('分布内を'), mi(lambda: I('N') + par(mu() + U(',') + sup(P_(), U('−1'))), 'N(mu,P-1)'), T('、向きを変えた画像を'),
                              mi(lambda: I('N') + brace(muo() + U(',') + par(U('1') + U('+') + frac(U('1'), I('c'))) + sup(P_(), U('−1'))), 'N(muo,...)'), T('とおくと')])], name='b0', anchor='ctr')
disp(s, 96, 506, 1580, 120, eq_llr(), 'llr'); eqno(s, 536, 8, 60)
caption(s, 96, 650, 1728, 60, ['向きを変えた並びの広がり（val2の較正画像。トレース：', mi(lambda: frac(U('tr') + par(Bd('P') + Bd('C')), I('p')), 'tr(PC)/p', HEAD, 24), '。', mi(lambda: Bd('C'), 'C', HEAD, 24), 'は向きを変えた並びの共分散）'])
s.table('t_kappa', 96, 722, 2.05)
s.text(1150, 726, 674, 300, [
    bul([T('モデルどおりなら'), mi(lambda: U('1') + U('+') + frac(U('1'), I('c')) + U('=') + U('1.125'), '1.125'), T('倍。測ると%s〜%s倍で合わない' % (N['KapMin'], N['KapMax']))], before=0),
    bul([mi(lambda: I('c'), 'c'), T('は当てはめた量ではなく、2つの項の重みである')])], name='body')
s.text(1240, 196, 440, 170, [P([T('式(5)〜(7)はどれも', BLACK, 24), mi(lambda: U('1') + U('×') + I('p'), '1xp', BLACK, 24), T('、', BLACK, 24), mi(lambda: I('p') + U('×') + I('p'), 'pxp', BLACK, 24), T('、', BLACK, 24),
                                    mi(lambda: I('p') + U('×') + U('1'), 'px1', BLACK, 24), T('の行列の積で、', BLACK, 24), mi(lambda: U('1') + U('×') + U('1'), '1x1', BLACK, 24), T('の行列（スカラー）', BLACK, 24)], sz=24)], name='scalar_note')

# L. 256
s = app_page(['L.', mi(lambda: U('256') + U('×') + U('256'), '256x256', HEAD, 48), 'の画像'], '4.結果の補足：ImageNet、ADM%s、OpenOOD v1.5%s（各集合1,000枚）' % (cite('dhariwal2021adm'), cite('zhang2023openood')))
s.table('t_hr', 0, 170, 2.45, center=960)
s.table('t_hrparts', 0, 500, 2.35, center=960)
s.text(96, 870, 1728, 150, [
    bul([T('向きを変えても並びがほとんど動かず（'), mi(lambda: U('Δ') + U('=') + U(N['HrDelta']), 'Delta'), T('）、スコアはほぼ距離の項だけ。分布外集合は元のファイルの量だけでAUROC%s%%以上で見分けられ、内容を見分けている証拠にならない' % N['FactsMin'])], before=0, sz=26)], name='body')

# M. pairs
s = app_page('M.組ごとの値（test）①', '4.結果の補足：組ごとのAUROC（%）。紺の太字はOLRの勝ち、赤は負け。CFS-kNN：CFSのkNNヘッド')
s.table('t_pairs1', 0, 176, 2.55, center=960)
s = app_page('M.組ごとの値（test）②', '4.結果の補足：組ごとのAUROC（%）。紺の太字はOLRの勝ち、赤は負け。CFS-kNN：CFSのkNNヘッド')
tx, ty, tw, th = s.table('t_pairs2', 96, 170, 2.0)
s.table('t_pairs3', 96, ty + th + 24, 2.0)

prs.save(OUT)
print('saved', OUT, len(prs.slides), 'slides')
