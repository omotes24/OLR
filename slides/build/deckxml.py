# deckxml.py -- raw DrawingML / OMML builders for the OLR slides (1920 x 1080 pt canvas, Keynote and PowerPoint).
#   Text boxes that hold equations are written the way PowerPoint writes them: an mc:AlternateContent whose Choice (a14) carries the
#   OMML (Keynote imports it as native equation objects) and whose Fallback carries plain text.
PT = 12700
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main', 'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math', 'a14': 'http://schemas.microsoft.com/office/drawing/2010/main',
      'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
XMLNS = ' '.join('xmlns:%s="%s"' % kv for kv in NS.items())

# colours of the earlier decks (sampled from their PDF)
BLACK, NAVY, NAVYT, RED, HEAD, GRAYT = '000000', '1F3A7A', '192E62', 'C0392B', '404040', '585858'
BAR_OFF, LAB_OFF, LAB_ON = 'DCDFE5', 'C0C5CD', '284D8D'
FONT = 'Hiragino Sans W6'
MSZ = 3000            # equations: Keynote imports sz 3000 as 28.3 pt


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# ------------------------------------------------------------------ OMML
class Ctx:
    color = BLACK
    sz = MSZ


def arpr():
    return ('<a:rPr lang="en-US" sz="%d" b="0" i="0"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
            '<a:latin typeface="Cambria Math"/><a:ea typeface="Cambria Math"/><a:cs typeface="Cambria Math"/></a:rPr>') % (Ctx.sz, Ctx.color)


def R(t, sty=None, scr=None):
    pr = ('<m:scr m:val="%s"/>' % scr if scr else '') + ('<m:sty m:val="%s"/>' % sty if sty else '')
    return '<m:r>%s%s<m:t xml:space="preserve">%s</m:t></m:r>' % ('<m:rPr>%s</m:rPr>' % pr if pr else '', arpr(), esc(t))


def I(t): return R(t)                    # italic (variables)
def U(t): return R(t, 'p')               # upright (numbers, operators, names)
def Bd(t): return R(t, 'b')              # bold upright (matrices)
def SC(t): return R(t, 'p', 'script')    # script capital


def ctrl(): return '<m:ctrlPr>%s</m:ctrlPr>' % arpr()
def sub(b, s): return '<m:sSub><m:sSubPr>%s</m:sSubPr><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>' % (ctrl(), b, s)
def sup(b, s): return '<m:sSup><m:sSupPr>%s</m:sSupPr><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>' % (ctrl(), b, s)
def subsup(b, s, p): return '<m:sSubSup><m:sSubSupPr>%s</m:sSubSupPr><m:e>%s</m:e><m:sub>%s</m:sub><m:sup>%s</m:sup></m:sSubSup>' % (ctrl(), b, s, p)
def frac(n, d): return '<m:f><m:fPr>%s</m:fPr><m:num>%s</m:num><m:den>%s</m:den></m:f>' % (ctrl(), n, d)


def delim(x, b, e):
    return '<m:d><m:dPr><m:begChr m:val="%s"/><m:endChr m:val="%s"/>%s</m:dPr><m:e>%s</m:e></m:d>' % (b, e, ctrl(), x)


def par(x): return delim(x, '(', ')')
def brace(x): return delim(x, '{', '}')
def brack(x): return delim(x, '[', ']')
def absv(x): return delim(x, '|', '|')
def acc(ch, x): return '<m:acc><m:accPr><m:chr m:val="%s"/>%s</m:accPr><m:e>%s</m:e></m:acc>' % (ch, ctrl(), x)
def vec(x): return acc('⃗', x)
def bar(x): return '<m:bar><m:barPr><m:pos m:val="top"/>%s</m:barPr><m:e>%s</m:e></m:bar>' % (ctrl(), x)


def nary(ch, lo, hi, e):
    hide = '' if hi else '<m:supHide m:val="1"/>'
    return ('<m:nary><m:naryPr><m:chr m:val="%s"/><m:limLoc m:val="undOvr"/>%s%s</m:naryPr><m:sub>%s</m:sub><m:sup>%s</m:sup><m:e>%s</m:e></m:nary>'
            % (ch, hide, ctrl(), lo, hi or '', e))


def summ(lo, hi, e): return nary('∑', lo, hi, e)
def func(name, x): return '<m:func><m:funcPr>%s</m:funcPr><m:fName>%s</m:fName><m:e>%s</m:e></m:func>' % (ctrl(), name, x)
def log(x): return func(U('log'), x)


def maxu(var, x):       # max over var, the limit written under max
    return func('<m:limLow><m:limLowPr>%s</m:limLowPr><m:e>%s</m:e><m:lim>%s</m:lim></m:limLow>' % (ctrl(), U('max'), var), x)


def col(c, f):          # an expression in colour c
    old = Ctx.color
    Ctx.color = c
    try:
        return f()
    finally:
        Ctx.color = old


def size(sz, f):
    old = Ctx.sz
    Ctx.sz = sz
    try:
        return f()
    finally:
        Ctx.sz = old


# ------------------------------------------------------------------ text
def trpr(color, sz, u=False, font=None, bold=False):
    f = font or FONT
    return ('<a:rPr lang="ja-JP" altLang="en-US" sz="%d" b="%d" i="0"%s><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
            '<a:latin typeface="%s"/><a:ea typeface="%s"/><a:cs typeface="%s"/></a:rPr>') % (
        int(round(sz * 100)), 1 if bold else 0, ' u="sng"' if u else '', color, f, f, f)


class Run:
    def __init__(self, kind, xml, plain):
        self.kind, self.xml, self.plain = kind, xml, plain


def T(text, color=BLACK, sz=30, u=False, font=None, bold=False):
    return Run('r', '<a:r>%s<a:t xml:space="preserve">%s</a:t></a:r>' % (trpr(color, sz, u, font, bold), esc(text)), (text, color, sz, u, font))


def M(omml, plain='?', color=None, sz=None):
    """an inline equation; colour and size of its runs are taken from the current Ctx unless given"""
    return Run('m', '<a14:m><m:oMath>%s</m:oMath></a14:m>' % omml, (plain, color or BLACK, (sz or 30), False, None))


def BR(sz=30):
    return Run('br', '<a:br>%s</a:br>' % trpr(BLACK, sz), ('', BLACK, sz, False, None))


class Para:
    def __init__(self, runs, bullet=False, algn=None, sz=30, before=0, indent_chars=1.0, lnpct=None, after=0, marl=None, hang=None):
        self.runs, self.bullet, self.algn, self.sz, self.before, self.indent, self.lnpct, self.after = runs, bullet, algn, sz, before, indent_chars, lnpct, after
        self.marl, self.hang = marl, hang        # marl: left margin (pt) without a bullet; hang: hanging indent (pt) with a tab stop

    def ppr(self):
        a = ''
        tabs = ''
        if self.bullet:
            m = int(round(self.sz * self.indent * PT))
            a += ' marL="%d" indent="%d"' % (m, -m)
        elif self.hang:
            m = int(round(self.hang * PT))
            a += ' marL="%d" indent="%d"' % (m, -m)
            tabs = '<a:tabLst><a:tab pos="%d" algn="l"/></a:tabLst>' % m
        elif self.marl:
            a += ' marL="%d" indent="0"' % int(round(self.marl * PT))
        else:
            a += ' marL="0" indent="0"'
        if self.algn:
            a += ' algn="%s"' % self.algn
        inner = ''
        if self.lnpct:
            inner += '<a:lnSpc><a:spcPct val="%d"/></a:lnSpc>' % int(self.lnpct * 1000)
        inner += '<a:spcBef><a:spcPts val="%d"/></a:spcBef>' % int(self.before * 100)
        inner += '<a:spcAft><a:spcPts val="%d"/></a:spcAft>' % int(self.after * 100)
        if self.bullet:
            inner += '<a:buFont typeface="%s"/><a:buChar char="・"/>' % FONT
        else:
            inner += '<a:buNone/>'
        inner += tabs
        return '<a:pPr%s>%s</a:pPr>' % (a, inner)


def DISP(omml, plain='?', algn='ctr', sz=30, before=0):
    """a display equation (its own paragraph)"""
    r = Run('m', '<a14:m><m:oMathPara><m:oMathParaPr><m:jc m:val="%s"/></m:oMathParaPr><m:oMath>%s</m:oMath></m:oMathPara></a14:m>'
            % ({'ctr': 'centerGroup', 'l': 'left', 'r': 'right'}[algn], omml), (plain, BLACK, sz, False, None))
    return Para([r], algn=algn, sz=sz, before=before)


_id = [100]


def _nid():
    _id[0] += 1
    return _id[0]


def textbox(x, y, w, h, paras, name=None, anchor='t', wrap=True):
    has_math = any(r.kind == 'm' for p in paras for r in p.runs)
    nid = _nid()
    nm = name or 'tb%d' % nid

    def sp(choice):
        ps = ''
        for p in paras:
            body = ''
            for r in p.runs:
                if r.kind == 'm' and not choice:
                    t, c, s, u, f = r.plain
                    body += '<a:r>%s<a:t xml:space="preserve">%s</a:t></a:r>' % (trpr(c, s, u, f), esc(t))
                else:
                    body += r.xml
            ps += '<a:p>%s%s<a:endParaRPr lang="ja-JP" altLang="en-US" sz="%d"/></a:p>' % (p.ppr(), body, int(p.sz * 100))
        return ('<p:sp %s><p:nvSpPr><p:cNvPr id="%d" name="%s"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
                '<a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
                '<p:txBody><a:bodyPr wrap="%s" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="%s"><a:noAutofit/></a:bodyPr>'
                '<a:lstStyle/>%s</p:txBody></p:sp>') % (XMLNS, nid, esc(nm), int(x * PT), int(y * PT), int(w * PT), int(h * PT),
                                                       'square' if wrap else 'none', anchor, ps)
    if not has_math:
        return sp(True)
    return '<mc:AlternateContent %s><mc:Choice Requires="a14">%s</mc:Choice><mc:Fallback>%s</mc:Fallback></mc:AlternateContent>' % (XMLNS, sp(True), sp(False))


def line(x1, y1, x2, y2, color, width, name=None, cap='flat'):
    nid = _nid()
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    flip = ''
    if (x2 < x1) != (y2 < y1) and w and h:
        flip = ' flipV="1"'
    return ('<p:cxnSp %s><p:nvCxnSpPr><p:cNvPr id="%d" name="%s"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr>'
            '<a:xfrm%s><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            '<a:ln w="%d" cap="%s"><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:prstDash val="solid"/></a:ln></p:spPr></p:cxnSp>') % (
        XMLNS, nid, esc(name or 'ln%d' % nid), flip, int(x * PT), int(y * PT), int(w * PT), int(h * PT), int(width * PT), cap, color)
