# ol_gen.py -- numbers and table bodies of the paper on the orientation likelihood-ratio score (working name OL): per U-Net block up to A
#   one number (the amplitude); score = -(a - m*)^T P (a - m*), m* = m + c (m - m_o), c = 8.  Standard library only.
#   inputs (gen/data/ol/, copied from the GPU server; PROTOCOL_IDO.md I64-I70):
#     ol_final.json            code/ol_final.py: the single evaluation (I68-I69); server names test7 = this paper's test6, ..., test2 = test
#     olx_sets.json            code/olx_eval.py sets: per-set AUROC / FPR95 of the stored scores of that evaluation
#     ol_vs_all_test2.json     code/ol_vs_all.py: pair by pair against the published rows on test
#     olx_cfs.json             CFS on test with the original images and under the identical history; paired bootstrap
#     olx_dq.json, olx_var.json, olx_hr.json   image quality, model variants, 256x256
#     olx_dev.json             development analyses on val2;  gx_val2*_s0.json  the rounds of the development;  olx_info.json  fitted quantities
#     bench_ol.jsonl           CPU seconds per image, one session (eu/experiments/bench_ol.py)
#     hl_meta.json, fk_ablate.json, ssb_diag.json, ssb_diag2.json, bench_rf256.jsonl   blocks of the models, calibration sizes, source files at 256
#   writes ../numbers_ol.tex, ../tab_*_ol.tex and ../supp_*_ol.tex
import os, re, json
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data', 'ol')
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; IDN = ['CIFAR-10', 'SVHN', 'CelebA', 'ImageNet-64', 'FashionMNIST']
NAME = dict(zip(TAGS, IDN)); KEY = {'edmc10': 'Cten', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'in64': 'IN', 'fmnist': 'FM'}
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
NAT = {'edmc10': ['svhn_test', 'textures', 'celeba_test', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val', 'cifar100_test'],
       'l6svhn': ['cifar10_test', 'cifar100_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'],
       'dpceleba': ['cifar10_test', 'cifar100_test', 'svhn_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'],
       'in64': ['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test'],
       'fmnist': ['g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist']}
SETN = {'svhn_test': 'SVHN', 'textures': 'Textures', 'celeba_test': 'CelebA', 'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN', 'imagenet_fix': 'ImageNet',
        'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365', 'cifar100_test': 'CIFAR-100', 'cifar10_test': 'CIFAR-10',
        'in64_ssb_hard': 'SSB-hard', 'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O',
        'in64_places365': 'Places365', 'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot',
        'g32_notmnist': 'notMNIST', 'synth_noise': '一様ノイズ', 'synth_const': '定数画像', 'g32_synth_noise': '一様ノイズ', 'g32_synth_const': '定数画像'}
PNAME = {'test7': 'test6', 'test6': 'test5', 'test5': 'test4', 'test4': 'test3', 'test3': 'test2', 'test2': 'test'}   # server name -> name in the paper
SPO = ['test2', 'test3', 'test4', 'test5', 'test6', 'test7']                                                       # the paper's order: test, test2, ..., test6
mean = lambda v: (lambda w: sum(w) / len(w))(list(v))
def f1(v): return f'{v:.1f}'
def f2(v): return f'{v:.2f}'
def f3(v): return f'{v:.3f}'
def sg(v, nd=1): return ('$+$' if v >= 0 else '$-$') + f'{abs(v):.{nd}f}'
def sgn(v, nd=1): return ('0.' + '0' * nd) if abs(v) < 0.5 * 10 ** -nd else sg(v, nd)
def tt(h): return '\\texttt{' + h.replace('model.', '').replace('unet.', '').replace('_', '\\_') + '}'
def cm(v): return f'{v:,}'.replace(',', '{,}')
def ci(b, nd=1): return f'{sg(b["diff"], nd)} [{sg(b["ci"][0], nd)}, {sg(b["ci"][1], nd)}]'
def blue(v): return f'\\cellcolor{{firstblue}}{v}'
def LJ(f): return json.load(open(f'{D}/{f}'))
M = {}
def W(name, L):
    while L[-1].startswith('\\midrule'): L.pop()
    if not L[-1].endswith('\\bottomrule'): L[-1] += '\\bottomrule'
    open(f'{OUT}/{name}', 'w').write('\n'.join(L) + '\n')
# ---- the single evaluation (ol_final.json) and its per-set values (olx_sets.json)
F = LJ('ol_final.json'); SP, SM, CLF = F['splits'], F['summary'], F['claims']; SE = LJ('olx_sets.json')
def s(sp, k='OL', met='nat', cond='original', t=None): r = SP[sp][cond]['rows'][k]; return r['mean'][met] if t is None else r[t][met]
def six(k='OL', met='nat', cond='original', t=None): return mean(s(sp, k, met, cond, t) for sp in SPO)
assert CLF['H4'] is True and six() >= 95.0      # H4: the statement registered about OLR alone (protocols/PROTOCOL_OLR.md, I68)
for sp in SPO:
    for cond in ('original', 'eq50'):
        for t in TAGS: assert abs(SE[sp][cond][t]['nat'] - s(sp, cond=cond, t=t)) < 1e-6 and abs(SE[sp][cond][t]['vflip'] - s(sp, met='vflip', cond=cond, t=t)) < 1e-6, (sp, cond, t)
M['TestOL'] = f1(s('test2')); M['TestVf'] = f1(s('test2', met='vflip')); M['SixMin'] = f1(min(s(sp) for sp in SPO)); M['SixMax'] = f1(max(s(sp) for sp in SPO)); M['SixMean'] = f1(six()); M['SixMeanB'] = f2(six())
M['SixVf'] = f1(six(met='vflip')); M['SixVfMin'] = f1(min(s(sp, met='vflip') for sp in SPO)); M['SixVfMax'] = f1(max(s(sp, met='vflip') for sp in SPO))
M['SixHf'] = f1(mean(SE[sp]['original']['mean']['hflip'] for sp in SPO)); M['SixFpr'] = f1(mean(SE[sp]['original']['mean']['fpr95'] for sp in SPO))
M['SixAll'] = f1(six('OLa')); M['SixL'] = f1(six('OLl')); M['SixT'] = f1(six('OLt')); M['SixAllGap'] = f2(six() - six('OLa')); M['SixLGap'] = f1(six() - six('OLl')); M['SixTGap'] = f1(six() - six('OLt'))
for t in TAGS:
    M['Six' + KEY[t]] = f1(six(t=t)); M['SixAll' + KEY[t]] = f1(six('OLa', t=t)); M['SixL' + KEY[t]] = f1(six('OLl', t=t)); M['SixT' + KEY[t]] = f1(six('OLt', t=t)); M['SixVf' + KEY[t]] = f1(six(met='vflip', t=t))
    M['SixHf' + KEY[t]] = f1(mean(SE[sp]['original'][t]['hflip'] for sp in SPO)); M['SixMin' + KEY[t]] = f1(min(s(sp, t=t) for sp in SPO)); M['SixMax' + KEY[t]] = f1(max(s(sp, t=t) for sp in SPO))
assert six('OLa', t='l6svhn') > six(t='l6svhn') + 2 and six('OLa', t='edmc10') < six(t='edmc10') and six('OLa', t='in64') < six(t='in64')      # the sentences on "all blocks" say this
assert six() > six('OLa') > six('OLl') > six('OLt')
M['EqSix'] = f1(six(cond='eq50')); M['EqSixMin'] = f1(min(s(sp, cond='eq50') for sp in SPO)); M['EqSixMax'] = f1(max(s(sp, cond='eq50') for sp in SPO)); M['EqSixDrop'] = f1(six() - six(cond='eq50'))
M['EqSixVf'] = f1(six(met='vflip', cond='eq50'))
for t in TAGS: M['EqSix' + KEY[t]] = f1(six(cond='eq50', t=t)); M['EqSixDrop' + KEY[t]] = f1(six(t=t) - six(cond='eq50', t=t))
M['NCten'] = str(SE['test7']['original']['edmc10']['n_id']); M['NTex'] = str(SE['test7']['original']['edmc10']['sets']['textures']['n']); assert M['NCten'] == '258'
L = ['% written by gen/ol_gen.py: OLR on the six splits (ol_final.json, olx_sets.json); the registered ablations as six-split means', '\\multicolumn{10}{@{}l}{\\textbf{\\Ours(Ours)}}\\\\']
for sp in SPO:
    e = SE[sp]['original']; L.append(' & '.join([f'\\quad {PNAME[sp]}'] + [f1(s(sp, t=t)) for t in TAGS] + [f1(s(sp)), f1(s(sp, met='vflip')), f1(e['mean']['hflip']), f1(e['mean']['fpr95'])]) + '\\\\')
L.append(' & '.join(blue(v) for v in ['\\quad 6分割の平均'] + [f1(six(t=t)) for t in TAGS] + [f1(six()), f1(six(met='vflip')), M['SixHf'], M['SixFpr']]) + '\\\\'); L.append('\\midrule')
L.append('\\multicolumn{10}{@{}l}{式の一部を変えたもの(6分割の平均)}\\\\')
for k, lab in (('OLa', '全ブロック'), ('OLl', '方向の項だけ'), ('OLt', '距離の項だけ')):
    L.append(' & '.join(['\\quad ' + lab] + [f1(six(k, t=t)) for t in TAGS] + [f1(six(k)), f1(six(k, met='vflip')), '--', '--']) + '\\\\')
W('tab_six_ol.tex', L)
# ---- fitted quantities, blocks and storage
INFO = LJ('olx_info.json'); MET = LJ('hl_meta.json'); AB = LJ('fk_ablate.json'); NREF = {t: AB[t]['n_val1_id'] for t in TAGS}
def hooks(t): return [h[0] for h in MET[t]['hooks']]
def blkA(t): return [h[0] for h in MET[t]['hooks'] if h[1] == 'dec' and h[3] == 16][0]
PB = {t: INFO[t]['p'] for t in TAGS}
for t in TAGS:
    assert INFO[t]['A'] == blkA(t) and INFO[t]['n_cal'] == NREF[t] and INFO[t]['c'] == 8.0 and INFO[t]['blocks_up_to_A'] == hooks(t).index(blkA(t)) + 1 and SE['test2']['original'][t]['blocks'] == PB[t], t
    assert PB[t] == sum(1 for h in MET[t]['hooks'][:INFO[t]['blocks_up_to_A']] if h[3] > 1), t                           # the blocks left out are those with a 1x1 output
def nstored(t): return PB[t] + PB[t] ** 2
M['NBlkCten'] = str(PB['edmc10']); M['NRefMin'] = cm(min(NREF.values())); M['NRefMax'] = cm(max(NREF.values())); M['NStored'] = cm(nstored('edmc10')); M['MBOurs'] = f'{2 * nstored("edmc10") / 1e6:.4f}'
M['DeltaMin'] = f1(min(INFO[t]['Delta'] for t in TAGS)); M['DeltaMax'] = f1(max(INFO[t]['Delta'] for t in TAGS)); M['NBlkMin'] = str(min(PB.values())); M['NBlkMax'] = str(max(PB.values()))
assert min(TAGS, key=lambda t: INFO[t]['Delta']) == 'in64' and max(TAGS, key=lambda t: INFO[t]['Delta']) == 'dpceleba'
for t in TAGS: M['Delta' + KEY[t]] = f1(INFO[t]['Delta'])
MODEL = {'edmc10': ('$32{\\times}32$', 'EDM(VP, DDPM++)', '\\cite{karras2022edm}'), 'l6svhn': ('$32{\\times}32$', 'VP-SDE', '\\cite{kamkari2024geometric}'),
         'dpceleba': ('$32{\\times}32$', 'iDDPM', '\\cite{nichol2021improved}'), 'in64': ('$64{\\times}64$', 'EDM2-XS(条件なし)', '\\cite{karras2024edm2}'),
         'fmnist': ('$32{\\times}32$', 'EDMの前処理によるU-Net', '本研究で学習')}
L = ['% written by gen/ol_gen.py: models, blocks, calibration images, stored numbers (olx_info.json, hl_meta.json)']
for t in TAGS:
    L.append(' & '.join([NAME[t], MODEL[t][0], MODEL[t][1], MODEL[t][2], f"{PB[t]} / {INFO[t]['blocks_up_to_A']} / {len(hooks(t))}", f'{tt(blkA(t))}', cm(NREF[t]), f2(INFO[t]['Delta']), cm(nstored(t))]) + '\\\\')
if 'in256c' in INFO:
    h = INFO['in256c']; assert h['A'] == 'output_blocks.2'
    L.append(' & '.join(['ImageNet(追加実験)', '$256{\\times}256$', 'ADM(条件なし)', '\\cite{dhariwal2021adm}', f"{h['p']} / {h['blocks_up_to_A']} / {h['blocks_all']}", tt(h['A']), cm(h['n_cal']), f2(h['Delta']), cm(h['p'] + h['p'] ** 2)]) + '\\\\')
    M['HrNCal'] = cm(h['n_cal']); M['HrNBlk'] = str(h['p'])
else: M['HrNCal'] = '??'; M['HrNBlk'] = '??'
W('supp_models_ol.tex', L)
# ---- computation (bench_ol.jsonl: one session of 7 methods)
BE = {r['method']: r for r in [json.loads(l) for l in open(f'{D}/bench_ol.jsonl') if l.strip()]}
assert len(BE) == 7 and all(r['threads'] == 2 and r['batch'] == 4 and r['reps'] == 15 for r in BE.values()) and len({r['when'][:10] for r in BE.values()}) == 1
sec = {k: BE[k]['sec_per_img'] for k in BE}; so, fa, fA = sec['ol'], sec['fwd_all'], sec['fwd_A']
M['SecOL'] = f3(so); M['FwdAll'] = f3(fa); M['FwdA'] = f3(fA); M['FwdAShare'] = f'{100 * fA / fa:.0f}'; M['JudgeMs'] = f1(1000 * (so - fA)); M['SecCfs'] = f3(sec['cfs']); M['SecCfsStop'] = f3(sec['cfs_stop'])
M['SecCfsKnn'] = f3(sec['cfsknn']); M['SecCfsKnnStop'] = f3(sec['cfsknn_stop']); M['SecRatioCfsKnn'] = f2(so / sec['cfsknn_stop']); M['SecRatioCfs'] = f2(so / sec['cfs_stop']); M['SecRatioCfsKnnFull'] = f2(so / sec['cfsknn'])
assert so < sec['cfs_stop'] < sec['cfsknn_stop'] and 0 < so - fA < 0.005
cpu = re.sub(r'\s+', ' ', BE['ol']['cpu'].split(':', 1)[1]).strip(); m_ = re.match(r'Intel\(R\) Core\(TM\) (\S+) CPU @ ([\d.]+)GHz', cpu); assert m_, cpu
M['CpuHere'] = f'Intel Core {m_.group(1)}, {m_.group(2)}\\,GHz'
# ---- pair by pair on test (server test2) against the published rows
J2 = LJ('ol_vs_all_test2.json'); assert J2['check_max_abs_diff_vs_ol_final'] < 1e-6 and J2['ours'] == 'OL'
EIGHT = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore']; PRIOR = EIGHT + ['CFS', 'CFS-kNN']
LABEL = {'NLL': 'NLL(尤度)', 'IC-png': 'IC(PNG)', 'IC-j2k': 'IC(JPEG2000)', 'TT': 'TT', 'MSMA*': 'MSMA型', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore', 'CFS': 'CFS', 'CFS-kNN': 'CFS(kNN ヘッド)'}
SHORT = {'NLL': 'NLL', 'IC-png': 'IC-PNG', 'IC-j2k': 'IC-J2K', 'TT': 'TT', 'MSMA*': 'MSMA型', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore', 'CFS': 'CFS', 'CFS-kNN': 'CFS-kNN'}
TXT = {'CFS-kNN': 'CFSの kNN ヘッド', 'MSMA*': 'MSMA型', 'NLL': 'NLL'}
PASSES = {'NLL': '128+128', 'IC-png': '128+128', 'IC-j2k': '128+128', 'TT': '128+128', 'MSMA*': '20', 'DiffPath': '10', 'GEPC': '14', 'EigenScore': '180', 'CFS': '1', 'CFS-kNN': '1'}
SEC = {'NLL': '46', 'IC-png': '46', 'IC-j2k': '46', 'TT': '46', 'MSMA*': '3.5', 'DiffPath': '1.5', 'GEPC': '2.3', 'EigenScore': '27'}       # the timing session of the published methods (numbers_bench.tex)
CITE = {'NLL': 'song2021score', 'IC-png': 'serra2020input', 'IC-j2k': 'serra2020input', 'TT': 'nalisnick2019typ', 'MSMA*': 'mahmood2021msma', 'DiffPath': 'heng2024diffpath',
        'GEPC': 'alexis2026gepc', 'EigenScore': 'shoushtari2026eigenscore', 'CFS': 'rouzoumka2026cfs', 'CFS-kNN': 'rouzoumka2026cfs'}
def jm(r, t): return J2['ids'][t]['ours'] if r is None else J2['ids'][t]['rows'][r]['mean']
def jvf(r, t): p = J2['pairs'][t][IDD[t] + '__vflip']; return p['ours'] if r is None else p['rows'][r]['auroc']
def jrow(r): v = [jm(r, t) for t in TAGS]; return v + [mean(v), mean(jvf(r, t) for t in TAGS)]
def wtl(r): v = J2['rows'][r]; return f"{v['wins']}/{v['ties']}/{v['losses']}"
def pname(t, o): return f'{NAME[t]}の上下反転' if o.endswith('__vflip') else f'{NAME[t]}を分布内とする{SETN.get(o, o)}'
npairs = sum(len(P) for P in J2['pairs'].values()); assert npairs == 51
for t in TAGS: assert abs(jm(None, t) - s('test2', t=t)) < 1e-6 and abs(jm('all blocks', t) - s('test2', 'OLa', t=t)) < 1e-6
M['Pairs'] = str(npairs); M['NPrior'] = str(len(PRIOR)); M['NStat'] = str(len(EIGHT))
best = max(PRIOR, key=lambda r: mean(jm(r, t) for t in TAGS)); bd = J2['mean_diff'][best]; assert best == 'CFS-kNN' and bd['ci'][0] > 0
M['BestPriorName'] = TXT.get(best, LABEL[best]); M['BestPriorMean'] = f1(mean(jm(best, t) for t in TAGS)); M['GapCfsKnn'] = f1(bd['diff']); M['GapCfsKnnLo'] = f1(bd['ci'][0]); M['GapCfsKnnHi'] = f1(bd['ci'][1])
dc = J2['mean_diff']['CFS']; M['GapCfs'] = f1(dc['diff']); M['GapCfsLo'] = f1(dc['ci'][0]); M['GapCfsHi'] = f1(dc['ci'][1]); M['CfsMean'] = f1(mean(jm('CFS', t) for t in TAGS))
assert abs(float(M['TestOL']) - float(M['BestPriorMean']) - float(M['GapCfsKnn'])) < 0.11
bv = max(PRIOR, key=lambda r: mean(jvf(r, t) for t in TAGS)); M['VfBestPriorName'] = TXT.get(bv, LABEL[bv]); M['VfBestPriorAvg'] = f1(mean(jvf(bv, t) for t in TAGS))
for t in TAGS:
    b = max(PRIOR, key=lambda r: jm(r, t)); v = J2['ids'][t]['rows'][b]; M['PriorBest' + KEY[t]] = f1(jm(b, t)); M['PriorBest' + KEY[t] + 'Name'] = TXT.get(b, LABEL[b]); M['Test' + KEY[t]] = f1(jm(None, t)); M['PriorBestGap' + KEY[t]] = sg(v['diff'])
    k = J2['ids'][t]['rows']['CFS-kNN']; M['GapCfsKnn' + KEY[t]] = sg(k['diff']); M['VerdictCfsKnn' + KEY[t]] = k['verdict']
M['PriorWinMin'] = str(min(J2['rows'][r]['wins'] for r in PRIOR)); M['PriorWinMax'] = str(max(J2['rows'][r]['wins'] for r in PRIOR)); M['WtlCfsKnn'] = wtl('CFS-kNN'); M['WtlCfs'] = wtl('CFS')
lmx = max(J2['rows'][r]['losses'] for r in PRIOR); M['PriorLossMax'] = str(lmx); M['PriorLossMaxName'] = ', '.join(TXT.get(r, LABEL[r]) for r in PRIOR if J2['rows'][r]['losses'] == lmx)
lostp = {(t, o): [r for r in PRIOR if p['rows'][r]['verdict'] == 'loss'] for t, P in J2['pairs'].items() for o, p in P.items()}; lostp = {k: v for k, v in lostp.items() if v}
M['NLostPrior'] = str(len(lostp)); M['NLostPriorSVHN'] = str(sum(t == 'l6svhn' for t, o in lostp)); M['NLostPriorVf'] = str(sum(o.endswith('__vflip') for t, o in lostp))
M['LostPriorList'] = ', '.join(f"{pname(t, o)}({'・'.join(SHORT[r] for r in v)})" for (t, o), v in lostp.items()) or 'なし'
assert all(J2['rows'][r]['wins'] > J2['rows'][r]['losses'] for r in PRIOR) and J2['rows']['CFS']['losses'] == 0 and J2['rows']['CFS-kNN']['losses'] == 0
SEC.update({'CFS': f2(sec['cfs_stop']), 'CFS-kNN': f2(sec['cfsknn_stop'])}); HEREM = ('CFS', 'CFS-kNN'); STOPPED = ('CFS', 'CFS-kNN')
NPASS = lambda r: sum(int(x) for x in PASSES[r].split('+')); BPASS = min(NPASS(r) for r in PRIOR); BSEC = min(float(SEC[r]) for r in PRIOR)
cols = {r: jrow(r) for r in PRIOR}; bestc = [max(round(cols[r][j], 1) for r in PRIOR) for j in range(7)]
L = ['% written by gen/ol_gen.py: AUROC (%) on test (server test2); wins / ties / losses of Ours (ol_vs_all_test2.json)']
for r in PRIOR:
    cells = [f1(v) + ('$^{\\dagger}$' if round(v, 1) == bestc[j] else '') for j, v in enumerate(cols[r])]
    sup = ('\\dagger' if float(SEC[r]) == BSEC else '') + ('\\ddagger' if r in HEREM else ''); last = SEC[r] + (f'$^{{{sup}}}$' if sup else '')
    sup = ('\\S' if r in STOPPED else '') + ('\\dagger' if NPASS(r) == BPASS else ''); pas = PASSES[r] + (f'$^{{{sup}}}$' if sup else '')
    L.append(f"{LABEL[r]}~\\cite{{{CITE[r]}}} & " + ' & '.join(cells) + f" & {wtl(r)} & {pas} & {last}\\\\")
L.append('\\midrule'); under = False
def oc(j, v):
    global under
    second = sorted({round(cols[r][j], 1) for r in PRIOR}, reverse=True)
    if round(v, 1) >= bestc[j]: return f'\\cellcolor{{firstblue}}\\textbf{{{f1(v)}}}'
    if len(second) > 1 and round(v, 1) >= second[1]: under = True; return f'\\underline{{{f1(v)}}}'
    return f1(v)
ocells = [oc(j, v) for j, v in enumerate(jrow(None))]; M['OursColsBest'] = str(sum('firstblue' in c for c in ocells))
L.append('\\textbf{\\Ours(Ours)} & ' + ' & '.join(ocells) + f" & -- & 1$^{{\\S}}$ & {f2(so)}$^{{\\ddagger}}$\\\\\\bottomrule")
W('tab_main_ol.tex', L); M['UnderlineNote'] = ' \\underline{下線}：その最良に次ぐ列.' if under else ''
M['TestSVHNBehind'] = f1(float(M['PriorBestSVHN']) - float(M['TestSVHN'])); M['TestFMBehind'] = f1(float(M['PriorBestFM']) - float(M['TestFM']))
# ---- time and what is needed (CIFAR-10 network; AUROC on test)
NB = {t: 10000 if t == 'in64' else 50000 for t in TAGS}
def crow(lab, stop, tot, img, num, mb, au, bl=False):
    c = [lab, stop, f3(tot), f2(tot / fA), img, num, mb, f1(au)]; return ' & '.join(blue(v) if bl else v for v in c) + '\\\\'
L = ['% written by gen/ol_gen.py: CPU seconds per image (bench_ol.py, one session), ID images needed, stored numbers and size (16 bit), five-ID natural mean on test']
L.append(crow('\\textbf{\\Ours(Ours)}', 'A', so, cm(NREF['edmc10']), cm(nstored('edmc10')), f'{2 * nstored("edmc10") / 1e6:.4f}', s('test2'), True)); L.append('\\midrule')
L.append(crow('CFS(kNN ヘッド)', '1/2', sec['cfsknn_stop'], cm(NB['edmc10']), cm(NB['edmc10'] * 1024), f'{2 * NB["edmc10"] * 1024 / 1e6:.0f}', mean(jm('CFS-kNN', t) for t in TAGS)))
L.append(crow('CFS(対角ガウス)', '1/2', sec['cfs_stop'], cm(NB['edmc10']), cm(2 * 1024), f'{2 * 2 * 1024 / 1e6:.4f}', mean(jm('CFS', t) for t in TAGS)))
W('tab_cost_ol.tex', L); M['NImgCfs'] = cm(NB['edmc10']); M['MBCfsKnn'] = f'{2 * NB["edmc10"] * 1024 / 1e6:.0f}'; M['ImgRatio'] = f'{NB["edmc10"] / NREF["edmc10"]:.0f}'
# ---- test, every pair against the best published row; the new split, every set
def dcell(x):
    d = sgn(x['diff']); return {'win': f'\\cellcolor{{firstblue}}\\textbf{{\\boldmath{d}}}', 'loss': f'\\cellcolor{{lossred}}{d}', 'tie': d}[x['verdict']]
def order(t): P = J2['pairs'][t]; o = NAT[t] + [o for o in P if o.endswith('__vflip')] + [o for o in P if 'synth' in o]; assert set(o) == set(P), (t, sorted(P)); return o
L = ['% written by gen/ol_gen.py: test (server test2), AUROC (%) per pair, mean of noise seeds 0-2 (ol_vs_all_test2.json)']
for t in TAGS:
    for i, o in enumerate(order(t)):
        p = J2['pairs'][t][o]; R_ = p['rows']; bp = max(PRIOR, key=lambda m: R_[m]['auroc']); w = R_[bp]
        L.append(f"{NAME[t] if i == 0 else ''} & {SETN.get(o) or '上下反転'} & {f1(p['ours'])} & {f1(w['auroc'])} & {SHORT[bp]} & {dcell(w)} & {f1(R_['CFS-kNN']['auroc'])} & {dcell(R_['CFS-kNN'])}\\\\")
    L.append('\\midrule')
W('supp_pairs_ol.tex', L)
L = ['% written by gen/ol_gen.py: test6 (server test7), OL per set: AUROC and FPR95 with the original images, AUROC under the identical history (olx_sets.json)']
for t in TAGS:
    e, q = SE['test7']['original'][t]['sets'], SE['test7']['eq50'][t]['sets']; names = NAT[t] + [IDD[t] + '__vflip', IDD[t] + '__hflip']; assert set(names) <= set(e), (t, sorted(e))
    for i, o in enumerate(names):
        L.append(' & '.join([NAME[t] if i == 0 else '', SETN.get(o) or ('上下反転' if o.endswith('__vflip') else '左右反転'), str(e[o]['n']), f1(e[o]['auroc']), f1(e[o]['fpr95']), f1(q[o]['auroc'])]) + '\\\\')
    L.append('\\midrule')
W('supp_perset_ol.tex', L)
L = ['% written by gen/ol_gen.py: OL under the identical history on the six splits (ol_final.json)']
for sp in SPO: L.append(' & '.join([PNAME[sp]] + [f1(s(sp, cond='eq50', t=t)) for t in TAGS] + [f1(s(sp, cond='eq50')), f1(s(sp, met='vflip', cond='eq50')), f1(SE[sp]['eq50']['mean']['fpr95'])]) + '\\\\')
L.append('\\midrule'); L.append(' & '.join(['6分割の平均'] + [f1(six(cond='eq50', t=t)) for t in TAGS] + [f1(six(cond='eq50')), f1(six(met='vflip', cond='eq50')), f1(mean(SE[sp]['eq50']['mean']['fpr95'] for sp in SPO))]) + '\\\\')
L.append(' & '.join(['元の画像との差'] + [sgn(six(cond='eq50', t=t) - six(t=t)) for t in TAGS] + [sg(six(cond='eq50') - six()), sg(six(met='vflip', cond='eq50') - six(met='vflip')), '--']) + '\\\\')
W('supp_splitseq_ol.tex', L)
# ---- CFS on the same images of test: the original images and the identical history (olx_cfs.json)
CF = LJ('olx_cfs.json'); CO, CE = CF['original'], CF['eq50']
for t in TAGS:
    assert abs(CO['rows']['OL'][t]['nat'] - s('test2', t=t)) < 1e-6 and abs(CE['rows']['OL'][t]['nat'] - s('test2', cond='eq50', t=t)) < 1e-6
    for r in ('CFS', 'CFS-kNN'): assert abs(CO['rows'][r][t]['nat'] - jm(r, t)) < 1e-6, (r, t)
def cfm(c, r, met='nat'): return c['rows'][r]['mean'][met]
M['EqOL'] = f1(cfm(CE, 'OL')); M['EqCfs'] = f1(cfm(CE, 'CFS')); M['EqCfsKnn'] = f1(cfm(CE, 'CFS-kNN')); M['OrigCfs'] = f1(cfm(CO, 'CFS')); M['OrigCfsKnn'] = f1(cfm(CO, 'CFS-kNN'))
M['EqDropOL'] = f1(cfm(CO, 'OL') - cfm(CE, 'OL')); M['EqDropCfs'] = f1(cfm(CO, 'CFS') - cfm(CE, 'CFS')); M['EqDropCfsKnn'] = f1(cfm(CO, 'CFS-kNN') - cfm(CE, 'CFS-kNN'))
b = CE['boot']['CFS-kNN']['mean']; M['EqGapCfsKnn'] = f1(b['diff']); M['EqGapCfsKnnLo'] = f1(b['ci'][0]); M['EqGapCfsKnnHi'] = f1(b['ci'][1]); assert b['ci'][0] > 0
b = CE['boot']['CFS']['mean']; M['EqGapCfs'] = f1(b['diff']); M['EqGapCfsLo'] = f1(b['ci'][0]); M['EqGapCfsHi'] = f1(b['ci'][1])
b = CO['boot']['CFS-kNN']['mean']; M['OrigGapCfsKnnB'] = f1(b['diff']); M['OrigGapCfsKnnBLo'] = f1(b['ci'][0]); M['OrigGapCfsKnnBHi'] = f1(b['ci'][1])
M['EqWtlCfsKnn'] = '/'.join(str(CE['wtl']['CFS-kNN'][k]) for k in ('wins', 'ties', 'losses')); M['EqWtlCfs'] = '/'.join(str(CE['wtl']['CFS'][k]) for k in ('wins', 'ties', 'losses')); M['EqNPairs'] = str(CE['n_pairs'])
M['OrigWtlCfsKnnB'] = '/'.join(str(CO['wtl']['CFS-kNN'][k]) for k in ('wins', 'ties', 'losses'))
eqlost = [k for k, v in CE['pairs']['CFS-kNN'].items() if v['verdict'] == 'loss']; M['EqNLostCfsKnn'] = str(len(eqlost)); M['EqLostCfsKnnList'] = ', '.join(pname(k.split('|')[0], k.split('|')[1][len('test2-'):]) for k in eqlost) or 'なし'
for t in TAGS: v = CE['boot']['CFS-kNN'][t]; M['EqGapCfsKnn' + KEY[t]] = sg(v['diff']); M['EqOL' + KEY[t]] = f1(CE['rows']['OL'][t]['nat']); M['EqCfsKnn' + KEY[t]] = f1(CE['rows']['CFS-kNN'][t]['nat'])
M['TestFpr'] = f1(cfm(CO, 'OL', 'fpr95')); M['CfsKnnFpr'] = f1(cfm(CO, 'CFS-kNN', 'fpr95')); M['CfsFpr'] = f1(cfm(CO, 'CFS', 'fpr95')); assert abs(cfm(CO, 'OL', 'fpr95') - SE['test2']['original']['mean']['fpr95']) < 1e-6
M['EqVfOL'] = f1(cfm(CE, 'OL', 'vflip')); M['EqVfCfsKnn'] = f1(cfm(CE, 'CFS-kNN', 'vflip'))
L = ['% written by gen/ol_gen.py: test (server test2), the original images and the identical resampling history: OL, CFS, CFS-kNN (olx_cfs.json)']
for c, cl in ((CO, '元の画像'), (CE, '履歴をそろえた対照')):
    for i, (r, lab) in enumerate((('OL', '\\textbf{\\Ours(Ours)}'), ('CFS-kNN', 'CFS(kNN ヘッド)'), ('CFS', 'CFS(対角ガウス)'))):
        v = [cl if i == 0 else '', lab] + [f1(c['rows'][r][t]['nat']) for t in TAGS] + [f1(cfm(c, r)), f1(cfm(c, r, 'vflip')), ('--' if r == 'OL' else ci(c['boot'][r]['mean']))]
        L.append(' & '.join(blue(x) if (r == 'OL' and j > 0) else x for j, x in enumerate(v)) + '\\\\')
    L.append('\\midrule')
W('tab_hist_ol.tex', L)
# ---- CFS fitted on the same val1 ID images as OL (cfs_n.json, code/cfs_fit_n.py; PROTOCOL_IDO.md I48, I52): test, noise seeds 0-2
EN = LJ('cfs_n.json'); ER = EN['rows']; assert EN['n_fit'] == {t: NREF[t] for t in TAGS} and max(EN['check'].values()) < 1e-6
assert abs(ER['CFS-kNN (50k)']['mean']['nat'] - mean(jm('CFS-kNN', t) for t in TAGS)) < 0.006 and abs(ER['CFS (50k)']['mean']['nat'] - mean(jm('CFS', t) for t in TAGS)) < 0.006
M['CfsValMean'] = f1(ER['CFS (val1)']['mean']['nat']); M['CfsKnnValMean'] = f1(ER['CFS-kNN (val1)']['mean']['nat']); M['GapCfsKnnVal'] = f1(s('test2') - ER['CFS-kNN (val1)']['mean']['nat'])
assert abs(ER['CFS-kNN (val1)']['mean']['nat'] - ER['CFS-kNN (50k)']['mean']['nat']) < 0.5 and ER['CFS (val1)']['mean']['nat'] > ER['CFS (50k)']['mean']['nat']
# ---- image quality: the ID images of test6 against their degraded copies (olx_dq.json)
DQ = LJ('olx_dq.json'); COND = DQ['cond']; CN = {'blur05': 'ぼかし0.5', 'blur10': 'ぼかし1.0', 'jpeg90': 'JPEG 90', 'jpeg75': 'JPEG 75', 'jpeg50': 'JPEG 50', 'down75': '3/4に縮小', 'down50': '1/2に縮小'}
assert DQ['check_max_abs_diff_of_the_ID_scores_to_the_evaluation'] < 2.0 and list(CN) == COND   # two runs of the same images differ by the GPU arithmetic: at most 0.6% of the SD of the scores, Spearman >= 0.999994 (I71)
def dq(r, c, t='mean'): return DQ['rows'][r][t][c]
L = ['% written by gen/ol_gen.py: AUROC (%) of the ID images of test6 against their degraded copies, five-ID mean (olx_dq.json); 50 = blind']
for r, lab in (('OL', '\\textbf{\\Ours(Ours)}'), ('OL-l', '\\quad 方向の項だけ'), ('OL-d2', '\\quad 距離の項だけ'), ('CFS-kNN', 'CFS(kNN ヘッド)'), ('CFS', 'CFS(対角ガウス)')):
    v = [lab] + [f1(dq(r, c)) for c in COND] + [f1(mean(dq(r, c) for c in COND))]; L.append(' & '.join(blue(x) if r == 'OL' else x for x in v) + '\\\\')
    if r == 'OL-d2': L.append('\\midrule')
W('tab_degr_ol.tex', L)
L = ['% written by gen/ol_gen.py: the same per ID (olx_dq.json)']
for r, lab in (('OL', '\\textbf{\\Ours(Ours)}'), ('CFS-kNN', 'CFS(kNN)')):
    for i, t in enumerate(TAGS): L.append(' & '.join([lab if i == 0 else '', NAME[t]] + [f1(dq(r, c, t)) for c in COND]) + '\\\\')
    L.append('\\midrule')
W('supp_degr_ol.tex', L)
for r, k in (('OL', 'OL'), ('CFS-kNN', 'CfsKnn'), ('CFS', 'Cfs'), ('OL-l', 'L'), ('OL-d2', 'T')):
    M['Dq' + k + 'Min'] = f1(min(dq(r, c) for c in COND)); M['Dq' + k + 'Max'] = f1(max(dq(r, c) for c in COND)); M['Dq' + k + 'Mean'] = f1(mean(dq(r, c) for c in COND))
M['DqNHigher'] = str(sum(dq('OL', c) > dq('CFS-kNN', c) for c in COND)); M['DqSVHNMax'] = f1(max(dq('OL', c, 'l6svhn') for c in COND)); M['DqINJpeg'] = f1(dq('OL', 'jpeg90', 'in64')); M['DqCfsKnnINJpeg'] = f1(dq('CFS-kNN', 'jpeg90', 'in64'))
M['DqBlurOL'] = f1(dq('OL', 'blur10')); M['DqBlurCfsKnn'] = f1(dq('CFS-kNN', 'blur10')); M['DqBlurHalfOL'] = f1(dq('OL', 'blur05')); M['DqJpegOL'] = f1(dq('OL', 'jpeg90')); M['DqJpegCfsKnn'] = f1(dq('CFS-kNN', 'jpeg90'))
# ---- what the model contributes (olx_var.json; test6, noise seed 0)
VA = LJ('olx_var.json')
def va(b, v, k='score', met='nat'): return VA[b][v][k][met]
def vam(v, k='score', met='nat'): return mean(va(b, v, k, met) for b in TAGS)
for b in TAGS: assert abs(va(b, 'trained') - SP['test7']['original']['rows']['OL'][b]['nat']) < 1.0 and all(VA[b][v]['finite'] for v in VA[b]) and {'rinit', 'rshuf'} <= set(VA[b]), b
BYP = [(b, v) for b in TAGS for v in VA[b] if v.startswith('by_')]; byv = lambda k='score', met='nat': mean(va(b, v, k, met) for b, v in BYP); BYT = sorted({b for b, v in BYP}); bytr = lambda k='score': mean(va(b, 'trained', k) for b, v in BYP)
L = ['% written by gen/ol_gen.py: test6, noise seed 0: the trained model, random weights, shuffled weights (olx_var.json)']
for v, lab in (('trained', '学習済み'), ('rinit', '初期化したまま'), ('rshuf', '並べ替え')):
    c = [lab] + [f1(va(b, v)) for b in TAGS] + [f1(vam(v)), f1(vam(v, met='vflip')), f1(vam(v, 'l')), f1(vam(v, 'd2'))]; L.append(' & '.join(blue(x) if v == 'trained' else x for x in c) + '\\\\')
W('tab_var_ol.tex', L)
L = ['% written by gen/ol_gen.py: the trained model of another dataset (olx_var.json)']
for b, v in BYP: L.append(' & '.join([NAME[b], NAME[v[3:]], f1(va(b, 'trained')), f1(va(b, v)), f1(va(b, v, 'l')), f1(va(b, v, 'd2')), f1(va(b, v, met='vflip'))]) + '\\\\')
W('supp_var_ol.tex', L)
M['VarTrained'] = f1(vam('trained')); M['VarRinit'] = f1(vam('rinit')); M['VarRshuf'] = f1(vam('rshuf')); M['VarRinitVf'] = f1(vam('rinit', met='vflip')); M['VarRshufVf'] = f1(vam('rshuf', met='vflip')); M['VarTrainedVf'] = f1(vam('trained', met='vflip'))
M['VarBy'] = f1(byv()); M['VarByTrained'] = f1(bytr()); M['VarByL'] = f1(byv('l')); M['VarByT'] = f1(byv('d2')); M['VarByVf'] = f1(byv(met='vflip')); M['VarNBy'] = str(len(BYP)); M['VarByMin'] = f1(min(va(b, v) for b, v in BYP)); M['VarByMax'] = f1(max(va(b, v) for b, v in BYP))
M['VarRinitL'] = f1(vam('rinit', 'l')); M['VarRinitT'] = f1(vam('rinit', 'd2')); M['VarTrainedL'] = f1(vam('trained', 'l')); M['VarTrainedT'] = f1(vam('trained', 'd2'))
for b in TAGS: M['VarRinit' + KEY[b]] = f1(va(b, 'rinit')); M['VarRshuf' + KEY[b]] = f1(va(b, 'rshuf'))
# ---- development analyses on val2 (olx_dev.json; mean of noise seeds 0-2)
DV = LJ('olx_dev.json')['rows']
def dv(r, met='nat', t='mean'): return DV[r][t][met]
CSW = [('c0', '0(距離の項だけ)'), ('c1', '1'), ('c2', '2'), ('c4', '4'), ('c8', '8(\\textbf{\\Ours})'), ('c16', '16'), ('c32', '32'), ('c64', '64'), ('cinf', '$\\infty$(方向の項だけ)')]
OSW = [('o_v', '上下反転だけ'), ('o_flips', '反転と180度の回転(3つ)'), ('o_quarter', '90度の回転と対角の反転(4つ)'), ('o_no_mirror', '左右反転を除く6つ'), ('c8', '7つすべて(\\textbf{\\Ours})')]
NSW = [('n50', '50'), ('n100', '100'), ('n200', '200'), ('n500', '500'), ('n1000', '1{,}000'), ('c8', 'すべて(\\textbf{\\Ours})')]
XSW = [('allblocks', '全ブロック'), ('logstd', '対数標準偏差'), ('pk', '定数なし：「そのまま」の事後確率'), ('pk_d2', '定数なし：事後確率と密度の積'), ('pk_d2sub', '定数なし：向きが異なる部分空間だけ')]
def drow(r, lab): c = [lab, f1(dv(r)), f1(dv(r, 'eq_nat')), f1(dv(r, 'vflip'))]; return ' & '.join(blue(x) if r == 'c8' else x for x in c) + '\\\\'
L = ['% written by gen/ol_gen.py: val2, mean of noise seeds 0-2 (olx_dev.json): the constant c and the set of orientations']
L.append('\\multicolumn{4}{@{}l}{{定数 $c$(7つの向き)}}\\\\'); L += [drow(r, lab) for r, lab in CSW]; L.append('\\midrule')
L.append('\\multicolumn{4}{@{}l}{{使う向き($c=8$)}}\\\\'); L += [drow(r, lab) for r, lab in OSW]; L.append('\\midrule')
L.append('\\multicolumn{4}{@{}l}{{分布内の画像の枚数 $n$}}\\\\'); L += [drow(r, lab) for r, lab in NSW]
W('tab_dev_ol.tex', L)
# the same rows, three blocks side by side (full-width table in the main text)
def dcells(r, lab): c = [lab, f1(dv(r)), f1(dv(r, 'eq_nat')), f1(dv(r, 'vflip'))]; return [blue(x) if r == 'c8' else x for x in c]
L = ['% written by gen/ol_gen.py: val2, mean of noise seeds 0-2 (olx_dev.json): c, orientations and n side by side']
for i in range(max(len(CSW), len(OSW), len(NSW))):
    row = []
    for blk in (CSW, OSW, NSW):
        row += dcells(*blk[i]) if i < len(blk) else ['', '', '', '']
    L.append(' & '.join(row) + '\\\\')
W('tab_devw_ol.tex', L)
L = ['% written by gen/ol_gen.py: val2, mean of noise seeds 0-2, per ID (olx_dev.json)']
for grp, rows in (('定数 $c$', CSW), ('使う向き', OSW[:-1]), ('枚数 $n$', NSW[:-1]), ('その他', XSW)):
    for i, (r, lab) in enumerate(rows):
        c = [grp if i == 0 else '', lab] + [f1(dv(r, t=t)) for t in TAGS] + [f1(dv(r)), f1(dv(r, 'vflip')), f1(dv(r, 'eq_nat')), f1(dv(r, 'eq_vflip'))]; L.append(' & '.join(blue(x) if r == 'c8' else x for x in c) + '\\\\')
    L.append('\\midrule')
W('supp_devfull_ol.tex', L)
for r, k in (('c0', 'Czero'), ('c1', 'Cone'), ('c4', 'Cfour'), ('c8', 'Ceight'), ('c16', 'Csixteen'), ('c64', 'Csixtyfour'), ('cinf', 'Cinf'), ('o_v', 'Ov'), ('o_no_mirror', 'Onomirror'), ('o_quarter', 'Oquarter'), ('o_flips', 'Oflips'),
             ('n50', 'Nfifty'), ('n100', 'Nhundred'), ('n500', 'Nfivehundred'), ('n1000', 'Nthousand'), ('allblocks', 'All'), ('logstd', 'Lstd'), ('pk', 'Pk'), ('pk_d2', 'Pkd'), ('pk_d2sub', 'Pkds')):
    M['Dev' + k] = f1(dv(r)); M['DevEq' + k] = f1(dv(r, 'eq_nat'))
M['DevVfCeight'] = f1(dv('c8', 'vflip')); M['DevVfCinf'] = f1(dv('cinf', 'vflip')); M['DevVfPk'] = f1(dv('pk', 'vflip')); M['DevSVHNOv'] = f1(dv('o_v', t='l6svhn')); M['DevSVHNOquarter'] = f1(dv('o_quarter', t='l6svhn'))
M['DevCeightB'] = f2(dv('c8')); M['DevEqCeightB'] = f2(dv('c8', 'eq_nat')); M['DevSVHNAll'] = f1(dv('allblocks', t='l6svhn')); M['DevSVHNCeight'] = f1(dv('c8', t='l6svhn'))
assert max(CSW, key=lambda x: dv(x[0]))[0] in ('c8', 'c16') and abs(dv('c8') - dv('c16')) < 0.2 and dv('c4') < dv('c8') - 0.4 and dv('c32') < dv('c8') and dv('cinf') < dv('c8') - 0.8 and dv('c0') < 90
assert dv('n100') < dv('n500') < dv('c8') and abs(dv('n1000') - dv('c8')) < 0.2 and dv('pk') < dv('c8') - 0.8 and dv('pk_d2') < dv('pk') and dv('o_v') < dv('c8') and dv('logstd') < dv('c8')
# ---- the rounds of the development on val2 (gx_val2*_s0.json; noise seed 0)
RND = [('gx_val2_s0', '入力の振幅に対する応答の比と残差', ['G1', 'G2', 'G3', 'G4', 'C1', 'C2', 'C3', 'S1', 'S2', 'S3', 'G5', 'G6'], []),
       ('gx_val2b_s0', '距離の測り方,振幅の形, 2つの向きを合わせた量', ['M1', 'M2', 'M3', 'M4', 'M5', 'K1', 'K2', 'K3', 'K4', 'P1', 'P2', 'P3'], []),
       ('gx_val2c_s0', 'ブロックAのチャネルごとの量', ['H1', 'H2', 'H2b', 'H3', 'H4', 'H5', 'H6', 'H7', 'LC', 'AC'], ['LCr', 'ACr']),
       ('gx_val2d_s0', '少数のブロック,上下反転した画像との判別の方向', ['N2', 'N3', 'ST', 'STs', 'LQ', 'L1d', 'LN', 'L1', 'L1s'], []),
       ('gx_val2e_s0', '上下反転した平均の反対側へ動かした点への距離($c=1$)', ['A1s', 'A1', 'Ts'], ['A0.5s', 'A2s', 'A4s', 'A1sp']),
       ('gx_val2f_s0', '重みを自動で決める形,片側だけの距離', ['Z1s', 'Z1a', 'TPs', 'TPa'], ['Ls', 'La', 'A2a', 'A4a', 'A8a', 'A16a', 'A8s', 'A16s', 'A64s', 'RKs', 'RKa']),
       ('gx_val2g_s0', '向きを変えた画像と切り貼りした画像(6種類)', ['M8s', 'M8a', 'LMs', 'LMa', 'Q8s', 'PKs'], ['MG8s', 'MJ8s', 'M4s', 'M16s', 'Lv', 'Lr90', 'Lr270', 'Lj2', 'Lj4', 'Lch', 'Lt1', 'Lt2']),
       ('gx_val2h_s0', '7つの向きすべて', ['O8s', 'O8a', 'PKo', 'PKoa', 'JFo', 'JRo'], ['LGo', 'O4s', 'O16s', 'O8s5', 'Lh', 'Lr180'])]      # variants: rows for understanding, not candidates (repeats of earlier rows left out)
def gx(f, r, met='nat'):
    J = LJ(f + '.json'); R = (J['kinds']['amp'] if 'kinds' in J else J['rows'])[r]; return R['mean'].get(met)      # round 1 stores the rows per statistic
L = ['% written by gen/ol_gen.py: the rounds of the development on val2, noise seed 0 (gx_val2*_s0.json): candidates and the variants that were not candidates']; ndev = 0; nvar = 0; BARN, BARE = 95.07, 90.33; VQ = {}
for i, (f, lab, el, vl) in enumerate(RND):
    v = {r: gx(f, r) for r in el}; assert all(x is not None for x in v.values()), (f, v); b = max(el, key=lambda r: v[r]); e = gx(f, b, 'eq_nat'); ndev += len(el); nvar += len(vl)
    q = (v[b] >= BARN and e is not None and e >= BARE); assert q == (f == 'gx_val2h_s0'), (f, b, v[b], e)
    c = [str(i + 1), lab, str(len(el)), f2(v[b]), f2(e) if e is not None else '--', '○' if q else '--']
    if vl:
        w = {r: gx(f, r) for r in vl}; assert all(x is not None for x in w.values()), (f, w); bv = max(vl, key=lambda r: w[r]); ev = gx(f, bv, 'eq_nat'); qv = w[bv] >= BARN and ev >= BARE; VQ[i + 1] = (bv, w[bv], ev, qv)
        c += [str(len(vl)), f2(w[bv]), f2(ev), '○' if qv else '--']
    else: c += ['0', '--', '--', '--']
    L.append(' & '.join(c) + '\\\\')
assert [k for k, x in VQ.items() if x[3]] == [6, 7, 8] and VQ[7][0] == 'MG8s' and abs(VQ[7][1] - 95.61) < 0.006 and VQ[8][0] == 'O8s5'
M['NVar'] = str(nvar); assert nvar == 35
M['VarSixName'] = {'A16a': '全ブロック, $c=16$'}.get(VQ[6][0], VQ[6][0]); M['VarSixNat'] = f2(VQ[6][1]); M['VarSixEq'] = f2(VQ[6][2]); M['VarSevenEq'] = f2(VQ[7][2]); M['VarEightNat'] = f2(VQ[8][1]); M['VarEightEq'] = f2(VQ[8][2])
M['CandEightNat'] = f2(gx('gx_val2h_s0', 'O8s')); M['CandEightEq'] = f2(gx('gx_val2h_s0', 'O8s', 'eq_nat')); M['CandSevenBest'] = f2(max(gx('gx_val2g_s0', r) for r in RND[6][2])); M['CandSevenBestEq'] = f2(gx('gx_val2g_s0', max(RND[6][2], key=lambda r: gx('gx_val2g_s0', r)), 'eq_nat'))
assert VQ[6][0] == 'A16a'
W('supp_devlog_ol.tex', L); M['NDev'] = str(ndev); assert ndev == 62
assert gx('gx_val2h_s0', 'O8s') == max(gx('gx_val2h_s0', r) for r in RND[-1][2]) and abs(gx('gx_val2h_s0', 'O8s') - 95.54) < 0.006
H3 = [LJ(f'gx_val2h_s{i}.json')['rows'] for i in (0, 1, 2)]; M['DevSeeds'] = ', '.join(f2(h['O8s']['mean']['nat']) for h in H3); M['DevSeedsEq'] = ', '.join(f2(h['O8s']['mean']['eq_nat']) for h in H3)
assert abs(mean(h['O8s']['mean']['nat'] for h in H3) - dv('c8')) < 0.006
G7 = LJ('gx_val2g_s0.json')['rows']; cut = [G7[r]['mean']['nat'] for r in ('Lj2', 'Lj4', 'Lch')]; M['DevCutMin'] = f'{min(cut):.0f}'; M['DevCutMax'] = f'{max(cut):.0f}'; assert max(cut) < 86 and G7['Lr90']['mean']['nat'] > 94
GI = LJ('gx_val2g_s0.json')['info']; cs_ = [GI[t]['cos_oracle']['r90'] for t in TAGS]; M['OracleCosMin'] = f2(min(cs_)); M['OracleCosMax'] = f2(max(cs_)); assert min(cs_) > 0.7 and all(GI[t]['cos_oracle']['v'] > 0.55 for t in TAGS)
CQ = LJ('olx_cos.json'); cq = [CQ[t]['cos_pooled'] for t in TAGS]; M['CosMin'] = f2(min(cq)); M['CosMax'] = f2(max(cq)); assert all(CQ[t]['p'] == PB[t] for t in TAGS) and min(cq) > 0.7
M['BarNat'] = f2(BARN); M['BarEq'] = f2(BARE); M['DevDiff'] = '0.003'; assert 'largest difference from the development numbers: 0.003 OK' in open(f'{D}/ol_devcheck_s0.log').read()
# ---- does the amplitude fall (olx_fall.json, olx_diag.json X9), OLR's direction against directions fitted with the OOD sets (X8b), the spread of the re-oriented profiles (X10); val2, seeds 0-2, blocks up to A
FA = LJ('olx_fall.json'); DG = LJ('olx_diag.json')['mean']; assert FA['margin'] == 0.2
for t in TAGS: assert FA['tags'][t]['p'] == PB[t] == DG[t]['p'], t
def pc(v): return f'{100 * v:.0f}'
def zf(v): return f'{v:.1f}'.replace('-', '$-$')
L = ['% written by gen/ol_gen.py: shares of the blocks up to A where the mean amplitude falls (z < -0.2) or rises (z > +0.2), z at block A (olx_fall.json)']
for t in TAGS:
    d = FA['tags'][t]; L.append(' & '.join([NAME[t], str(d['p']), pc(d['orient']['fall']), pc(d['orient']['rise']), zf(d['orient']['atA']), pc(d['ood']['fall']), pc(d['ood']['rise']), zf(d['ood']['atA']), pc(d['id_val2']['fall']), pc(d['id_val2']['rise']), zf(d['id_val2']['atA'])]) + '\\\\')
W('tab_fall_ol.tex', L)
fo = [FA['tags'][t]['orient'] for t in TAGS]; fd = [FA['tags'][t]['ood'] for t in TAGS]; fi = [FA['tags'][t]['id_val2'] for t in TAGS]
M['FallOrMin'] = pc(min(x['fall'] for x in fo)); M['FallOrMax'] = pc(max(x['fall'] for x in fo)); M['RiseOrMax'] = pc(max(x['rise'] for x in fo))
M['ZaOrMin'] = f"{min(x['atA'] for x in fo):.1f}"; M['ZaOrMax'] = f"{max(x['atA'] for x in fo):.1f}"; assert max(x['atA'] for x in fo) < -0.2 and max(x['rise'] for x in fo) < 0.25 and min(x['fall'] for x in fo) > max(x['rise'] for x in fo)
M['FallOodMin'] = pc(min(x['fall'] for x in fd)); M['FallOodMax'] = pc(max(x['fall'] for x in fd)); M['RiseOodSVHN'] = pc(FA['tags']['l6svhn']['ood']['rise']); M['FallOodSVHN'] = pc(FA['tags']['l6svhn']['ood']['fall'])
assert FA['tags']['l6svhn']['ood']['rise'] > FA['tags']['l6svhn']['ood']['fall'] and all(FA['tags'][t]['ood']['fall'] > FA['tags'][t]['ood']['rise'] for t in TAGS if t != 'l6svhn')
za = [v['atA'] for t in TAGS for v in FA['tags'][t]['ood_sets'].values()]; M['NSetsA'] = str(len(za)); M['NSetsAMargin'] = str(sum(v < -0.2 for v in za)); assert max(za) < 0 and all(DG[t]['atA_ood'] == 1.0 for t in TAGS)
assert all(x['fall'] == 0 and x['rise'] == 0 for x in fi) and max(abs(x['atA']) for x in fi) < 0.1
M['IdZaAbsMax'] = f2(max(abs(x['atA']) for x in fi))
L = ['% written by gen/ol_gen.py: the same with the sign only (olx_diag.json X9): share of the blocks up to A below the ID mean']
for t in TAGS:
    d = DG[t]; L.append(' & '.join([NAME[t], str(d['p']), pc(d['orient']), pc(d['ood_mean']), pc(d['ood_min']), pc(d['ood_max']), pc(d['id_val2']), pc(d['atA_orient']), pc(d['atA_ood'])]) + '\\\\')
W('supp_fallsign_ol.tex', L)
M['SignIdMin'] = pc(min(DG[t]['id_val2'] for t in TAGS)); M['SignIdMax'] = pc(max(DG[t]['id_val2'] for t in TAGS))
L = ['% written by gen/ol_gen.py: the spread of the re-oriented profiles against S (olx_diag.json X10)']
for t in TAGS:
    k = DG[t]['kappa']; L.append(' & '.join([NAME[t], str(DG[t]['p']), f2(k['pooled']['trace']), f2(k['pooled']['det']), f2(k['pooled']['eig_min']), f2(k['pooled']['eig_med']), f2(k['pooled']['eig_max']), f2(k['within']['trace']), f2(k['id_val2']['trace'])]) + '\\\\')
W('supp_kappa_ol.tex', L)
kp = [DG[t]['kappa']['pooled']['trace'] for t in TAGS]; M['KapMin'] = f2(min(kp)); M['KapMax'] = f2(max(kp)); M['KapNLeOne'] = str(sum(v <= 1.0 for v in kp)); M['KapCelebA'] = f2(DG['dpceleba']['kappa']['pooled']['trace'])
kw = [DG[t]['kappa']['within']['trace'] for t in TAGS]; M['KapWMin'] = f2(min(kw)); M['KapWMax'] = f2(max(kw)); ki = [DG[t]['kappa']['id_val2']['trace'] for t in TAGS]; M['KapIdMin'] = f2(min(ki)); M['KapIdMax'] = f2(max(ki))
assert max(kp) == DG['dpceleba']['kappa']['pooled']['trace'] and sorted(kp)[-2] < 1.125 and max(kw) < 1.0
X8 = {t: DG[t]['auroc'] for t in TAGS}; XS = {t: DG[t]['auroc_sets'] for t in TAGS}
L = ['% written by gen/ol_gen.py: val2, blocks up to A, seeds 0-2, two folds: OLR, its direction term, LDA directions fitted with the OOD sets; cosine (olx_diag.json X8b)']
for k, lab in (('OLR', '\\textbf{\\Ours(Ours)}'), ('dir', '\\Ours{}の方向の項だけ'), ('lda_pool', '分布外集合をまとめて決めた方向'), ('lda_set', '分布外集合ごとに決めた方向')):
    L.append(' & '.join([lab] + [f1(X8[t][k]) for t in TAGS] + [f1(mean(X8[t][k] for t in TAGS))]) + '\\\\')
    if k == 'dir': L.append('\\midrule')
L.append('\\midrule'); L.append(' & '.join(['余弦(\\Ours{}の方向とまとめて決めた方向)'] + [f2(DG[t]['cos_pool']) for t in TAGS] + [f2(mean(DG[t]['cos_pool'] for t in TAGS))]) + '\\\\')
W('supp_oracle_ol.tex', L)
cq = [DG[t]['cos_pool'] for t in TAGS]; M['CosMin'] = f2(min(cq)); M['CosMax'] = f2(max(cq)); assert all(abs(DG[t]['cos_pool'] - CQ[t]['cos_pooled']) < 0.05 for t in TAGS)
for k, q in (('OLR', 'OLR'), ('dir', 'Dir'), ('lda_pool', 'Pool'), ('lda_set', 'Set')): M['X' + q] = f1(mean(X8[t][k] for t in TAGS))
M['XSsbSet'] = f1(XS['in64']['lda_set']['fresh-in64_ssb_hard']); M['XSsbOLR'] = f1(XS['in64']['OLR']['fresh-in64_ssb_hard']); M['XCHundredSet'] = f1(XS['edmc10']['lda_set']['fresh-cifar100_test']); M['XCHundredOLR'] = f1(XS['edmc10']['OLR']['fresh-cifar100_test'])
assert mean(X8[t]['OLR'] for t in TAGS) < mean(X8[t]['lda_pool'] for t in TAGS) < mean(X8[t]['lda_set'] for t in TAGS) and abs(mean(X8[t]['OLR'] for t in TAGS) - dv('c8')) < 0.1
M['DevMGeight'] = f2(LJ('gx_val2g_s0.json')['rows']['MG8s']['mean']['nat'])
# ---- 256x256 (olx_hr.json), the source files of the sets (ssb_diag.json, ssb_diag2.json)
HSET = ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o']
if os.path.exists(f'{D}/olx_hr.json'):
    HR = LJ('olx_hr.json'); HC, HD = HR['in256c'], HR['in256d']; assert HC['A'] == 'output_blocks.2' and all(v == 1000 for v in HC['n'].values())
    def hrow(T, r): d = T['rows'][r]; return [d['auroc'][o] for o in HSET] + [d['near'], d['far'], d['natural'], d['vflip']]
    L = ['% written by gen/ol_gen.py: the 256x256 tier and its 112-px control: AUROC (%) per set, means, upside-down ID (olx_hr.json)']
    for T, tl in ((HC, '224ピクセルを拡大'), (HD, '112ピクセル経由(対照)')):
        pb = [max(round(hrow(T, r)[j], 1) for r in ('CFS', 'CFS-kNN')) for j in range(9)]
        for i, (r, lab) in enumerate((('OL', '\\textbf{\\Ours(Ours)}'), ('CFS-kNN', 'CFS(kNN ヘッド)'), ('CFS', 'CFS(対角ガウス)'))):
            v = hrow(T, r); c = [tl if i == 0 else '', lab] + [(f'\\cellcolor{{firstblue}}\\textbf{{{f1(x)}}}' if (r == 'OL' and round(x, 1) >= pb[j]) else f1(x)) for j, x in enumerate(v)]; L.append(' & '.join(c) + '\\\\')
        L.append('\\midrule')
    W('tab_hr_ol.tex', L)
    L = ['% written by gen/ol_gen.py: the 256x256 tier, the parts of the score (olx_hr.json)']
    for T, tl in ((HC, '224ピクセルを拡大'), (HD, '112ピクセル経由')):
        for i, (r, lab) in enumerate((('OL', '\\textbf{\\Ours(Ours)}'), ('OLa', '全ブロック'), ('OLl', '方向の項だけ'), ('OLt', '距離の項だけ'))):
            L.append(' & '.join([tl if i == 0 else '', lab] + [f1(x) for x in hrow(T, r)] + [f1(T['rows'][r]['fpr95'])]) + '\\\\')
        L.append('\\midrule')
    W('supp_hr_ol.tex', L)
    for T, k in ((HC, 'Hr'), (HD, 'Hd')):
        for r, q in (('OL', 'OL'), ('CFS', 'Cfs'), ('CFS-kNN', 'CfsKnn'), ('OLl', 'L'), ('OLt', 'T'), ('OLa', 'All')):
            d = T['rows'][r]; M[k + q] = f1(d['natural']); M[k + q + 'Near'] = f1(d['near']); M[k + q + 'Far'] = f1(d['far']); M[k + q + 'Vf'] = f1(d['vflip'])
        for c, q in (('CFS', 'Cfs'), ('CFS-kNN', 'CfsKnn')):
            b = T['boot'][c]; M[k + 'Gap' + q] = sg(b['diff']); M[k + 'Gap' + q + 'Lo'] = sg(b['ci'][0]); M[k + 'Gap' + q + 'Hi'] = sg(b['ci'][1]); M[k + 'Wtl' + q] = '/'.join(str(T['wtl'][c][x]) for x in ('wins', 'ties', 'losses'))
        for o, q in zip(HSET, ('Ssb', 'Ninco', 'Inat', 'Tex', 'Oio')): M[k + 'OL' + q] = f1(T['rows']['OL']['auroc'][o]); M[k + 'CfsKnn' + q] = f1(T['rows']['CFS-kNN']['auroc'][o]); M[k + 'Cfs' + q] = f1(T['rows']['CFS']['auroc'][o])
        M[k + 'Hf'] = f1(T['hflip_OL']); M[k + 'BestName'] = max(('CFS', 'CFS-kNN'), key=lambda r: T['rows'][r]['natural'])
    HB = {r['method']: r for r in [json.loads(l) for l in open(f'{D}/bench_rf256.jsonl') if l.strip()]}; hs = {k: v['sec_per_img'] for k, v in HB.items()}
    M['HrFwdA'] = f1(hs['fwd_A']); M['HrFwdAll'] = f1(hs['fwd_all']); M['HrSecCfsStop'] = f1(hs['cfs_stop']); M['HrFwdAShare'] = f'{100 * hs["fwd_A"] / hs["fwd_all"]:.0f}'
    SB = LJ('ssb_diag.json'); S2 = LJ('ssb_diag2.json'); SBS = ['imagenet_test'] + HSET; SBN = {'imagenet_test': 'ImageNet(分布内)', 'ssb_hard': 'SSB-hard', 'ninco': 'NINCO', 'inaturalist': 'iNaturalist', 'textures': 'Textures', 'openimage_o': 'OpenImage-O'}
    assert all(v == 0 for v in SB['a_source']['unreadable'].values()); sm, cv = SB['a_source']['median'], SB['b_cv']; sh = {st_: 100 * S2['sampling_share'][st_]['2'] for st_ in SBS}; bst = lambda d, *ks: max(d[k] for k in ks)
    L = ['% written by gen/ol_gen.py: source files and low-level statistics of the 256x256 tier (ssb_diag.json, ssb_diag2.json), OL under the 112-px control (olx_hr.json)']
    for st_ in SBS:
        c = [SBN[st_], f"{sm[st_]['shorter_side']:.0f}", f2(sm[st_]['aspect_long_over_short']), f2(sm[st_]['bits_per_pixel']), f1(sm[st_]['jpeg_luma_qtable_mean']), f"{sh[st_]:.0f}"]
        c += ['--'] * 4 if st_ == 'imagenet_test' else [f1(bst(cv[st_], 'source_facts_lr', 'source_facts_hgb')), f1(bst(cv[st_], 'image_stats_lr', 'image_stats_hgb')), f1(bst(S2['cv'][st_], 'lr', 'hgb')), f1(HD['rows']['OL']['auroc'][st_])]
        L.append(' & '.join(c) + '\\\\')
        if st_ == 'imagenet_test': L.append('\\midrule')
    W('supp_ssbsrc_ol.tex', L)
    hc, hd = HC['rows'], HD['rows']; assert hc['OL']['vflip'] < min(six(met='vflip', t=t) for t in TAGS) and abs(hc['OL']['natural'] - hc['OLt']['natural']) < 1 and hc['OLl']['natural'] < 65 and hc['OLa']['natural'] > hc['OL']['natural']
    assert hc['OL']['natural'] > hc['CFS']['natural'] > hc['CFS-kNN']['natural'] and hd['OL']['natural'] > hd['CFS']['natural'] > hd['CFS-kNN']['natural'] and HC['boot']['CFS']['ci'][0] > 0 and HD['boot']['CFS']['ci'][0] > 0
    M['HrDelta'] = f2(INFO['in256c']['Delta']); M['HdDelta'] = f2(INFO['in256d']['Delta']); assert INFO['in256c']['Delta'] < min(INFO[t]['Delta'] for t in TAGS)
    M['HdOLSsbRank'] = str(1 + sorted([HD['rows']['OL']['auroc'][o] for o in HSET], reverse=True).index(HD['rows']['OL']['auroc']['ssb_hard']))
    M['SsbShort'] = f"{sm['ssb_hard']['shorter_side']:.0f}"; M['InShort'] = f"{sm['imagenet_test']['shorter_side']:.0f}"; M['SsbShare'] = f1(sh['ssb_hard']); M['InShare'] = f1(sh['imagenet_test'])
    M['SsbFacts'] = f1(bst(cv['ssb_hard'], 'source_facts_lr', 'source_facts_hgb')); M['FactsMin'] = f1(min(bst(cv[o], 'source_facts_lr', 'source_facts_hgb') for o in HSET))
else:
    for n in ('tab_hr_ol.tex', 'supp_hr_ol.tex', 'supp_ssbsrc_ol.tex'): open(f'{OUT}/{n}', 'w').write('\\multicolumn{2}{l}{(pending)}\\\\\\bottomrule\n')
    print('WARNING: olx_hr.json is missing; the 256x256 tables and macros are placeholders')
# ---- sentences of the text that are not macros
for r in ('NLL', 'TT', 'MSMA*', 'DiffPath'): assert jm(None, 'edmc10') - jm(r, 'edmc10') > 20 and jm(None, 'in64') - jm(r, 'in64') > 20, r          # 'more than 20 points below OL on CIFAR-10 and ImageNet-64'
assert {r for v in lostp.values() for r in v} <= {'NLL', 'TT', 'MSMA*', 'DiffPath', 'IC-png', 'IC-j2k', 'EigenScore'} and all(t == 'l6svhn' for t, o in lostp)
assert min(six(t=t) for t in ('l6svhn', 'dpceleba', 'fmnist')) > 96 and min(min(s(sp, t=t) for sp in SPO) for t in ('l6svhn', 'dpceleba', 'fmnist')) == min(s(sp, t='l6svhn') for sp in SPO)
for t in ('edmc10', 'dpceleba', 'in64'): assert abs(mean(SE[sp]['original'][t]['hflip'] for sp in SPO) - 50) < 1, t                                # mirrored natural images are not told apart
assert six(t='l6svhn') - six('OLl', t='l6svhn') > 1.5 and six(t='in64') - six('OLl', t='in64') > 1.5 and all(abs(six(t=t) - six('OLl', t=t)) < 0.5 for t in ('edmc10', 'dpceleba', 'fmnist'))
assert all(k.startswith('fmnist|') for k in eqlost) and len(eqlost) == 4
assert six(t='in64') - six(cond='eq50', t='in64') > 10 and six(t='edmc10') - six(cond='eq50', t='edmc10') > 5 and all(abs(six(t=t) - six(cond='eq50', t=t)) < 1 for t in ('l6svhn', 'dpceleba', 'fmnist'))
assert min(BYP, key=lambda x: va(*x)) == ('edmc10', 'by_l6svhn') and max(BYP, key=lambda x: va(*x)) == ('l6svhn', 'by_edmc10')
assert abs(float(M['EqDropOL']) - float(M['EqDropCfsKnn'])) < 0.3 and abs(float(M['EqGapCfsKnn']) - float(M['GapCfsKnn'])) < 0.3      # 'the gap is the same as with the original images'
with open(f'{OUT}/numbers_ol.tex', 'w') as fo:
    fo.write('% written by gen/ol_gen.py\n')
    for k, v in M.items(): assert re.fullmatch(r'[A-Za-z]+', k), k; fo.write(f'\\newcommand{{\\ol{k}}}{{{v}}}\n')
print('numbers_ol.tex:', len(M), 'macros')
for k in ('TestOL', 'BestPriorMean', 'GapCfsKnn', 'SixMin', 'SixMax', 'SixMean', 'SixVf', 'SecOL', 'SecRatioCfsKnn', 'EqOL', 'EqCfsKnn', 'EqGapCfsKnn', 'VarTrained', 'VarRinit', 'VarRshuf', 'VarBy', 'NStored', 'DevCeight', 'XPool', 'NLostPrior', 'FallOrMin', 'FallOrMax', 'KapMin', 'KapMax', 'HrOL', 'HdOL', 'FactsMin'): print(' ', k, '=', M[k])
