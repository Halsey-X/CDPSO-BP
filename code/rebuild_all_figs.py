"""重画全部 7 张图，落实用户三项要求：
  1. 图中不写标题（面板标题 suptitle 全部删除，交给正文题注）
  2. 图例不得与曲线/数据重叠
  3. 文字标注不得压曲线或数据点

同时把敏感性分析扩到 3 个数据集（原仅 Heart，避免单数据集结论不可靠）。
"""
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

import run_experiment as R

BASE = r'D:/02XIHO Files/Daily Work/03Scientific Research/00Article_File/07改进粒子群算法'
RES = f'{BASE}/results'
OUT = f'{BASE}/14_投稿包_ESWA/01_稿件/图表'
MM = 1 / 25.4

# ESWA/Elsevier 惯例：图内不放标题与图号
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 8,
    'axes.linewidth': 0.7, 'axes.labelsize': 8, 'axes.titlesize': 0,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.2,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_PSO, C_IPS, C_CD = '#7F7F7F', '#E36C0A', '#1F5FA8'
C_CTRL, C_NEU, C_GRN = '#C0504D', '#B0B0B0', '#2E8B57'
DS_ALL = ['Heart', 'LoL', 'Pima', 'WDBC', 'Ionosphere', 'Sonar',
          'Banknote', 'Spambase', 'Haberman', 'Parkinson', 'Blood', 'Phoneme']


def finish(fig, fname, wspace=0.30):
    fig.subplots_adjust(wspace=wspace)
    p = f'{OUT}/{fname}'
    fig.savefig(p, dpi=300)
    plt.close(fig)
    from PIL import Image
    im = Image.open(p)
    d = im.info.get('dpi', (300, 300))
    print(f'  {fname:32s} {im.size[0]}x{im.size[1]}px  '
          f'{im.size[0]/d[0]*25.4:.0f}mm  {os.path.getsize(p)/1024:.0f}KB')


def load_acc():
    lines = [ln for ln in open(f'{RES}/raw_results.csv', encoding='utf-8-sig')
             if not ln.lstrip('\ufeff').startswith('#')]
    acc = defaultdict(list)
    for r in csv.DictReader(lines, skipinitialspace=True):
        if r.get('accuracy') not in ('', 'nan', None):
            acc[(r['dataset'], r['method'])].append(float(r['accuracy']))
    return acc


acc = load_acc()

# ===========================================================================
# Fig. 1  Search-stage convergence  (no titles, legend outside curves)
# ===========================================================================
sc = list(csv.DictReader(open(f'{RES}/search_curves.csv', encoding='utf-8-sig')))
tr = defaultdict(list)
for r in sc:
    tr[(r['dataset'], r['method'])].append(
        (int(r['gen']), float(r['gbest_fit']), float(r['rho']), float(r['inertia'])))

SEL = ['LoL', 'Heart', 'Sonar']
SUB = {'LoL': '(a)  LoL  (n = 9,879)', 'Heart': '(b)  Heart  (n = 918)',
       'Sonar': '(c)  Sonar  (n = 208)'}
fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM))
axes = axes.ravel()
for k, ds in enumerate(SEL):
    ax = axes[k]
    for lbl, col, ls in [('PSO', C_PSO, '-'), ('IPSO', C_IPS, '--'),
                         ('CDPSO', C_CD, '-')]:
        arr = tr[(ds, lbl)]
        gens = sorted(set(x[0] for x in arr))
        fit = np.array([[x[1] for x in arr if x[0] == g][0] for g in gens])
        ax.plot(gens, fit, ls, color=col, lw=1.0, label=lbl)
        ax.fill_between(gens, fit - fit.std(), fit + fit.std(), color=col,
                        alpha=0.12, lw=0)
    ax.set_xlabel('Generation')
    if k == 0:
        ax.set_ylabel('Search fitness (MSE)')
    # 面板标识放在坐标区左上角空白处（不是标题）
    ax.text(0.0, 1.03, SUB[ds], transform=ax.transAxes, fontsize=7.2,
            va='bottom', ha='left', color='#202020')
    ax.set_xlim(1, 40)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(color='#EFEFEF', lw=0.5, zorder=0)
hs = [plt.Line2D([], [], color=c, ls=ls, lw=1.1, label=l)
      for l, c, ls in [('PSO', C_PSO, '-'), ('IPSO', C_IPS, '--'),
                       ('CDPSO', C_CD, '-')]]
fig.tight_layout(rect=(0, 0.075, 1, 1))
fig.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=3,
           frameon=False, fontsize=7.2, handlelength=1.8, columnspacing=1.8)
fig.subplots_adjust(wspace=0.24)
p = f'{OUT}/Fig1_search_convergence.png'
fig.savefig(p, dpi=300)
plt.close(fig)
from PIL import Image as _I
_im = _I.open(p)
print(f'  Fig1_search_convergence.png                {_im.size[0]}x{_im.size[1]}px  '
      f'{os.path.getsize(p)/1024:.0f}KB')
print('Fig1 完成（无标题，图例置底）')

# ===========================================================================
# Fig. 2  Diversity + inertia  (legend must not overlap curves)
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 58 * MM))
axes = axes.ravel()
COL = {'LoL': C_CD, 'Heart': C_IPS, 'Sonar': C_GRN}

ax = axes[0]
for ds in SEL:
    arr = tr[(ds, 'CDPSO')]
    gens = sorted(set(x[0] for x in arr))
    rho = np.array([[x[2] for x in arr if x[0] == g][0] for g in gens])
    ax.plot(gens, rho, lw=1.0, color=COL[ds], label=ds)
ax.axhline(R.RHO_MIN, color=C_CTRL, ls='--', lw=0.9)
ax.set_xlabel('Generation')
ax.set_ylabel('Diversity retention $\\rho(t)$')
ax.set_xlim(1, 40)
ax.set_ylim(0, 1.0)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.text(0.0, 1.03, '(a)  Retention collapses, then oscillates',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
# 图例外置于右侧
hs = [plt.Line2D([], [], color=COL[d], lw=1.2, label=d) for d in SEL]
hs.append(plt.Line2D([], [], color=C_CTRL, ls='--', lw=1.0,
                     label=f'trigger $\\rho$ < {R.RHO_MIN}'))
ax.legend(handles=hs, loc='upper left', bbox_to_anchor=(1.02, 1.0),
          frameon=False, fontsize=7.0, handlelength=1.5, borderaxespad=0)
ax.grid(color='#EFEFEF', lw=0.5, zorder=0)

ax = axes[1]
base_w = [R.W_MIN + (R.W_MAX - R.W_MIN) * (1 - t / R.T_MAX) ** 2
          for t in range(1, R.T_MAX + 1)]
ax.plot(range(1, R.T_MAX + 1), base_w, '--', color=C_NEU, lw=1.1,
        label='schedule baseline')
for ds in SEL:
    arr = tr[(ds, 'CDPSO')]
    gens = sorted(set(x[0] for x in arr))
    w = np.array([[x[3] for x in arr if x[0] == g][0] for g in gens])
    ax.plot(gens, w, lw=1.0, color=COL[ds], label=ds)
ax.set_xlabel('Generation')
ax.set_ylabel('Inertia weight $w(t)$')
ax.set_xlim(1, 40)
ax.set_ylim(0.12, 1.02)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.text(0.0, 1.03, '(b)  Weight stays below the baseline',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False,
          fontsize=7.0, handlelength=1.5, borderaxespad=0, ncol=1)
ax.grid(color='#EFEFEF', lw=0.5, zorder=0)
finish(fig, 'Fig2_diversity_dynamics.png', wspace=0.52)
print('Fig2 完成（两图例均外置）')

# ===========================================================================
# Fig. 3  Break-even  (was mis-filed; rebuild)
# ===========================================================================
bc = list(csv.DictReader(open(f'{RES}/budget_curve.csv', encoding='utf-8-sig')))
for r in bc:
    r['epoch'] = int(r['epoch'])
    for _k in ('accuracy', 'f1', 'auc', 'bal_acc'):
        r[_k] = float(r[_k]) if r[_k] not in ('', 'nan', None) else np.nan
EPS = sorted(set(r['epoch'] for r in bc))


def curve(d):
    return [np.mean([r['accuracy'] for r in bc
                     if r['dataset'] == d and r['epoch'] == E]) for E in EPS]


be = []
for d in DS_ALL:
    tgt = np.mean(acc[(d, 'CDPSO-BP')])
    ys = curve(d)
    be.append((d, next((E for E, y in zip(EPS, ys) if y >= tgt), None)))
MED = int(np.median([b for _, b in be if b is not None]))

fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM))
axes = axes.ravel()

ax = axes[0]
for d in DS_ALL:
    ax.plot(EPS, curve(d), '-', color='#A8BED4', lw=0.7, zorder=1)
    ax.plot([30], [curve(d)[EPS.index(30)]], 'o', ms=3.0, color=C_CD, zorder=4)
ax.axvline(30, color=C_NEU, ls=':', lw=0.8, zorder=0)
ax.set_xlabel('BP gradient budget (epochs)')
ax.set_ylabel('Test accuracy')
ax.set_xlim(0, 310)
ax.set_ylim(0.47, 1.03)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.text(0.0, 1.03, '(a)  Budget curves and the 30-epoch point',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
ax.text(36, 0.492, 'CDPSO-BP: 30 epochs', fontsize=6.5, color='#505050',
        va='bottom')
ax.text(305, 0.52, 'plain BP, 12 datasets', fontsize=6.6, color='#5A7FA5',
        ha='right', va='bottom')
ax.grid(axis='y', color='#EFEFEF', lw=0.5, zorder=0)

ax = axes[1]
be_sorted = sorted(be, key=lambda x: (x[1] is None, x[1] or 0))
names = [d for d, _ in be_sorted]
vals = [(b if b is not None else 312) for _, b in be_sorted]
cols = [C_CD if b is not None else C_CTRL for _, b in be_sorted]
yp = np.arange(len(names))
ax.barh(yp, vals, color=cols, height=0.6)
ax.set_yticks(yp)
ax.set_yticklabels(names, fontsize=7.2)
ax.axvline(MED, color=C_IPS, ls='--', lw=1.0, zorder=4)
ax.set_xlabel('Break-even epoch $E^*$')
ax.set_xlim(0, 330)
ax.set_ylim(-0.7, 11.9)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.text(0.0, 1.03, '(b)  Epochs of plain BP needed to match',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
# 关键：两条说明放坐标区**外**（右侧），绝不压条形
ax.text(1.015, 0.97, 'median $E^*$ = %d' % MED, color=C_IPS, fontsize=6.6,
        ha='left', va='top', transform=ax.transAxes)
ax.text(1.015, 0.03, 'red: $E^*$ > 300', color=C_CTRL, fontsize=6.4,
        ha='left', va='bottom', transform=ax.transAxes)
ax.grid(axis='x', color='#EFEFEF', lw=0.5, zorder=0)
finish(fig, 'Fig3_breakeven.png', wspace=0.56)
print('Fig3 完成')

# ===========================================================================
# Fig. 4  Per-dataset paired differences
# ===========================================================================
def se_pair(d, m1, m2):
    a, b = np.array(acc[(d, m1)]), np.array(acc[(d, m2)])
    return np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


order = sorted(DS_ALL, key=lambda d: np.mean(acc[(d, 'CDPSO-BP')])
               - np.mean(acc[(d, 'BP-30')]))
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM))
axes = axes.ravel()
yp = np.arange(len(order))

ax = axes[0]
d30 = np.array([np.mean(acc[(d, 'CDPSO-BP')]) - np.mean(acc[(d, 'BP-30')])
                for d in order]) * 100
e30 = np.array([se_pair(d, 'CDPSO-BP', 'BP-30') for d in order]) * 100
ax.barh(yp, d30, xerr=e30, color=C_CD, height=0.58,
        error_kw=dict(ecolor='#606060', lw=0.7, capsize=1.6))
ax.set_yticks(yp)
ax.set_yticklabels(order, fontsize=7.2)
ax.axvline(0, color='#333333', lw=0.8, zorder=3)
ax.set_xlim(0, 12.4)
ax.xaxis.set_major_locator(MultipleLocator(2))
for i, (v, e) in enumerate(zip(d30, e30)):
    if v + e > 10.6:     # 接近 x 轴上限，标签放条内
        ax.text(v - 0.3, i, f'{v:+.1f}', va='center', ha='right',
                fontsize=6.8, color='white', fontweight='bold')
    else:
        ax.text(v + e + 0.3, i, f'{v:+.1f}', va='center', fontsize=6.8,
                color='#1A1A1A')
ax.set_xlabel('CDPSO-BP $-$ BP-30 (pp)')
ax.text(0.0, 1.03, '(a)  Equal-budget control: wins on all 12',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
ax.grid(axis='x', color='#EFEFEF', lw=0.5, zorder=0)

ax = axes[1]
dBP = np.array([np.mean(acc[(d, 'CDPSO-BP')]) - np.mean(acc[(d, 'BP')])
                for d in order]) * 100
eBP = np.array([se_pair(d, 'CDPSO-BP', 'BP') for d in order]) * 100
ax.barh(yp, dBP, xerr=eBP, color=[C_CD if v > 0 else C_CTRL for v in dBP],
        height=0.58, error_kw=dict(ecolor='#606060', lw=0.7, capsize=1.6))
ax.set_yticks(yp)
ax.set_yticklabels([])
ax.axvline(0, color='#333333', lw=0.8, zorder=3)
lim = 10.0
for i, (v, e) in enumerate(zip(dBP, eBP)):
    right = v >= 0
    end = v + e if right else v - e
    pad = 0.35
    ax.text(end + (pad if right else -pad), i, f'{v:+.1f}', va='center',
            ha='left' if right else 'right', fontsize=6.8,
            color=C_CD if right else C_CTRL)
ax.set_xlim(-lim, lim)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.set_xlabel('CDPSO-BP $-$ fully trained BP (pp)')
ax.text(0.0, 1.03, '(b)  Strong control: 6 wins, 6 losses',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
ax.grid(axis='x', color='#EFEFEF', lw=0.5, zorder=0)
ax.legend(handles=[Patch(facecolor=C_CD, label='CDPSO-BP better'),
                   Patch(facecolor=C_CTRL, label='fully trained BP better')],
          loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
          frameon=False, fontsize=7.0, handlelength=1.2, handleheight=0.8,
          columnspacing=1.4)
finish(fig, 'Fig4_per_dataset.png', wspace=0.12)
print('Fig4 完成')

# ===========================================================================
# Fig. 5  Metric divergence  (fix label collision)
# ===========================================================================
POS = {'Heart': 0.553, 'LoL': 0.499, 'Pima': 0.349, 'WDBC': 0.373,
       'Ionosphere': 0.641, 'Sonar': 0.466, 'Banknote': 0.445,
       'Spambase': 0.394, 'Haberman': 0.265, 'Parkinson': 0.754,
       'Blood': 0.238, 'Phoneme': 0.207}
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM))
axes = axes.ravel()

ax = axes[0]
xs, av, uv, bv = [], [], [], []
for d in DS_ALL:
    sub = [r for r in bc if r['dataset'] == d and r['epoch'] == 30]
    if not sub:
        continue
    xs.append(POS[d])
    av.append(np.nanmean([r['accuracy'] for r in sub]))
    uv.append(np.nanmean([r['auc'] for r in sub]))
    bv.append(np.nanmean([r['bal_acc'] for r in sub]))
xs, av, uv, bv = map(np.array, (xs, av, uv, bv))
o = np.argsort(xs)
ax.plot(xs[o], av[o], 'o-', color=C_PSO, lw=1.0, ms=3.6, label='Accuracy')
ax.plot(xs[o], uv[o], 's-', color=C_CD, lw=1.0, ms=3.6, label='AUC')
ax.plot(xs[o], bv[o], '^-', color=C_CTRL, lw=1.0, ms=3.6,
        label='Balanced accuracy')
ax.axvline(0.5, color=C_NEU, ls=':', lw=0.8)
ax.set_xlabel('Positive-class rate')
ax.set_ylabel('Score of the 30-epoch BP baseline')
ax.set_xlim(0.15, 0.82)
ax.set_ylim(0.45, 1.02)
ax.text(0.0, 1.03, '(a)  Accuracy hides the collapse on skewed data',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
# 关键：图例外置右侧；只标注两个极端点，且用引线避开曲线
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3,
          frameon=False, fontsize=7.0, handlelength=1.4, columnspacing=1.2)
# 只标注最不平衡的那个点，且放在点的左上方空白处
dmin = min(DS_ALL, key=lambda d: POS[d])
xi = POS[dmin]
yi = np.interp(xi, xs[np.argsort(xs)], av[np.argsort(xs)])
ax.annotate(dmin, (xi, yi), textcoords='offset points', xytext=(7, 7),
            fontsize=6.2, color='#404040', ha='left',
            arrowprops=dict(arrowstyle='-', color='#909090', lw=0.5))
ax.grid(color='#EFEFEF', lw=0.5, zorder=0)

ax = axes[1]
acc_v, bal_v = {}, {}
for d in DS_ALL:
    sub = [r for r in bc if r['dataset'] == d and r['epoch'] == 30]
    if not sub:
        continue
    acc_v[d] = np.nanmean([r['accuracy'] for r in sub])
    bal_v[d] = np.nanmean([r['bal_acc'] for r in sub])
common = [d for d in DS_ALL if d in acc_v]
ra = {d: r for r, d in enumerate(sorted(common, key=lambda x: -acc_v[x]), 1)}
rb = {d: r for r, d in enumerate(sorted(common, key=lambda x: -bal_v[x]), 1)}
n = len(common)
for d in common:
    ax.plot([ra[d], rb[d]], [ra[d], rb[d]], '-', color='#C8D4E0', lw=0.9,
            zorder=1)
    ax.scatter(ra[d], rb[d], s=24, color=C_CD, zorder=3)
ax.plot([1, n], [1, n], '--', color=C_CTRL, lw=0.9, zorder=2)
ax.set_xlabel('Rank by accuracy (1 = best)')
ax.set_ylabel('Rank by balanced accuracy (1 = best)')
ax.set_xlim(0, n + 1)
ax.set_ylim(0, n + 1)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.text(0.0, 1.03, '(b)  The two metrics rank differently',
        transform=ax.transAxes, fontsize=7.2, va='bottom', color='#202020')
moved = sorted([(d, ra[d], rb[d]) for d in common if abs(ra[d] - rb[d]) >= 3],
               key=lambda x: -abs(x[1] - x[2]))
# 关键：说明文字放**坐标区外下方**，用引线指向点，不压点
ax.text(0.5, -0.20,
        f'{len(moved)} of {n} datasets shift by $\\geq$3 ranks: '
        + ', '.join(d for d, _, _ in moved),
        transform=ax.transAxes, fontsize=6.3, color='#404040', ha='center',
        va='top')
for d, x, y in moved:
    ax.annotate(d, (x, y), textcoords='offset points', xytext=(7, -3),
                fontsize=6.2, color='#404040')
ax.grid(color='#EFEFEF', lw=0.5, zorder=0)
finish(fig, 'Fig5_metric_divergence.png', wspace=0.46)
print('Fig5 完成')

# ===========================================================================
# Fig. 6  Effect sizes
# ===========================================================================
FIN = f'{RES}/final'
EN = {'BP-30': 'BP-30 (equal budget)', 'BP': 'BP (fully trained)',
      'PSO-BP': 'PSO-BP', 'IPSO-BP': 'IPSO-BP', 'IPSO': 'IPSO',
      'IPSO+OBL': 'IPSO+OBL', 'CDPSO-无重启': 'w/o restart',
      'CDPSO-无扰动': 'w/o perturbation',
      'CDPSO-无多样性': 'w/o diversity feedback',
      'CDPSO-无OBL': 'w/o OBL (clean)'}
t1 = list(csv.DictReader(open(f'{FIN}/table1_main_comparison.csv',
                               encoding='utf-8-sig')))
t2 = list(csv.DictReader(open(f'{FIN}/table2_ablation.csv',
                               encoding='utf-8-sig')))
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 60 * MM),
                         gridspec_kw={'width_ratios': [1, 1.18]})
axes = axes.ravel()

ax = axes[0]
lab = [EN.get(r['对照方法'], r['对照方法']) for r in t1][::-1]
val = [float(r['平均差(pp)']) for r in t1][::-1]
sig = ['Holm p = 0.002' in r['判定'] for r in t1][::-1]
yp = np.arange(len(lab))
ax.barh(yp, val, color=[C_CD if s else '#B4C4D4' for s in sig], height=0.58)
ax.set_yticks(yp)
ax.set_yticklabels(lab, fontsize=7.4)
ax.axvline(0, color='#333333', lw=0.8, zorder=3)
ax.set_xlim(-4.6, 1.5)
ax.xaxis.set_major_locator(MultipleLocator(1))
for i, (v, s) in enumerate(zip(val, sig)):
    if v < -1.2:      # 长条：标签放条内右端，避免与面板标识/轴标签相撞
        ax.text(v + 0.14, i, f'{v:+.2f}', va='center', ha='left',
                fontsize=6.9, color='white', fontweight='bold')
    else:
        off = 0.16 if v >= 0 else -0.16
        ax.text(v + off, i, f'{v:+.2f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=6.9,
                color='#1A1A1A', fontweight='bold' if s else 'normal')
ax.set_xlabel('Mean accuracy difference vs CDPSO-BP (pp)')
ax.text(0.0, 1.04, '(a)  Controls: only the equal-budget one is significant',
        transform=ax.transAxes, fontsize=7.2, va='bottom', ha='left',
        color='#202020')
ax.grid(axis='x', color='#EFEFEF', lw=0.5, zorder=0)

ax = axes[1]
lab = [EN.get(r['对照方法'], r['对照方法']) for r in t2][::-1]
val = [float(r['平均差(pp)']) for r in t2][::-1]
yp = np.arange(len(lab))
ax.barh(yp, val, color='#94AFC7', height=0.58)
ax.set_yticks(yp)
ax.set_yticklabels(lab, fontsize=7.4)
ax.axvline(0, color='#333333', lw=0.8, zorder=3)
ax.set_xlim(-0.30, 0.16)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
for i, v in enumerate(val):
    off = 0.008 if v >= 0 else -0.008
    ax.text(v + off, i, f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right',
            fontsize=6.9)
ax.set_xlabel('Mean accuracy change when removed (pp)')
ax.text(0.0, 1.04, '(b)  One-at-a-time ablation: none significant',
        transform=ax.transAxes, fontsize=7.2, va='bottom', ha='left',
        color='#202020')
ax.grid(axis='x', color='#EFEFEF', lw=0.5, zorder=0)
finish(fig, 'Fig6_effects.png', wspace=0.42)
print('Fig6 完成')

# ===========================================================================
# Fig. 7  Parameter sensitivity on 3 datasets (wider evidence base)
# ===========================================================================
sens = list(csv.DictReader(open(f'{RES}/sensitivity_3ds.csv', encoding='utf-8-sig')))
groups = defaultdict(lambda: defaultdict(dict))
default = {}
for r in sens:
    m = r['acc_std'].split('±')[0]
    if not m:
        continue
    v = float(m)
    if r['param'] == 'default':
        default[r['dataset']] = v
    else:
        groups[r['param']][r['dataset']][r['value']] = v

LBL = [('p_m', 'Perturbation\nprobability $p_m$'),
       ('sigma0', 'Perturbation\nscale $\\sigma_0$'),
       ('stagnation', 'Stagnation\nthreshold'),
       ('N', 'Population\nsize $N$'),
       ('T', 'Generations\n$T$')]
DS_S = [d for d in ['Heart', 'Sonar', 'Banknote'] if d in default]
CMAP = {'Heart': C_CD, 'Sonar': C_IPS, 'Banknote': C_GRN}

fig, axes = plt.subplots(1, len(LBL), figsize=(180 * MM, 62 * MM))
axes = np.atleast_1d(axes).ravel()
allv = []
for gk, _ in LBL:
    for d in DS_S:
        for v in groups[gk].get(d, {}).values():
            allv.append((v - default[d]) * 100)
ylim = max(0.8, max(abs(min(allv)), abs(max(allv))) * 1.15)

for ax, (gk, glab) in zip(axes, LBL):
    keys = sorted(groups[gk].get(DS_S[0], {}).keys(), key=lambda x: float(x))
    nk = len(keys)
    w = 0.26
    for j2, d in enumerate(DS_S):
        items = sorted(groups[gk].get(d, {}).items(), key=lambda x: float(x[0]))
        if not items:
            continue
        xs = np.arange(nk) + (j2 - 1) * w
        ys = [(v - default[d]) * 100 for _, v in items]
        ax.bar(xs, ys, width=w, color=CMAP[d])
    # 数值标签：水平、放在条形外端，字体小但可读
    for j2, d in enumerate(DS_S):
        items = sorted(groups[gk].get(d, {}).items(), key=lambda x: float(x[0]))
        if not items:
            continue
        xs = np.arange(nk) + (j2 - 1) * w
        ys = [(v - default[d]) * 100 for _, v in items]
        for x, y in zip(xs, ys):
            ax.text(x, y + (0.05 if y >= 0 else -0.05), f'{y:+.2f}',
                    ha='center', va='bottom' if y >= 0 else 'top',
                    fontsize=4.4, color='#303030', rotation=90)
    ax.set_xticks(np.arange(nk))
    ax.set_xticklabels(keys, fontsize=6.6)
    ax.axhline(0, color='#333333', lw=0.8, zorder=3)
    ax.set_ylim(-ylim, ylim)
    # y 轴只标 3 档，避免拥挤
    ax.set_yticks([-ylim * 0.75, 0, ylim * 0.75])
    ax.set_yticklabels([f'{-ylim*0.75:+.1f}', '0', f'{ylim*0.75:+.1f}'],
                       fontsize=6.6)
    ax.set_xlabel(glab, fontsize=7.4)
    ax.grid(axis='y', color='#EFEFEF', lw=0.5, zorder=0)
axes[0].set_ylabel(r'$\Delta$ accuracy vs default (pp)', fontsize=7.6)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right',
           bbox_to_anchor=(0.995, 1.05), ncol=3, frameon=False, fontsize=7.0,
           handlelength=1.2, columnspacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.subplots_adjust(wspace=0.20)
p = f'{OUT}/Fig7_sensitivity.png'
fig.savefig(p, dpi=300)
plt.close(fig)
from PIL import Image
im = Image.open(p)
print(f'  Fig7_sensitivity.png（3 数据集）          {im.size[0]}x{im.size[1]}px  '
      f'{os.path.getsize(p)/1024:.0f}KB')
print('\n全部 7 张图完成，无标题、图例外置、文字不压数据')
