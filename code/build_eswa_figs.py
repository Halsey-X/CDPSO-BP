"""生成 ESWA 投稿所需的图与图摘要。"""
import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = r'D:/02XIHO Files/Daily Work/03Scientific Research/00Article_File/07改进粒子群算法'
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, '14_投稿包_ESWA', '01_稿件', '图表')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.dpi': 300,
})

DS = ['Banknote', 'WDBC', 'Spambase', 'Sonar', 'Ionosphere', 'LoL',
      'Blood', 'Phoneme', 'Parkinson', 'Pima', 'Haberman', 'Heart']
EPS = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]

bc = list(csv.DictReader(open(os.path.join(RES, 'budget_curve.csv'),
                             encoding='utf-8-sig')))
for r in bc:
    r['epoch'] = int(r['epoch'])
    r['accuracy'] = float(r['accuracy']) if r['accuracy'] not in ('', 'nan') else np.nan

raw = [ln for ln in open(os.path.join(RES, 'raw_results.csv'), encoding='utf-8-sig')
       if not ln.lstrip('\ufeff').startswith('#')]
rawr = list(csv.DictReader(raw, skipinitialspace=True))
acc = defaultdict(list)
for r in rawr:
    if r.get('accuracy') not in ('', 'nan', None):
        acc[(r['dataset'], r['method'])].append(float(r['accuracy']))

# ---------------- Figure 1: break-even 主图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

ax = axes[0]
cd_mean = [np.mean(acc[(d, 'CDPSO-BP')]) for d in DS]
ax.plot([30] * len(DS), cd_mean, 'o', ms=4.5, color='#1F5FA8',
        label='CDPSO-BP (30 ep)', zorder=5)
for d, m in zip(DS, cd_mean):
    sub = [r['accuracy'] for r in bc if r['dataset'] == d]
    ys = [np.mean([r['accuracy'] for r in bc
                   if r['dataset'] == d and r['epoch'] == E]) for E in EPS]
    ax.plot(EPS, ys, '-', lw=0.9, color='#9AB8D8', alpha=0.75, zorder=1)
ax.axvline(30, color='#888', ls=':', lw=0.8, zorder=0)
ax.set_xlabel('BP gradient budget (epochs)')
ax.set_ylabel('Test accuracy')
ax.set_title('(a) Budget curves, 12 datasets', fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc='lower right')
ax.set_xlim(0, 310)

ax = axes[1]
be = []
for d in DS:
    tgt = np.mean(acc[(d, 'CDPSO-BP')])
    ys = [np.mean([r['accuracy'] for r in bc
                   if r['dataset'] == d and r['epoch'] == E]) for E in EPS]
    b = next((E for E, y in zip(EPS, ys) if y >= tgt), None)
    be.append(b if b is not None else np.nan)
be_sorted = sorted([(d, b) for d, b in zip(DS, be)],
                   key=lambda x: (np.isnan(x[1]), x[1] if not np.isnan(x[1]) else 0))
names = [d for d, _ in be_sorted]
vals = [b if not np.isnan(b) else 320 for _, b in be_sorted]
colors = ['#1F5FA8' if not np.isnan(b) else '#C0504D' for _, b in be_sorted]
ypos = np.arange(len(names))
ax.barh(ypos, vals, color=colors, height=0.62)
ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=7.5)
ax.axvline(np.nanmedian(be), color='#E36C0A', ls='--', lw=1.1)
ax.text(np.nanmedian(be) + 4, 0.2, f'median E* = {int(np.nanmedian(be))}',
        color='#E36C0A', fontsize=7.5, va='bottom')
ax.set_xlabel('Break-even epoch E*')
ax.set_title('(b) Epochs of plain BP needed to match CDPSO-BP', fontsize=9)
ax.set_xlim(0, 360)

fig.suptitle('Fig. 1. Budget-equivalent analysis of metaheuristic weight '
             'initialization.', fontsize=9.5, y=1.02)
fig.tight_layout()
p1 = os.path.join(OUT, 'Fig1_breakeven.png')
fig.savefig(p1, bbox_inches='tight')
plt.close(fig)
print('saved:', p1)

# ---------------- Figure 2: 效应量与消融 ----------------
fin = os.path.join(RES, 'final')
t1 = list(csv.DictReader(open(os.path.join(fin, 'table1_main_comparison.csv'),
                               encoding='utf-8-sig')))
t2 = list(csv.DictReader(open(os.path.join(fin, 'table2_ablation.csv'),
                               encoding='utf-8-sig')))

EN = {'BP-30': 'BP-30 (equal budget)', 'BP': 'BP (fully trained)',
      'PSO-BP': 'PSO-BP', 'IPSO-BP': 'IPSO-BP',
      'IPSO': 'IPSO', 'IPSO+OBL': 'IPSO+OBL',
      'CDPSO-无重启': 'w/o restart', 'CDPSO-无扰动': 'w/o perturbation',
      'CDPSO-无多样性': 'w/o diversity feedback',
      'CDPSO-无OBL': 'w/o OBL (clean)'}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax = axes[0]
lab = [EN.get(r['对照方法'], r['对照方法']) for r in t1][::-1]
val = [float(r['平均差(pp)']) for r in t1][::-1]
sig = ['Holm p = 0.002' in r['判定'] for r in t1][::-1]
cols = ['#1F5FA8' if s else '#B8C4D0' for s in sig]
ax.barh(np.arange(len(lab)), val, color=cols, height=0.6)
ax.set_yticks(np.arange(len(lab)))
ax.set_yticklabels(lab, fontsize=8)
ax.axvline(0, color='#333', lw=0.8)
for i, v in enumerate(val):
    ax.text(v + (0.12 if v >= 0 else -0.12), i, f'{v:+.2f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=7.5)
ax.set_xlabel('Mean difference vs CDPSO-BP (pp)')
ax.set_title('(a) Equal-budget and strong controls', fontsize=9)
ax.set_xlim(-4.3, 1.6)

ax = axes[1]
lab = [EN.get(r['对照方法'], r['对照方法']) for r in t2][::-1]
val = [float(r['平均差(pp)']) for r in t2][::-1]
ax.barh(np.arange(len(lab)), val, color='#7A9BB8', height=0.6)
ax.set_yticks(np.arange(len(lab)))
ax.set_yticklabels(lab, fontsize=7.8)
ax.axvline(0, color='#333', lw=0.8)
for i, v in enumerate(val):
    ax.text(v + (0.012 if v >= 0 else -0.012), i, f'{v:+.3f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=7.5)
ax.set_xlabel('Mean difference when removed (pp)')
ax.set_title('(b) One-at-a-time ablation', fontsize=9)
ax.set_xlim(-0.34, 0.34)

fig.suptitle('Fig. 2. Effect sizes under the dataset-level analysis unit '
             '(n = 12).', fontsize=9.5, y=1.03)
fig.tight_layout()
p2 = os.path.join(OUT, 'Fig2_effects.png')
fig.savefig(p2, bbox_inches='tight')
plt.close(fig)
print('saved:', p2)

# ---------------- Graphical abstract (SVG) ----------------
svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360">
<rect width="760" height="360" fill="#FFFFFF"/>
<text x="380" y="34" font-family="Helvetica,Arial,sans-serif" font-size="19" font-weight="bold" text-anchor="middle" fill="#1F3A5F">What is metaheuristic weight initialization worth?</text>

<rect x="42" y="62" width="200" height="112" rx="10" fill="#E8F0F8" stroke="#1F5FA8" stroke-width="1.4"/>
<text x="142" y="86" font-family="Helvetica,Arial,sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#1F3A5F">Budget-equal control</text>
<text x="142" y="108" font-family="Helvetica,Arial,sans-serif" font-size="12" text-anchor="middle" fill="#333333">BP, 30 epochs</text>
<text x="142" y="128" font-family="Helvetica,Arial,sans-serif" font-size="12" text-anchor="middle" fill="#333333">same fine-tuning</text>
<text x="142" y="156" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#C0504D">CDPSO-BP wins</text>
<text x="142" y="172" font-family="Helvetica,Arial,sans-serif" font-size="11" text-anchor="middle" fill="#555555">+3.62 pp, 12/12 datasets</text>

<rect x="280" y="62" width="200" height="112" rx="10" fill="#FFF4E6" stroke="#E36C0A" stroke-width="1.4"/>
<text x="380" y="86" font-family="Helvetica,Arial,sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#7A4A08">How much is that worth?</text>
<text x="380" y="110" font-family="Helvetica,Arial,sans-serif" font-size="12" text-anchor="middle" fill="#333333">13-point budget sweep</text>
<text x="380" y="128" font-family="Helvetica,Arial,sans-serif" font-size="12" text-anchor="middle" fill="#333333">4,680 extra runs</text>
<text x="380" y="156" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#E36C0A">E* = 100 epochs</text>
<text x="380" y="172" font-family="Helvetica,Arial,sans-serif" font-size="11" text-anchor="middle" fill="#555555">median, range 75-200</text>

<rect x="518" y="62" width="200" height="112" rx="10" fill="#F2F2F2" stroke="#777777" stroke-width="1.4"/>
<text x="618" y="86" font-family="Helvetica,Arial,sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#333333">Strong controls</text>
<text x="618" y="110" font-family="Helvetica,Arial,sans-serif" font-size="12" text-anchor="middle" fill="#333333">BP 300 ep + Adam</text>
<text x="618" y="128" font-family="Helvetica,Arial,sans-serif" font-size="12" text-anchor="middle" fill="#333333">CMA-ES + 10 more</text>
<text x="618" y="156" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#555555">No advantage</text>
<text x="618" y="172" font-family="Helvetica,Arial,sans-serif" font-size="11" text-anchor="middle" fill="#555555">p = 0.79, rank 4 of 20</text>

<path d="M 242 118 L 278 118" stroke="#555" stroke-width="1.6" marker-end="url(#a)"/>
<path d="M 480 118 L 516 118" stroke="#555" stroke-width="1.6" marker-end="url(#a)"/>
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/></marker></defs>

<rect x="42" y="206" width="676" height="112" rx="10" fill="#F7F9FB" stroke="#1F5FA8" stroke-width="1.2"/>
<text x="66" y="230" font-family="Helvetica,Arial,sans-serif" font-size="13" font-weight="bold" fill="#1F3A5F">What we conclude</text>
<text x="66" y="252" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#333333">Metaheuristic initialization buys about 100 gradient epochs, at roughly 10x their cost.</text>
<text x="66" y="272" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#333333">It is a way to buy a fixed gradient budget cheaply when compute is constrained, not a route to higher accuracy.</text>
<text x="66" y="292" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#333333">None of the four hybrid components is individually significant after Holm correction. Negative results, fully released.</text>
<text x="66" y="310" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#666666">12 datasets | 30 seeds | 21 configurations | 12,240 real runs | dataset-level analysis unit</text>
</svg>'''
p3 = os.path.join(OUT, 'GraphicalAbstract.svg')
open(p3, 'w', encoding='utf-8').write(svg)
print('saved:', p3)
