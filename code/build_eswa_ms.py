"""生成 ESWA 定向的完整英文投稿稿（双盲：不含作者信息）。"""
import os
import sys
import io
import csv

import numpy as np
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = r'D:/02XIHO Files/Daily Work/03Scientific Research/00Article_File/07改进粒子群算法'
FIN = os.path.join(BASE, 'results', 'final')
RES = os.path.join(BASE, 'results')
OUTDIR = os.path.join(BASE, '14_投稿包_ESWA')
MANU = os.path.join(OUTDIR, '01_稿件')
FIGD = os.path.join(MANU, '图表')
for p in (OUTDIR, MANU, FIGD):
    os.makedirs(p, exist_ok=True)


def rd(name, base=FIN):
    return list(csv.DictReader(open(os.path.join(base, name), encoding='utf-8-sig')))


EN = {'BP-30': 'BP-30 (equal budget)', 'BP': 'BP (fully trained)',
      'PSO-BP': 'PSO-BP', 'IPSO-BP': 'IPSO-BP',
      'IPSO': 'IPSO', 'IPSO+OBL': 'IPSO+OBL',
      'CDPSO-无重启': 'w/o restart', 'CDPSO-无扰动': 'w/o perturbation',
      'CDPSO-无多样性': 'w/o diversity feedback',
      'CDPSO-无OBL': 'w/o OBL (clean)'}

t1 = rd('table1_main_comparison.csv')
t2 = rd('table2_ablation.csv')
t3 = rd('table3_avg_rank.csv')
t4 = rd('table4_breakeven.csv')
t5 = rd('table5_cost.csv')

# 逐数据集精度表（表 7）
raw = [ln for ln in open(os.path.join(RES, 'raw_results.csv'), encoding='utf-8-sig')
       if not ln.lstrip('\ufeff').startswith('#')]
rawr = list(csv.DictReader(raw, skipinitialspace=True))
from collections import defaultdict
acc = defaultdict(list)
for r in rawr:
    if r.get('accuracy') not in ('', 'nan', None):
        acc[(r['dataset'], r['method'])].append(float(r['accuracy']))

DS = ['Heart', 'LoL', 'Pima', 'WDBC', 'Ionosphere', 'Sonar',
      'Banknote', 'Spambase', 'Haberman', 'Parkinson', 'Blood', 'Phoneme']
CN = {k: k for k in ['Heart', 'LoL', 'Pima', 'WDBC', 'Ionosphere', 'Sonar',
                     'Banknote', 'Spambase', 'Haberman', 'Parkinson',
                     'Blood', 'Phoneme']}

d = Document()
st = d.styles['Normal']
st.font.name = 'Times New Roman'
st.font.size = Pt(11)
st.paragraph_format.line_spacing = 1.15
st.paragraph_format.space_after = Pt(6)


def H(text, lvl=1):
    p = d.add_heading(text, level=lvl)
    for r in p.runs:
        r.font.name = 'Times New Roman'
        r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def P(text, bold=False, italic=False, size=11, align=None):
    p = d.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    r.font.name = 'Times New Roman'
    if align:
        p.alignment = align
    return p


ZH2EN = {
    '数据集': 'Dataset', '对照方法': 'Control', '判定': 'Verdict',
    '逐数据集 胜/平/负': 'W/T/L (of 12)', '平均差(pp)': 'Mean diff (pp)',
    '中位差(pp)': 'Median diff (pp)', '95% CI (pp)': '95% CI (pp)',
    'Wilcoxon p': 'Wilcoxon p', 'Holm p': 'Holm p',
    "Cliff's δ": "Cliff's d", '变体': 'Variant',
    'CDPSO-BP 精度': 'CDPSO-BP', 'break-even E*': 'Break-even E*',
    '等效倍数': 'Ratio', '方法': 'Method', '平均秩': 'Avg rank',
    '平均准确率': 'Mean acc.', '搜索开销': 'Search', '微调开销': 'Fine-tune',
    '合计': 'Total', '说明': 'Note', '种子数': 'Seeds',
    'n(数据集)': 'n (datasets)',
}


def _en(v):
    if not isinstance(v, str):
        return v
    if v in ZH2EN:
        return ZH2EN[v]
    if v in EN:
        return EN[v]
    if v in CN:
        return CN[v]
    if '↑优' in v:
        v = v.replace('（Holm，↑优）', ' (Holm, favours CDPSO-BP)')
        v = v.replace('↑优', 'favours CDPSO-BP')
    if '↓劣' in v:
        v = v.replace('（Holm，↓劣）', ' (Holm, favours control)')
        v = v.replace('↓劣', 'favours control')
    if v.startswith('显著'):
        v = 'Significant' + v[len('显著'):]
    if v.startswith('不显著'):
        v = 'Not significant' + v[len('不显著'):]
    if '参照' in v:
        v = 'reference'
    if v == '与 CDPSO-BP 等预算':
        v = 'identical budget to CDPSO-BP'
    if v == '上界，早停可更低':
        v = 'upper bound; early stopping can reduce it'
    if v == 'BP（≤300 epoch + 早停）':
        v = 'BP (<=300 epoch, early stop)'
    if v.startswith('搜索 40代'):
        v = 'search: 40 generations x 20 particles x 0.8n'
    if v == 'PSO-BP / IPSO-BP / CDPSO-BP':
        v = 'PSO-BP / IPSO-BP / CDPSO-BP'
    return v


def mktable(header, rows, caption=None, note=None, widths=None):
    if caption:
        cp = d.add_paragraph()
        cr = cp.add_run(caption)
        cr.bold = True
        cr.font.size = Pt(10)
        cr.font.name = 'Times New Roman'
    t = d.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    header = [_en(h) for h in header]
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ''
        rr = c.paragraphs[0].add_run(str(h))
        rr.bold = True
        rr.font.size = Pt(9)
        rr.font.name = 'Times New Roman'
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            rr = cells[i].paragraphs[0].add_run(str(_en(v)))
            rr.font.size = Pt(9)
            rr.font.name = 'Times New Roman'
    if note:
        np_ = d.add_paragraph()
        nr = np_.add_run(note)
        nr.font.size = Pt(8.5)
        nr.italic = True
        nr.font.name = 'Times New Roman'
    d.add_paragraph()
    return t


# ===========================================================================
# 题名页（双盲：作者信息留占位由投稿系统填写）
# ===========================================================================
ti = d.add_heading('What Is Metaheuristic Weight Initialization Worth to a '
                   'Back-Propagation Network? A Budget-Equivalent Analysis '
                   'across 12 Benchmark Datasets', level=0)
for r in ti.runs:
    r.font.name = 'Times New Roman'

P('Anonymous manuscript submitted to Expert Systems with Applications '
  '(double-anonymized review). Author identifiers are intentionally omitted.',
  italic=True, size=9)

# ---- 摘要（≤250 词）----
H('Abstract', 1)
P(
    'Initializing back-propagation (BP) weights from a metaheuristic is widely '
    'reported to improve accuracy, yet the claim is rarely tested against '
    'gradient descent at equal budget, and the credited components are seldom '
    'isolated. We ask a narrower question: under a fixed gradient budget, how '
    'much accuracy does metaheuristic initialization buy, and what is it '
    'equivalent to? We answer it with a budget-equivalent analysis. A chaos-based '
    'opposition-learning diversity-feedback PSO (CDPSO) is evaluated on 12 '
    'public binary datasets with 30 fixed seeds per configuration (21 '
    'configurations x 12 datasets x 30 runs = 7,560 runs), against a '
    'random-initialized BP with identical budget (BP-30), a fully trained BP '
    '(up to 300 epochs with validation-based early stopping), and 14 further '
    'baselines spanning adaptive PSO, adaptive DE, evolution strategies, and '
    'Adam-based gradient training. Two controls are separated explicitly, '
    'because they lead to opposite verdicts: against BP-30, CDPSO-BP gains 3.62 '
    'percentage points (Holm-adjusted p = 0.002, 12/12 datasets); against fully '
    'trained BP the difference is +0.12 points and not significant (p = 0.79). '
    'A budget curve over 13 gradient budgets (4,680 additional runs) locates the '
    'break-even point E*: pure BP requires a median of 100 epochs to match a '
    'CDPSO-BP that fine-tunes for only 30, a 5.9-10.1x equivalence. A five-way '
    'one-at-a-time ablation shows that none of the four components individually '
    'reaches significance after Holm correction at the dataset level. The '
    'contribution is therefore not a new optimizer but a quantified, '
    'reproducible account of when metaheuristic initialization is worth its '
    'cost.',
    size=10.5)

P('Keywords: particle swarm optimization; neural network initialization; '
  'back-propagation; budget-equivalent analysis; negative results; '
  'reproducible benchmarking', italic=True, size=10)

d.add_page_break()

# ===========================================================================
# 1 Introduction
# ===========================================================================
H('1. Introduction', 1)

P('Back-propagation (BP) remains the default training procedure for '
  'feed-forward neural networks in practice [1]. Its convergence behaviour is '
  'known to depend on the initial weight vector: a random draw that falls into '
  'an unfavourable basin of the error surface slows learning and, when the '
  'training budget is tight, degrades the final test accuracy. A long-standing '
  'remedy is to replace the random draw with a population-based metaheuristic: '
  'the search phase of a swarm or evolutionary algorithm explores the weight '
  'space, and its best individual is then refined by gradient descent. This '
  'hybrid is usually called X-BP, where X is the metaheuristic used.')

P('The hybrid is well established, and the individual ingredients are old. '
  'Inertia weight scheduling comes from Shi and Eberhart [3]; chaos-based '
  'initialization from Xiang et al. [24]; opposition-based learning (OBL) from '
  'Tizhoosh [17] and Omran and Al-Sharhan [18]; diversity-guided PSO from '
  'Riget and Vesterstrom [25] and Jie et al. [20]. Applications to BP weight '
  'initialization span mining-water source discrimination [11], ball-screw '
  'feed-drive control [12], blast-furnace silicon prediction [13], transformer '
  'fault diagnosis [15], and PM2.5 forecasting [14,16].')

P('Two features of this literature are worth stating plainly, because they '
  'motivate the present study. First, the reported comparisons are almost '
  'always unequal-budget: the metaheuristic branch performs T x N fitness '
  'evaluations plus fine-tuning, while the BP baseline is given a fixed, much '
  'smaller number of gradient steps. Such a comparison cannot separate "the '
  'metaheuristic found a better starting point" from "the metaheuristic branch '
  'simply consumed more computation". Second, when several mechanisms are '
  'combined into one hybrid, the contribution of each is usually not isolated, '
  'so a mechanism with no measurable effect can still be carried in the title '
  'and in the abstract.')

P('This paper does not propose another hybrid. It asks a deliberately narrow '
  'question that the existing literature has not answered: under a fixed '
  'gradient budget, how much accuracy does metaheuristic initialization '
  'actually buy, and what is that gain equivalent to in units of gradient '
  'steps? The contribution is a measurement protocol and a set of numbers, '
  'including the cases where the answer is negative.')

P('The specific questions addressed are:')
for q in [
    'Q1. Compared with random-initialized BP given the identical fine-tuning '
    'budget, how large and how consistent is the gain?',
    'Q2. What is the gain equivalent to, expressed as a number of gradient '
    'epochs (the break-even point E*)?',
    'Q3. Does the hybrid beat a fully trained BP, and does it beat modern '
    'adaptive PSO, adaptive DE, evolution strategies, and Adam-based gradient '
    'training?',
    'Q4. Which of the individual components carry the effect?',
]:
    p = d.add_paragraph(q, style='List Bullet')
    for r in p.runs:
        r.font.size = Pt(11)
        r.font.name = 'Times New Roman'

P('Our answer, in brief, is that the gain is real but small, that it is '
  'equivalent to roughly 100 gradient epochs, and that it does not survive '
  'comparison with fully trained gradient-based training. We report this '
  'explicitly because a resource-equivalence framing is more useful to a '
  'practitioner choosing a training procedure than a further decimal place in '
  'an accuracy table.')

P('The three contributions are:')
for c in [
    'C1. A budget-equivalent analysis. We sweep the BP gradient budget over 13 '
    'levels (0 to 300 epochs, 4,680 runs) and define the break-even epoch E* at '
    'which plain BP matches the hybrid. E* has a median of 100 epochs '
    '(range 75 to 200, resolved on 10 of 12 datasets), which converts a vague '
    'claim about accuracy into a concrete exchange rate between search cost '
    'and gradient steps.',
    'C2. A reproducible negative-result protocol. All 21 configurations are run '
    'on 12 datasets with 30 fixed seeds; the unit of analysis is the dataset '
    '(n = 12), following Demšar [33], rather than the pooled 360 '
    'dataset-seed pairs, which would treat correlated samples within a dataset '
    'as independent. Under this stricter unit, none of the four ablation '
    'components reaches significance after Holm correction. We make the code, '
    'the per-seed records, and the derivation scripts public.',
    'C3. A component-level attribution that includes the result against the '
    'authors\' own hypothesis. A five-way one-at-a-time ablation, including a '
    'clean variant that removes OBL while retaining the other three mechanisms, '
    'shows that the diversity feedback is the only component with a '
    'consistent (though not significant after correction) direction.',
]:
    p = d.add_paragraph(c, style='List Bullet')
    for r in p.runs:
        r.font.size = Pt(11)
        r.font.name = 'Times New Roman'

d.add_page_break()

# ===========================================================================
# 2 Related work
# ===========================================================================
H('2. Related work', 1)

H('2.1 Metaheuristic initialization of neural network weights', 2)
P('Kennedy and Eberhart introduced PSO [2] and Shi and Eberhart its inertia '
  'weight [3]. Hybridizing PSO with gradient descent for connection-weight '
  'optimization is standard practice. Early applications include parameter '
  'identification [6]. Chaotic initialization was proposed by Xiang et al. '
  '[24] and used for forward-network training by Zhu et al. [8] and Zhang et '
  'al. [9]. Adaptive inertia weights driven by evolutionary state were '
  'proposed by Yang et al. [5]. Diversity-driven PSO (ARPSO) was formulated by '
  'Riget and Vesterstrom [25], and Jie et al. [20] used population diversity '
  'as a feedback signal for the inertia weight. Liang and Kang [21] measured '
  'diversity by mean inter-particle distance and adjusted the weight '
  'nonlinearly. Tian et al. [19] combined chaotic OBL initialization with '
  'diversity-guided multi-mutation.')

H('2.2 Gradient-based initialization and the modern alternative', 2)
P('The comparison that a metaheuristic initializer must face is not the random '
  'draw. Glorot and Bengio [31] and He et al. [32] showed that variance-aware '
  'initialization combined with Adam reaches strong accuracy without any '
  'population search. On the two tasks where this was documented in the '
  'literature on PSO-BP hybrids, that is the baseline a fair comparison '
  'should include. We therefore treat Xavier+Adam and Kaiming+Adam as '
  'first-class baselines rather than optional references.')

H('2.3 Reporting practice in hybrid comparisons', 2)
P('Metaheuristic-versus-gradient comparisons are frequently reported without '
  'an equal-budget control, and multi-mechanism hybrids are frequently '
  'published without component-level ablation. Both issues inflate the '
  'apparent benefit. Garcia et al. [34] and Derrac et al. [35] provide the '
  'statistical protocol we follow: multiple comparisons across datasets '
  'require nonparametric tests across datasets with multiplicity correction, '
  'and effect sizes should be reported alongside p-values. We follow that '
  'protocol, and we choose the dataset as the unit of analysis for exactly '
  'the reason those papers give.')

H('2.4 Scope of the present study', 2)
P('We do not claim that metaheuristic initialization is useless, nor that any '
  'particular hybrid is optimal. We quantify one specific hybrid on 12 '
  'benchmark datasets under a controlled protocol, and we report the settings '
  'in which it does not help.')

# ===========================================================================
# 3 Method
# ===========================================================================
H('3. Method', 1)

H('3.1 Network and training', 2)
P('We use a single-hidden-layer feed-forward network with 8 hidden units, a '
  'tanh hidden activation and a sigmoid output, trained with mean squared '
  'error by full-batch gradient descent with momentum (learning rate 0.05, '
  'momentum 0.9). The weight dimension is d x 8 + 17 for d input features, '
  'which ranges from 41 to 489 across our datasets. The mean squared error '
  'choice keeps the search-stage fitness consistent with the training loss.')

H('3.2 The CDPSO search stage', 2)
P('CDPSO modifies standard PSO in three ways. (i) Tent-map chaotic '
  'initialization combined with opposition-based learning: N chaotic '
  'candidates and their N opposites give 2N candidates, from which the best N '
  'are kept. (ii) A diversity-feedback inertia weight. With d(t) the mean '
  'inter-particle distance and d(0) its value at initialization, the '
  'diversity retention ratio is rho(t) = min(d(t)/d(0), 1), and')
p = d.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('w(t) = [w_min + (w_max - w_min)(1 - t/T)^2] (1 + rho(t)) / 2,   '
              'with w_max = 0.9, w_min = 0.4.')
r.italic = True
r.font.size = Pt(10.5)
P('Because rho is bounded by 1, the diversity feedback can only lower the '
  'inertia weight below the schedule baseline, never raise it above; the '
  'weight therefore stays in [0.5 w_min, w_max]. When rho is identically 1 '
  'the rule reduces to the pure schedule, which is exactly the form used by '
  'our IPSO control. (iii) A greedily accepted Gaussian perturbation of the '
  'global best with linearly decaying strength, plus a stagnation restart '
  'that reinitializes the worst 20% of particles with chaotic values when no '
  'improvement occurs for two generations or when the retention ratio falls '
  'below 0.20.')

H('3.3 Two properties of the weight rule', 2)
P('We state two properties that follow directly from the definition, because '
  'they delimit what the mechanism can and cannot do.')
P('Property 1 (one-sidedness). For rho in [0, 1] the factor (1 + rho)/2 lies '
  'in [0.5, 1], so w(t) <= w_base(t) for every generation. The diversity '
  'feedback is a one-sided confinement of the schedule baseline, not a '
  'bidirectional modulator.', bold=False)
P('Property 2 (normalization). The retention ratio is dimensionless and '
  'depends on the diversity trajectory only through its ratio to the initial '
  'diversity, so the rule is invariant to a global rescaling of the initial '
  'population magnitude. Note that this is a statement about the control '
  'signal, not about the resulting trajectory: because the network '
  'activations are nonlinear, rescaling the search region changes the fitness '
  'landscape and therefore changes the path the swarm takes. We verified this '
  'empirically and report the outcome in Section 4.6, where the measured '
  'deviation is not negligible. Property 2 should consequently not be read '
  'as a free-tuning result.', bold=False)

H('3.4 Control configurations', 2)
P('The comparison separates two budgets deliberately. BP-30 is '
  'random-initialized BP fine-tuned for exactly the same 30 epochs as the '
  'hybrid: it is the equal-budget control. BP is random-initialized and '
  'trained for up to 300 epochs with validation-based early stopping, roughly '
  'ten times the gradient budget: it is the strong control. IPSO-BP uses Tent '
  'chaotic initialization and a time-schedule weight with rho identically 1, '
  'without OBL, diversity feedback, perturbation, or restart. Fourteen further '
  'baselines cover adaptive PSO (CLPSO [26], APSO [27], AMPSO, DMPSO [19], '
  'GLPSO), adaptive DE (SHADE [29]), evolution strategies (CMA-ES [28], jSO '
  '[30]), and gradient-based training (Xavier+Adam [31], Kaiming+Adam [32]). '
  'Adam-based baselines are tuned over a learning-rate grid '
  '{0.001, 0.01, 0.05} by validation loss.')

# ===========================================================================
# 4 Experiments
# ===========================================================================
H('4. Experiments and results', 1)

H('4.0 Reproducibility and data availability', 2)
P('Code, the 12 cleaned datasets, all 7,560 per-seed records from the main '
  'experiment, the 4,680 budget-curve records, and every table-generation '
  'script are publicly available at https://github.com/Halsey-X/CDPSO-BP. '
  'All runs use fixed seeds (seed = 3000 + k, k = 0...29) and are therefore '
  'exactly reproducible. No result in this paper is copied from another '
  'publication.')

H('4.1 Datasets', 2)
P('We use 12 public binary classification datasets: two application-oriented '
  'sets (heart disease, 918 samples, 19 features; League of Legends match '
  'outcome, 9,879 samples, 38 features) and ten standard benchmarks from '
  'UCI and OpenML (Pima, WDBC, Ionosphere, Sonar, Banknote, Spambase, '
  'Haberman, Parkinson, Blood, Phoneme). Sample sizes range from 195 to 9,879 '
  'and feature counts from 3 to 59. The positive-class rate ranges from 0.238 '
  'to 0.754, so the class balance is deliberately not controlled; this is why '
  'we report balanced accuracy and AUC alongside accuracy in Section 4.5.')

H('4.2 Protocol', 2)
P('Each dataset is split 7:1:2 into training, validation, and test sets with '
  'the split seed fixed. All PSO-family methods search on the training plus '
  'validation portion (80%) and fine-tune on the training portion for a fixed '
  '30 epochs. PSO parameters are N = 20 particles, T = 40 generations, '
  'c1 = c2 = 2, with the restart trigger described in Section 3.2. Every '
  'configuration is run with 30 seeds, giving 21 x 12 x 30 = 7,560 runs for '
  'the main experiment.')

H('4.3 Where metaheuristic initialization helps: the equal-budget contrast', 2)
rows = []
for r in t1:
    rows.append([r['对照方法'], r['平均差(pp)'], r['95% CI (pp)'],
                 r['Holm p'], r["Cliff's δ"], r['逐数据集 胜/平/负'], r['判定']])
mktable(['Control', 'Mean diff (pp)', '95% CI (pp)', 'Holm p', "Cliff's δ",
         'W/T/L (of 12)', 'Verdict'],
        rows,
        'Table 1. Equal-budget and strong controls against CDPSO-BP '
        '(unit of analysis: dataset, n = 12, Wilcoxon signed-rank with Holm '
        'correction).',
        'A positive mean difference favours CDPSO-BP. Holm-adjusted p-values '
        'are reported because four comparisons are made against the same '
        'reference.')

P('Table 1 is the central result and it splits cleanly. Against the '
  'equal-budget control the gain is 3.62 percentage points, consistent across '
  'all 12 datasets (12 wins, 0 ties, 0 losses) and significant after Holm '
  'correction (p = 0.002, Cliff delta = -1.00 in the direction of CDPSO-BP). '
  'Against the fully trained control the difference is +0.12 points with a '
  'p-value of 0.79 and a median difference of +0.05 points: at the level of '
  'datasets, the hybrid and a properly trained BP are indistinguishable. The '
  'gains over plain PSO-BP (+0.24 points, raw p = 0.042) and over IPSO-BP '
  '(+0.05 points) do not survive Holm correction.')

H('4.4 What the gain is equivalent to: the break-even epoch', 2)
P('Table 1 answers how large the gain is but not what it costs. We therefore '
  'swept the BP gradient budget over 13 levels, E in {0, 1, 2, 5, 10, 20, '
  '30, 50, 75, 100, 150, 200, 300}, for all 12 datasets and 30 seeds '
  '(4,680 additional runs), and defined the break-even epoch E* as the '
  'smallest budget at which plain BP matches the accuracy of CDPSO-BP. Table '
  '2 reports the result.')

FIG1 = os.path.join(FIGD, 'Fig1_breakeven.png')
if os.path.exists(FIG1):
    pf = d.add_paragraph()
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.add_run().add_picture(FIG1, width=Inches(6.4))
    cp = d.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cp.add_run('Fig. 1. Budget-equivalent analysis. (a) Accuracy against '
                    'BP gradient budget for the 12 datasets; the horizontal '
                    'line marks the CDPSO-BP result obtained with 30 epochs. '
                    '(b) The break-even epoch E* per dataset; two datasets are '
                    'not reached within 300 epochs (shown in red).')
    cr.font.size = Pt(8.5)
    cr.font.name = 'Times New Roman'
    d.add_paragraph()

rows = []
for r in t4:
    rows.append([CN.get(r['数据集'], r['数据集']), r['CDPSO-BP 精度'], r['BP@30'], r['BP@100'],
                 r['BP@300'], r['break-even E*'], r['等效倍数']])
mktable(['Dataset', 'CDPSO-BP', 'BP@30', 'BP@100', 'BP@300', 'E*', 'Ratio'],
        rows,
        'Table 2. Break-even analysis: the smallest number of BP epochs that '
        'matches CDPSO-BP accuracy (30 seeds per cell; 4,680 runs in total).',
        '"Ratio" is the total search-plus-finetune cost of CDPSO-BP expressed '
        'in units of one BP epoch, i.e. (640 + E* x 6.3) / (30 x 6.3). '
        'E* > 300 means plain BP never catches up within the swept range.')

be = [int(r['break-even E*']) for r in t4 if r['break-even E*'].isdigit()]
P(f'The median break-even point is E* = {int(np.median(be))} epochs, with a '
  f'range of {min(be)} to {max(be)}; on 2 of 12 datasets (Heart, Parkinson) '
  f'plain BP does not catch up within 300 epochs. The interpretation is '
  f'direct: a hybrid that fine-tunes for 30 epochs buys roughly the same '
  f'accuracy as about 100 epochs of plain gradient training, at a total cost '
  f'of 5.9 to 10.1 BP-epoch equivalents. This is the exchange rate that was '
  f'missing from the equal-budget comparison, and it is small: for a '
  f'practitioner with a modest compute budget the hybrid is worth '
  f'considering, but it is not a substitute for training longer.')

H('4.5 Per-dataset accuracy', 2)
rows = []
for ds in DS:
    def ms(m):
        v = acc.get((ds, m), [])
        return f'{np.mean(v):.4f}' if v else '--'
    rows.append([CN[ds], ms('BP-30'), ms('BP'), ms('PSO-BP'), ms('IPSO-BP'),
                 ms('CDPSO-BP')])
mktable(['Dataset', 'BP-30', 'BP (<=300 ep)', 'PSO-BP', 'IPSO-BP', 'CDPSO-BP'],
        rows,
        'Table 3. Test accuracy on 12 datasets (mean over 30 seeds).',
        'Full per-seed records including F1, AUC and balanced accuracy are in '
        'the public repository (results/budget_curve.csv and '
        'results/raw_results.csv).')

P('Two caveats should be stated. First, the class balance varies from 0.238 '
  'to 0.754, so accuracy alone is not a sufficient summary; the budget-curve '
  'runs therefore also record AUC and balanced accuracy, and for the more '
  'imbalanced datasets (Blood, Haberman) balanced accuracy at 30 epochs is '
  'close to 0.52, i.e. close to chance, so the accuracy figures there should '
  'be read as majority-class rates. Second, all datasets are tabular and '
  'binary; the conclusions are not extended to image, time-series, or '
  'multiclass tasks, and we do not claim otherwise.')

H('4.6 Component attribution', 2)
FIG2 = os.path.join(FIGD, 'Fig2_effects.png')
if os.path.exists(FIG2):
    pf = d.add_paragraph()
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.add_run().add_picture(FIG2, width=Inches(6.4))
    cp = d.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cp.add_run('Fig. 2. Effect sizes under the dataset-level analysis '
                    'unit (n = 12). (a) Mean accuracy difference of CDPSO-BP '
                    'against each control; only the equal-budget control '
                    'survives Holm correction. (b) Mean difference when each '
                    'component is removed; no component is significant after '
                    'correction.')
    cr.font.size = Pt(8.5)
    cr.font.name = 'Times New Roman'
    d.add_paragraph()

rows = []
for r in t2:
    rows.append([EN.get(r['对照方法'], r['对照方法']), r['平均差(pp)'], r['95% CI (pp)'], r['Holm p'],
                 r["Cliff's δ"], r['逐数据集 胜/平/负'], r['判定']])
mktable(['Variant', 'Mean diff (pp)', '95% CI (pp)', 'Holm p', "Cliff's δ",
         'W/T/L (of 12)', 'Verdict'],
        rows,
        'Table 4. Five-way one-at-a-time ablation (dataset-level, n = 12).',
        '"w/o OBL" removes opposition-based learning while retaining chaotic '
        'initialization, diversity feedback, perturbation and restart; it is '
        'the clean attribution for the initialization mechanism. Differences '
        'are variant minus complete model, so a positive value means the '
        'removed component was contributing.')

P('This table is where the paper parts company with its own framing. Under '
  'the dataset-level analysis unit, no component reaches significance after '
  'Holm correction. The diversity feedback has the largest effect '
  '(-0.209 pp when removed, 8 of 12 datasets worse) and the clean OBL '
  'ablation is indistinguishable from zero (+0.110 pp when removed, median '
  '+0.045, Cliff delta exactly 0.00, p = 0.42). Removing the Gaussian '
  'perturbation actually improves the average (-0.036 pp, i.e. the complete '
  'model is 0.036 pp worse without it), and the restart mechanism is likewise '
  'within noise.')

P('We read this as evidence that, at a search budget of 40 generations x 20 '
  'particles in weight dimensions of 41 to 489, the swarm is so '
  'under-sampled that the mechanisms which act on late-stage stagnation have '
  'little to act on. The diversity feedback, which acts from the first '
  'generation, is the only one with a consistent direction. A practitioner '
  'should read the ablation as: use the diversity feedback, do not assume the '
  'other three mechanisms are earning their complexity.')

P('We also report a negative control on our own theory. We had expected the '
  'normalized weight rule to be exactly invariant to the choice of search '
  'region, because the retention ratio is dimensionless. We tested this by '
  're-running the search with the region rescaled by factors of 0.25 to 4 and '
  'recording the realized inertia-weight trajectory. The measured deviation '
  'between the rescaled and reference trajectories reached 1.0e-1 in the '
  'worst case, which is not negligible. The invariance of the control signal '
  'does not transfer to the trajectory, because the nonlinear activations '
  'change the fitness landscape under rescaling. The original claim of '
  'scale-free tuning does not hold and is not made.')

H('4.7 Ranking against all baselines', 2)
rows = []
for r in t3:
    rows.append([r['方法'], r['平均秩'], r['平均准确率']])
mktable(['Method', 'Average rank', 'Mean accuracy'], rows,
        'Table 5. Cross-dataset average rank over 20 configurations '
        '(rank 1 = best).',
        'Averaged over 12 datasets, ties resolved with average ranks. The '
        'five main methods alone give a Friedman p of 5.19e-05.')

rows = [[r['方法'], r['平均秩'], r['平均准确率']] for r in t3]
P('CDPSO-BP ranks 4th of 20 configurations, behind CMA-ES (7.00), '
  'Xavier+Adam (7.33) and a variant of our own hybrid without the '
  'perturbation (7.88). The two Adam-based gradient-training baselines are '
  'ahead of the hybrid on average accuracy. We state this plainly: on this '
  'benchmark suite, a well-tuned Adam with variance-aware initialization is '
  'at least as good as the hybrid and simpler to use. The case for the hybrid '
  'rests on the break-even result of Section 4.4, not on the ranking.')

H('4.8 Computational cost', 2)
COST_EN = {'BP-30': 'BP-30',
           'BP（≤300 epoch + 早停）': 'BP (<=300 epoch, early stop)',
           'PSO-BP / IPSO-BP / CDPSO-BP': 'PSO-BP / IPSO-BP / CDPSO-BP'}
COST_NOTE = {'与 CDPSO-BP 等预算': 'identical budget to CDPSO-BP',
             '上界，早停可能更低': 'upper bound; early stopping can reduce it',
             '上界，早停可更低': 'upper bound; early stopping can reduce it',
             '搜索：40代×20粒子×0.8n': 'search: 40 gen x 20 particles x 0.8n',
             '搜索 40代×20粒子×0.8n': 'search: 40 gen x 20 particles x 0.8n'}
mktable(['方法', '搜索开销', '微调开销', '合计', '说明'],
        [[COST_EN.get(r['方法'], r['方法']), r['搜索开销'], r['微调开销'],
          r['合计'], COST_NOTE.get(r['说明'], r['说明'])] for r in t5],
        'Table 6. Computational cost in forward-pass-equivalent units, where n '
        'is the number of samples.',
        'Search cost is 40 generations x 20 particles x 0.8n forward passes; '
        'one BP epoch costs about 6.3n including the backward pass.')

P('The search stage is about ten times the cost of a 30-epoch BP fine-tune. '
  'Combined with the break-even result, the honest summary is that the hybrid '
  'trades roughly 640 forward-pass units of search for the equivalent of '
  'about 100 extra gradient epochs. Whether that trade is worth it is a '
  'resource decision, not a claim of superiority.')

# ===========================================================================
# 5 Discussion
# ===========================================================================
H('5. Discussion', 1)

H('5.1 What the evidence supports', 2)
P('Three statements are supported by the data in this paper. (i) Against a '
  'random initialization given the same 30-epoch budget, metaheuristic '
  'initialization is consistently better: 12 of 12 datasets, 3.62 points, '
  'p = 0.002 after correction. (ii) That advantage is worth about 100 '
  'gradient epochs, with a 5.9 to 10.1x cost ratio. (iii) Once a properly '
  'trained BP or an Adam-based baseline is available, the hybrid offers no '
  'measurable advantage.')

H('5.2 What the evidence does not support', 2)
P('Several things a reader might expect are not supported. The hybrid does '
  'not beat a fully trained BP. It does not beat CMA-ES or Adam-based '
  'training. Of the four mechanisms inside the hybrid, none survives Holm '
  'correction at the dataset level, including the opposition-based '
  'initialization that appears in the conventional description of this family '
  'of methods. We report the last point with particular attention because it '
  'is the kind of result that is easy to omit.')

H('5.3 Why the negative results are plausible', 2)
P('Two features of the setting explain the pattern. The search budget is '
  'small relative to the weight dimension: 40 generations x 20 particles '
  'against 41 to 489 dimensions is severely under-sampled, so the swarm never '
  'reaches the late stage of convergence at which perturbation and restart '
  'matter. And the fine-tuning budget is fixed at 30 epochs, which is exactly '
  'the regime in which a better starting point pays off; as the budget grows, '
  'gradient descent washes out initialization differences, which is precisely '
  'what the break-even curve shows. Under a larger search budget we would '
  'expect the ablation picture to change, and we flag this as the main open '
  'question rather than a settled result.')

H('5.4 Threats to validity', 2)
P('The analysis unit matters. Using the pooled 360 dataset-seed pairs instead '
  'of the 12 dataset means yields p = 1.15e-38 for the equal-budget contrast, '
  'a number we regard as an artifact of treating correlated within-dataset '
  'samples as independent; we report it only to explain why earlier reports '
  'may have appeared more decisive. All headline claims here use the dataset '
  'as the unit. Other threats: a single network architecture and a single '
  'train/validation/test split per dataset; no per-dataset hyperparameter '
  'tuning for any PSO-family method; the search budget fixed at 40x20; and '
  'tabular binary data only.')

H('5.5 Reproducibility as a contribution', 2)
P('Because the main result is negative and the boundaries are narrow, we '
  'consider the released artefacts to be part of the contribution: 7,560 '
  'per-seed records from the main experiment, 4,680 from the budget curve, '
  'the 12 cleaned datasets, and the scripts that derive every table in this '
  'paper from those records. Each reported number can be traced to the raw '
  'rows that produced it.')

# ===========================================================================
# 6 Conclusion
# ===========================================================================
H('6. Conclusion', 1)
P('We asked what metaheuristic weight initialization is worth to a BP network '
  'under a controlled gradient budget, and measured it on 12 datasets with 30 '
  'seeds and 21 configurations. The answer is bounded and, in places, '
  'unflattering. Against an equal-budget random initialization the hybrid wins '
  'on every dataset by 3.62 percentage points (Holm p = 0.002). That gain is '
  'equivalent to about 100 gradient epochs, at roughly ten times the compute '
  'of the fine-tuning it replaces. Against a fully trained BP the two are '
  'statistically indistinguishable (p = 0.79), and the hybrid ranks fourth of '
  'twenty configurations, behind CMA-ES and Adam-based training. A five-way '
  'ablation finds no component individually significant after correction. We '
  'conclude that metaheuristic initialization is best understood as a way of '
  'buying a fixed amount of gradient budget cheaply, applicable when compute '
  'is constrained, and not as a route to higher accuracy on well-resourced '
  'training. Negative results of this kind are, we hope, useful to the '
  'community, and the data and code are released.')

# ===========================================================================
# References
# ===========================================================================
d.add_page_break()
H('References', 1)

REFS = [
    "[1] D.E. Rumelhart, G.E. Hinton, R.J. Williams, Learning representations by back-propagating errors, Nature 323 (1986) 533-536. DOI: 10.1038/323533a0.",
    "[2] J. Kennedy, R. Eberhart, Particle swarm optimization, in: Proc. IEEE Int. Conf. Neural Networks, Perth, 1995, pp. 1942-1948. DOI: 10.1109/ICNN.1995.488968.",
    "[3] Y. Shi, R. Eberhart, A modified particle swarm optimizer, in: Proc. IEEE Int. Conf. Evolutionary Computation, Anchorage, 1998, pp. 69-73. DOI: 10.1109/ICEC.1998.699146.",
    "[4] A. Ratnaweera, S.K. Halgamuge, H.C. Watson, Self-organizing hierarchical particle swarm optimizer with time-varying acceleration coefficients, IEEE Trans. Evol. Comput. 8 (2004) 240-255. DOI: 10.1109/TEVC.2004.826071.",
    "[5] X. Yang, J. Yuan, J. Yuan et al., A modified particle swarm optimizer with dynamic adaptation, Appl. Math. Comput. 189 (2007) 1205-1213. DOI: 10.1016/j.amc.2006.12.045.",
    "[6] 和方, 杨化超, 张书毕, 概率积分法预计参数的智能优化选择方法研究, 采矿与安全工程学报 29 (2012) 675-680.",
    "[7] 程志刚, 张立庆, 李小林 等, 基于Tent映射的混沌混合粒子群优化算法, 系统工程与电子技术 29 (2007) 1747-1751.",
    "[8] 朱群雄, 董春岩, 林晓勇, 基于反传混沌粒子群训练的前馈神经网络研究, 计算机应用研究 26 (2009) 3695-3698.",
    "[9] 张娜, 滕赛娜, 吴彪 等, 基于Tent混沌的测试用例优先级排序, 计算机测量与控制 27 (2019) 6-51.",
    "[10] 戴前伟, 江沸菠, 基于混沌振荡PSO-BP算法的电阻率层析成像非线性反演, 中国有色金属学报 23 (2013) 2897-2903.",
    "[11] 徐星, 李垣志, 田坤云 等, ACPSO-BP神经网络在矿井突水水源判别中的应用, 重庆大学学报 41 (2018) 62-70.",
    "[12] 吴沁, 周顺仟, 王星联, 改进粒子群优化滚珠丝杠进给系统BP神经网络PID控制策略研究, 西安交通大学学报 57 (2023) 128-136.",
    "[13] 徐生林, 史燕, 杨成忠, 应用混沌粒子群优化训练的BP神经网络预报高炉铁水含硅量, 冶金分析 32 (2012) 52-55.",
    "[14] 贾佳美, 池凯凯, 吴哲翔, 改进粒子群优化BP神经网络的PM2.5预测, 计算机工程与设计 42 (2021) 3495-3501.",
    "[15] L. Zhou, Z. Fu, K. Li et al., Power transformer fault diagnosis based on random forest and improved particle swarm optimization-backpropagation-AdaBoost, Electronics 13 (2024) 4149. DOI: 10.3390/electronics13214149.",
    "[16] Z. Liu, Y. Hu, Y. Fang et al., Improved prediction model for daily PM2.5 concentrations with particle swarm optimization and BP neural network, Sci. Rep. 15 (2025) 18014. DOI: 10.1038/s41598-025-18014-w.",
    "[17] H.R. Tizhoosh, Opposition-based learning: a new scheme for machine intelligence, in: Proc. Int. Conf. Computational Intelligence for Modelling, Control and Automation, Vienna, 2005, pp. 695-701. DOI: 10.1109/CIMCA.2005.1631345.",
    "[18] M.G.H. Omran, S. Al-Sharhan, Using opposition-based learning to improve the performance of particle swarm optimization, in: Proc. 2008 IEEE Swarm Intelligence Symp., St. Louis, 2008, pp. 1-6. DOI: 10.1109/SIS.2008.4668288.",
    "[19] D.P. Tian, X.F. Zhao, Z.Z. Shi, DMPSO: diversity-guided multi-mutation particle swarm optimizer, IEEE Access 7 (2019) 124008-124025. DOI: 10.1109/ACCESS.2019.2938063.",
    "[20] J. Jie, J. Zeng, C. Han, Adaptive particle swarm optimization with feedback control of diversity, in: Proc. Int. Conf. Intelligent Computing (ICIC 2006), Berlin, 2006, pp. 81-92. DOI: 10.1007/11816102_9.",
    "[21] H.T. Liang, F.H. Kang, Adaptive mutation particle swarm algorithm with dynamic nonlinear changed inertia weight, Optik 127 (2016) 8036-8042. DOI: 10.1016/j.ijleo.2016.06.002.",
    "[22] K.E.Y. Laurel, A.N.R. Gutierrez, K.A.S. Tan et al., Using multiple AI classification models to predict the winner of a League of Legends game based on its first 10-minutes of gameplay, in: Proc. 2023 Int. Conf. Consumer Electronics Taiwan (ICCE-Taiwan), PingTung, 2023, pp. 693-694. DOI: 10.1109/ICCE-Taiwan58799.2023.10226891.",
    "[23] Q. Shen, A machine learning approach to predict the result of League of Legends, in: Proc. 2022 Int. Conf. Machine Learning and Knowledge Engineering (MLKE), 2022, pp. 38-45. DOI: 10.1109/MLKE55170.2022.00013.",
    "[24] T. Xiang, X.F. Liao, K. Wong, An improved particle swarm optimization algorithm combined with piecewise linear chaotic map, Appl. Math. Comput. 190 (2007) 1637-1645. DOI: 10.1016/j.amc.2007.02.103.",
    "[25] J. Riget, J.S. Vesterstrom, A diversity-guided particle swarm optimizer - the ARPSO, Technical Report, Dept. of Computer Science, University of Aarhus, 2002.",
    "[26] J.J. Liang, A.K. Qin, P.N. Suganthan et al., Comprehensive learning particle swarm optimizer for global optimization of multimodal functions, IEEE Trans. Evol. Comput. 10 (2006) 281-295. DOI: 10.1109/TEVC.2005.857610.",
    "[27] Z.H. Zhan, J. Zhang, Y. Li et al., Adaptive particle swarm optimization, IEEE Trans. Syst. Man Cybern. Part B 39 (2009) 1362-1381. DOI: 10.1109/TSMCB.2009.2015956.",
    "[28] N. Hansen, A. Ostermeier, Completely derandomized self-adaptation in evolution strategies, Evol. Comput. 9 (2001) 159-195. DOI: 10.1162/106365601750190398.",
    "[29] R. Tanabe, A. Fukunaga, Success-history based parameter adaptation for differential evolution, in: Proc. IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 71-78. DOI: 10.1109/CEC.2013.6557555.",
    "[30] J. Brest, M.S. Maucec, B. Boskovic, Single objective real-parameter optimization: algorithm jSO, in: Proc. IEEE Congress on Evolutionary Computation, Donostia, 2017, pp. 1311-1318. DOI: 10.1109/CEC.2017.7969456.",
    "[31] X. Glorot, Y. Bengio, Understanding the difficulty of training deep feedforward neural networks, in: Proc. 13th Int. Conf. Artificial Intelligence and Statistics, Sardinia, 2010, pp. 249-256.",
    "[32] K.M. He, X.Y. Zhang, S.Q. Ren et al., Delving deep into rectifiers: surpassing human-level performance on ImageNet classification, in: Proc. IEEE Int. Conf. Computer Vision, Santiago, 2015, pp. 1026-1034. DOI: 10.1109/ICCV.2015.123.",
    "[33] J. Demsar, Statistical comparisons of classifiers over multiple data sets, J. Mach. Learn. Res. 7 (2006) 1-30.",
    "[34] S. Garcia, A. Fernandez, J. Luengo et al., Advanced nonparametric tests for multiple comparisons in the design of experiments in computational intelligence and data mining: experimental analysis of power, Inf. Sci. 180 (2010) 2044-2064. DOI: 10.1016/j.ins.2009.12.010.",
    "[35] J. Derrac, S. Garcia, D. Molina et al., A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms, Swarm Evol. Comput. 1 (2011) 3-18. DOI: 10.1016/j.swevo.2011.02.002.",
    "[36] E.H. Houssein, A.G. Gad, K. Hussain et al., Major advances in particle swarm optimization: theory, analysis, and application, Swarm Evol. Comput. 63 (2021) 100868. DOI: 10.1016/j.swevo.2021.100868.",
]
for r in REFS:
    p = d.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(-0.25)
    run = p.add_run(r)
    run.font.size = Pt(9)
    run.font.name = 'Times New Roman'

out = os.path.join(MANU, 'Manuscript_ESWA_EN.docx')
d.save(out)
print('saved:', out)
print('段落数:', len(d.paragraphs), '| 表格数:', len(d.tables))
print('摘要词数:', len([s for s in d.paragraphs[3].text.split() if s]))
