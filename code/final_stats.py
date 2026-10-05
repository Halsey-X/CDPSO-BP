"""
最终统计脚本（面向 ESWA 投稿）：
  - 分析单位改为数据集级（n=12），pooled 结果降为附录
  - 纳入干净消融 CDPSO w/o OBL
  - 产出 break-even 分析表
输出到 results/final/
"""
import os
import sys
import io
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')
OUT = os.path.join(RES, 'final')
os.makedirs(OUT, exist_ok=True)


def read_raw():
    path = os.path.join(RES, 'raw_results.csv')
    lines = [ln for ln in open(path, encoding='utf-8-sig')
             if not ln.lstrip('\ufeff').startswith('#')]
    return list(csv.DictReader(lines, skipinitialspace=True))


def read_csv(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def holm(pvals):
    """Holm–Bonferroni 校正，返回校正后 p 值（保持原顺序）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        run = max(run, val)
        adj[idx] = min(1.0, run)
    return adj


def dataset_level_table(data, ref, targets, label, datasets):
    """数据集级配对检验：每个数据集先算 30 种子均值，再做 n=12 的 Wilcoxon。

    这是 Demšar (2006) 推荐的分析单位，避免把数据集内相关样本当独立。
    """
    header = ['对照方法', 'n(数据集)', '平均差(pp)', '中位差(pp)', '95% CI (pp)',
              'Wilcoxon p', 'Holm p', "Cliff's δ", '逐数据集 胜/平/负', '判定']
    rows, ps, deltas, wins = [], [], [], []
    for m in targets:
        diffs, w, n = [], 0, 0
        for ds in datasets:
            if (ds, ref) not in data or (ds, m) not in data:
                continue
            a = np.array(data[(ds, ref)], float)
            b = np.array(data[(ds, m)], float)
            d = (b.mean() - a.mean()) * 100
            diffs.append(d)
            n += 1
            if d > 1e-9:
                w += 1
            elif d < -1e-9:
                w -= 1
        if n == 0 or not diffs:
            continue
        d = np.array(diffs)
        try:
            p = wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-12) else 1.0
        except ValueError:
            p = 1.0
        ps.append(p)
        deltas.append(d)
        wins.append((max(w, 0), n - abs(w), max(-w, 0)))
        rows.append([m, n, f'{d.mean():+.3f}', f'{np.median(d):+.3f}', '',
                     f'{p:.4g}', '', '', '', ''])
    if ps:
        adj = holm(np.array(ps, dtype=float))
        for i, r in enumerate(rows):
            dd = deltas[i]
            k = len(dd)
            se = dd.std(ddof=1) / np.sqrt(k) if k > 1 else 0.0
            r[4] = f'±{1.96 * se:.3f}'
            r[6] = f'{adj[i]:.4g}'
            r[7] = f'{cliffs_delta(dd, np.zeros_like(dd)):+.3f}'
            ww, tt, ll = wins[i]
            r[8] = f'{ww}/{tt}/{ll}'
            sig = '显著' if adj[i] < 0.05 else '不显著'
            arrow = '↑优' if dd.mean() > 0 else ('↓劣' if dd.mean() < 0 else '≈')
            r[9] = f'{sig}（Holm，{arrow}）'
    with open(os.path.join(OUT, f'{label}.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return rows


def main():
    raw = read_raw()
    data = defaultdict(list)
    for r in raw:
        a = r.get('accuracy', '')
        if a in ('', 'nan', None):
            continue
        data[(r['dataset'], r['method'])].append(float(a))

    datasets = sorted(set(ds for ds, _ in data))
    print(f'载入 {len(raw)} 条主实验记录，{len(datasets)} 个数据集')
    print(f'方法: {sorted(set(m for _, m in data))}')

    MAIN = ['BP-30', 'BP', 'PSO-BP', 'IPSO-BP', 'CDPSO-BP']
    ABL = ['IPSO', 'IPSO+OBL', 'CDPSO-无重启', 'CDPSO-无扰动',
           'CDPSO-无多样性', 'CDPSO(完整)']

    # ---- 表1：主方法对比（数据集级）----
    print('\n[表1] 主方法对比（数据集级 n=12，Holm 校正）')
    r1 = dataset_level_table(data, 'CDPSO-BP', [m for m in MAIN if m != 'CDPSO-BP'],
                             'table1_main_comparison', datasets)
    for r in r1:
        print('   ', ' | '.join(str(x) for x in r))

    # ---- 表2：消融（含新的 w/o OBL）----
    print('\n[表2] 消融（数据集级 n=12）')
    abl = [m for m in ABL if m != 'CDPSO(完整)']
    # 加入干净消融
    noobl = read_csv('ablation_no_obl.csv')
    for r in noobl:
        data[(r['dataset'], 'CDPSO-无OBL')].append(float(r['accuracy']))
    if any(('CDPSO-无OBL', 'CDPSO(完整)') or (d, 'CDPSO(完整)') in data
           for d in datasets):
        abl.append('CDPSO-无OBL')
    r2 = dataset_level_table(data, 'CDPSO(完整)', abl, 'table2_ablation', datasets)
    for r in r2:
        print('   ', ' | '.join(str(x) for x in r))

    # ---- 表3：跨数据集平均秩 ----
    print('\n[表3] 跨数据集平均秩')
    DUP = ('CDPSO(完整)', 'IPSO')
    allm = [m for m in MAIN + abl + [
        'CMA-ES', 'Xavier+Adam', 'Kaiming+Adam', 'CLPSO', 'APSO', 'AMPSO',
        'DMPSO', 'GLPSO', 'SHADE', 'jSO'] if m not in DUP
        and any((d, m) in data for d in datasets)]
    ranks = defaultdict(list)
    for ds in datasets:
        accs = [(m, np.mean(data[(ds, m)])) for m in allm if (ds, m) in data]
        # 精度越高排名越靠前（rank 1 = 最优）
        accs.sort(key=lambda x: -x[1])
        rk = rankdata([-a for _, a in accs], method='average')
        for (m, _), v in zip(accs, rk):
            ranks[m].append(v)
    t3 = sorted([(m, float(np.mean(v)), float(np.mean(
        [np.mean(data[(d, m)]) for d in datasets if (d, m) in data])))
        for m, v in ranks.items()], key=lambda x: x[1])
    with open(os.path.join(OUT, 'table3_avg_rank.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['方法', '平均秩', '平均准确率'])
        for m, r, a in t3:
            w.writerow([m, f'{r:.2f}', f'{a:.4f}'])
    cd_rank = [r for m, r, _ in t3 if m == 'CDPSO-BP'][0]
    print(f'    CDPSO-BP 平均秩 {cd_rank:.2f}，第 '
          f'{[m for m,_,_ in t3].index("CDPSO-BP")+1}/{len(t3)} 名')
    for m, r, a in t3[:5]:
        print(f'    {m:28s} {r:.2f}  {a:.4f}')

    # 主方法内部 Friedman
    sub = [m for m in MAIN if all((d, m) in data for d in datasets)]
    if len(sub) >= 3:
        mat = np.array([[np.mean(data[(d, m)]) for m in sub] for d in datasets])
        fr = friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
        print(f'    五种主方法跨数据集 Friedman: p = {fr.pvalue:.3g}')

    # ---- 表4：break-even 分析 ----
    print('\n[表4] break-even：纯 BP 追平 CDPSO-BP 所需 epoch')
    bc = read_csv('budget_curve.csv')
    for r in bc:
        r['epoch'] = int(r['epoch'])
        r['accuracy'] = float(r['accuracy']) if r['accuracy'] not in ('', 'nan') else np.nan
    eps = sorted(set(r['epoch'] for r in bc))
    t4 = []
    for ds in datasets:
        tgt = np.mean(data[(ds, 'CDPSO-BP')])
        curve = {E: np.mean([r['accuracy'] for r in bc
                             if r['dataset'] == ds and r['epoch'] == E]) for E in eps}
        be = next((E for E in eps if curve[E] >= tgt), None)
        t4.append([ds, len(data[(ds, 'CDPSO-BP')]), f'{tgt:.4f}',
                   f'{curve[30]:.4f}', f'{curve[100]:.4f}', f'{curve[300]:.4f}',
                   be if be is not None else '>300',
                   f'{(640 + be*6.3)/(30*6.3):.1f}' if be is not None else '—'])
    with open(os.path.join(OUT, 'table4_breakeven.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['数据集', '种子数', 'CDPSO-BP 精度', 'BP@30', 'BP@100', 'BP@300',
                    'break-even E*', '等效倍数'])
        w.writerows(t4)
    be_v = [row[6] for row in t4 if isinstance(row[6], int)]
    print(f'    break-even E*: 覆盖 {len(be_v)}/{len(t4)}，'
          f'中位数 {np.median(be_v):.0f}，范围 {min(be_v)}–{max(be_v)}')
    for row in t4:
        print('    ', ' | '.join(str(x) for x in row))

    # ---- 表5：计算开销 ----
    print('\n[表5] 计算开销（以单次 BP-epoch 前向等价单位 n 计）')
    t5 = [['BP-30', '—', '63n', '63n', '与 CDPSO-BP 等预算'],
          ['BP（≤300 epoch + 早停）', '—', '630n', '≤630n', '上界，早停可更低'],
          ['PSO-BP / IPSO-BP / CDPSO-BP', '640n', '63n', '703n', '搜索 40代×20粒子×0.8n']]
    with open(os.path.join(OUT, 'table5_cost.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['方法', '搜索开销', '微调开销', '合计', '说明'])
        w.writerows(t5)
    for r in t5:
        print('    ', ' | '.join(r))

    # ---- 汇总 ----
    summary = {
        'n_datasets': len(datasets),
        'n_seeds': 30,
        'n_main_methods': len(MAIN),
        'n_ablation': len(abl),
        'breakeven_median': float(np.median(be_v)) if be_v else None,
        'breakeven_min': int(min(be_v)) if be_v else None,
        'breakeven_max': int(max(be_v)) if be_v else None,
        'breakeven_coverage': f'{len(be_v)}/{len(t4)}',
        'cdpso_avg_rank': cd_rank,
    }
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n汇总:', json.dumps(summary, ensure_ascii=False))
    print(f'\n输出目录: {OUT}')


if __name__ == '__main__':
    main()
