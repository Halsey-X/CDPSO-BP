"""参数敏感性：3 个代表性数据集（原 sensitivity.py 仅跑 Heart，单数据集结论不可靠）。

数据集选择依据：Heart(918样本/19特征)、Sonar(208/59，最高维)、Banknote(1372/4，最低维)
"""
import csv
import os
import time

import numpy as np

import run_experiment as R

OUT = 'results/sensitivity.csv'
OUT3 = 'results/sensitivity_3ds.csv'
SEEDS = 10
DS_LIST = ['Heart', 'Sonar', 'Banknote']

GRID = [
    ('p_m', [0.0, 0.1, 0.4]),
    ('sigma0', [0.1, 0.4]),
    ('stagnation', [1, 3, 6]),
    ('N', [10, 40]),
    ('T', [20, 80]),
]


def run_one(Xtr, ytr, Xva, yva, Xte, yte, d_in, seed, pm, sig, stag, npop, tmax):
    """在给定超参下跑一遍 CDPSO-BP，返回测试准确率。"""
    old = (R.P_M, R.SIGMA0, R.STAG_LIMIT, R.N_POP, R.T_MAX)
    R.P_M, R.SIGMA0, R.STAG_LIMIT, R.N_POP, R.T_MAX = pm, sig, stag, npop, tmax
    try:
        rng = np.random.default_rng(seed)
        X_fit = np.vstack([Xtr, Xva])
        y_fit = np.concatenate([ytr, yva])
        g, _ = R.pso_search(X_fit, y_fit, d_in, seed, 'cdpso_full', rng=rng)
        w = R.bp_finetune(g, Xtr, ytr, d_in, epochs=30)
        acc, f1 = R.evaluate(w, Xte, yte, d_in)
        return acc
    finally:
        R.P_M, R.SIGMA0, R.STAG_LIMIT, R.N_POP, R.T_MAX = old


def main():
    ds = R.load_all_datasets()
    rows = []
    t0 = time.time()
    total = len(DS_LIST) * (1 + sum(len(v) for _, v in GRID)) * SEEDS
    done = 0
    for name in DS_LIST:
        X, y = ds[name]
        d_in = X.shape[1]
        splits = [R.split_standardize(X, y, s) for s in range(SEEDS)]

        def measure(pm, sig, stag, npop, tmax):
            accs = []
            for s in range(SEEDS):
                Xtr, ytr, Xva, yva, Xte, yte = splits[s]
                accs.append(run_one(Xtr, ytr, Xva, yva, Xte, yte, d_in, s,
                                    pm, sig, stag, npop, tmax))
            return float(np.mean(accs)), float(np.std(accs, ddof=1))

        m, sd = measure(R.P_M, R.SIGMA0, R.STAG_LIMIT, R.N_POP, R.T_MAX)
        rows.append(dict(param='default', value='default', dataset=name,
                         acc_std=f'{m:.4f}±{sd:.4f}', rank='',
                         note='default configuration'))
        done += SEEDS

        for pname, vals in GRID:
            for v in vals:
                kw = dict(pm=R.P_M, sig=R.SIGMA0, stag=R.STAG_LIMIT,
                          npop=R.N_POP, tmax=R.T_MAX)
                if pname == 'p_m':
                    kw['pm'] = v
                elif pname == 'sigma0':
                    kw['sig'] = v
                elif pname == 'stagnation':
                    kw['stag'] = v
                elif pname == 'N':
                    kw['npop'] = v
                elif pname == 'T':
                    kw['tmax'] = v
                m2, sd2 = measure(**kw)
                rows.append(dict(param=pname, value=str(v), dataset=name,
                                 acc_std=f'{m2:.4f}±{sd2:.4f}', rank='',
                                 note=f'delta={(m2-m)*100:+.4f}pp'))
                done += SEEDS
        print(f'  {name:10s} 完成 ({done}/{total})  {time.time()-t0:.0f}s')

    with open(OUT3, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'共 {len(rows)} 行 -> {OUT3}')


if __name__ == '__main__':
    main()
