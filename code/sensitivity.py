# -*- coding: utf-8 -*-
"""
sensitivity.py —— 参数敏感性实验（表6）
========================================
对 CDPSO-BP 关键参数做网格：p_m, sigma0, 停滞阈值, 种群 N, 迭代 T。
在 Heart 数据集上各跑 10 个种子，输出 results/sensitivity.csv
（列: param,value,dataset,acc_std,rank,note），供 exp_harness.py 生成表6。

通过临时修改 run_experiment 的全局参数实现，避免重复实现 pso_search。
"""

import os
import csv
import argparse
import numpy as np

import run_experiment as R
from run_experiment import (
    load_heart, split_standardize, bp_finetune, evaluate, mse, forward,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SENS_PATH = os.path.join(HERE, "results", "sensitivity.csv")
DATA_DIR = os.path.join(HERE, "data")


def run_one_cdpso(Xtr, ytr, Xva, yva, Xte, yte, d_in, seed):
    rng = np.random.default_rng(seed)
    X_fit = np.vstack([Xtr, Xva])
    y_fit = np.concatenate([ytr, yva])
    g, g_fit = R.pso_search(X_fit, y_fit, d_in, seed, "cdpso_full", rng=rng)
    w = bp_finetune(g, Xtr, ytr, d_in, epochs=30)
    acc, f1 = evaluate(w, Xte, yte, d_in)
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    X, y = load_heart(os.path.join(DATA_DIR, "heart.csv"))
    d_in = X.shape[1]

    # 默认参数（论文 §4.2）
    defaults = {"p_m": 0.2, "sigma0": 0.2, "stagnation": 2, "N": 20, "T": 40}
    grid = {
        "p_m":        [0.0, 0.1, 0.2, 0.4],
        "sigma0":     [0.1, 0.2, 0.4],
        "stagnation": [1, 3, 6],
        "N":          [10, 20, 40],
        "T":          [20, 40, 80],
    }

    # 预先把默认参数下的 10 种子结果算出来，作为基准
    base_accs = []
    for seed in range(args.seeds):
        R.N_POP = defaults["N"]; R.T_MAX = defaults["T"]
        R.P_M = defaults["p_m"]; R.SIGMA0 = defaults["sigma0"]
        R.STAG_LIMIT = defaults["stagnation"]
        Xtr, ytr, Xva, yva, Xte, yte = split_standardize(X, y, seed)
        a, _ = run_one_cdpso(Xtr, ytr, Xva, yva, Xte, yte, d_in, seed)
        base_accs.append(a)
    base_mean = float(np.mean(base_accs))
    base_std = float(np.std(base_accs, ddof=1))

    rows = []
    # 基准行
    rows.append(["default", "p_m=0.2,sigma0=0.2,stag=2,N=20,T=40", "Heart",
                 f"{base_mean:.4f}±{base_std:.4f}", 0,
                 "基准（论文默认参数）"])

    for param, values in grid.items():
        for val in values:
            # 跳过默认值本身（已作基准）
            is_default = (param == "p_m" and abs(val - 0.2) < 1e-9) or \
                         (param == "sigma0" and abs(val - 0.2) < 1e-9) or \
                         (param == "stagnation" and val == 2) or \
                         (param == "N" and val == 20) or \
                         (param == "T" and val == 40)
            if is_default:
                continue
            accs = []
            for seed in range(args.seeds):
                R.N_POP = defaults["N"]; R.T_MAX = defaults["T"]
                R.P_M = defaults["p_m"]; R.SIGMA0 = defaults["sigma0"]
                R.STAG_LIMIT = defaults["stagnation"]
                if param == "p_m":
                    R.P_M = val
                elif param == "sigma0":
                    R.SIGMA0 = val
                elif param == "stagnation":
                    R.STAG_LIMIT = val
                elif param == "N":
                    R.N_POP = val
                elif param == "T":
                    R.T_MAX = val
                Xtr, ytr, Xva, yva, Xte, yte = split_standardize(X, y, seed)
                a, _ = run_one_cdpso(Xtr, ytr, Xva, yva, Xte, yte, d_in, seed)
                accs.append(a)
            m = float(np.mean(accs)); s = float(np.std(accs, ddof=1))
            delta = m - base_mean
            note = f"较基准 {'+' if delta >= 0 else ''}{delta:.4f}"
            rows.append([param, str(val), "Heart", f"{m:.4f}±{s:.4f}", "", note])
            print(f"  {param}={val}: ACC {m:.4f}±{s:.4f}  ({note})")

    # 恢复默认
    R.N_POP = defaults["N"]; R.T_MAX = defaults["T"]
    R.P_M = defaults["p_m"]; R.SIGMA0 = defaults["sigma0"]
    R.STAG_LIMIT = defaults["stagnation"]

    with open(SENS_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "dataset", "acc_std", "rank", "note"])
        for r in rows:
            w.writerow(r)
    print(f"\n完成：{len(rows)} 行敏感性结果已写入 {SENS_PATH}")


if __name__ == "__main__":
    main()
