# -*- coding: utf-8 -*-
"""
baselines.py —— Q1 补充基线实现（表5）
========================================
在 run_experiment.py 基础上，实现 ≥10 个补充基线，与 CDPSO-BP 在同一预算
（搜索 20 pop × 40 iter 或等价 800 次适应度评估；梯度基线用 30 epoch）下
优化 BP 初始权值，再统一 30 epoch 微调（梯度基线直接训练 30 epoch）。

基线清单：
  梯度基线（强初始化 + Adam）：Xavier+Adam, Kaiming+Adam
  PSO 变体：CLPSO（综合学习）, APSO（自适应）, AMPSO（自适应变异）,
            DMPSO（多样性多变异）, GLPSO（遗传学习）
  差分进化：SHADE, jSO
  进化策略：CMA-ES（cma 库）

输出：append 到 results/raw_results.csv（source=real_experiment）。
"""

import os
import csv
import argparse
import numpy as np

from run_experiment import (
    load_heart, load_lol, split_standardize, forward, mse, bp_finetune,
    evaluate, decode, HIDDEN, B, V_CLIP, N_POP, T_MAX, C1, C2, W_MAX, W_MIN,
    tent_sequence,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "results", "raw_results.csv")
DATA_DIR = os.path.join(HERE, "data")


def dim_of(d_in):
    return d_in * HIDDEN + HIDDEN + HIDDEN + 1


# ----------------------------------------------------------------------------
# 梯度基线：Xavier / Kaiming 初始化 + Adam 训练 30 epoch
# ----------------------------------------------------------------------------
def _compute_grad(w, X, y, d_in):
    n = X.shape[0]
    W1, b1, W2, b2 = decode(w, d_in)
    z1 = X @ W1.T + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2.T + b2
    yh = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))
    yh = yh.reshape(-1)
    dL = 2.0 * (yh - y) / n
    delta2 = (dL * yh * (1.0 - yh)).reshape(-1, 1)
    gW2 = delta2.T @ a1
    gb2 = delta2.sum(axis=0)
    delta1 = (delta2 @ W2) * (1.0 - a1 ** 2)
    gW1 = delta1.T @ X
    gb1 = delta1.sum(axis=0)
    return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])


def _init_weight(shape, mode, rng):
    fan_in = shape[1] if len(shape) == 2 else shape[0]
    fan_out = shape[0] if len(shape) == 2 else 1
    if mode == "xavier":
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-lim, lim, size=shape)
    else:  # kaiming (He)
        sd = np.sqrt(2.0 / max(fan_in, 1))
        return rng.normal(0, sd, size=shape)


def adam_train(X, y, d_in, init_mode, rng, epochs=30, lr=0.001):
    dim = dim_of(d_in)
    W1 = _init_weight((HIDDEN, d_in), init_mode, rng)
    b1 = np.zeros(HIDDEN)
    W2 = _init_weight((1, HIDDEN), init_mode, rng)
    b2 = np.zeros(1)
    w = np.concatenate([W1.ravel(), b1, W2.ravel(), b2])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        g = _compute_grad(w, X, y, d_in)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        mh = m / (1 - beta1 ** t)
        vh = v / (1 - beta2 ** t)
        w = w - lr * mh / (np.sqrt(vh) + eps)
    return w


# ----------------------------------------------------------------------------
# PSO 变体（均搜索 train+val MSE，再交 30 epoch 微调）
# ----------------------------------------------------------------------------
def _run_pso_variant(X_fit, y_fit, d_in, rng, variant):
    dim = dim_of(d_in)
    X = rng.uniform(-B, B, size=(N_POP, dim))
    V = rng.uniform(-B, B, size=(N_POP, dim))
    fit = np.array([mse(forward(X_fit, x, d_in), y_fit) for x in X])
    pbest = X.copy(); pbest_fit = fit.copy()
    gi = int(np.argmin(pbest_fit)); g = pbest[gi].copy(); g_fit = pbest_fit[gi]

    # CLPSO 综合学习范例与概率
    Pc = 0.05 + 0.45 * (np.arange(dim) / (dim - 1)) if dim > 1 else np.full(dim, 0.5)
    exemplar = np.arange(N_POP)  # 每粒子每维的学习对象（初始为自己）
    # APSO 进化因子初值
    f_evo = 0.0

    for t in range(1, T_MAX + 1):
        r1 = rng.uniform(0, 1, size=(N_POP, dim))
        r2 = rng.uniform(0, 1, size=(N_POP, dim))
        if variant == "CLPSO":
            w = W_MIN + (W_MAX - W_MIN) * (1 - t / T_MAX) ** 2
            c = 1.49445
            # 每粒子按 Pc 决定向自己 pbest 或他人 pbest 学习
            learn = (rng.uniform(0, 1, size=(N_POP, dim)) < Pc[None, :])
            pbest_mat = pbest  # (N,dim)
            # 为每个粒子构造学习目标：learn 时为随机他人 pbest，否则自己 pbest
            target = pbest.copy()
            for i in range(N_POP):
                others = [j for j in range(N_POP) if j != i]
                for d in range(dim):
                    if learn[i, d]:
                        j = int(rng.choice(others))
                        target[i, d] = pbest[j, d]
            V = w * V + c * r1 * (target - X)
        elif variant in ("APSO", "AMPSO"):
            # 进化因子 f 基于平均距离
            diag = np.linalg.norm(X - X.mean(axis=0), axis=1)
            dmax = diag.max(); dmin = diag.min()
            f = (diag.mean() - dmin) / (dmax - dmin + 1e-12)
            f_evo = f
            w = 1.0 / (1.0 + 1.5 * np.exp(-2.6 * f))
            c1 = 2.5 - 2.0 * f
            c2 = 0.5 + 2.0 * f
            V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (g - X)
            # jumping-out（勘探态对 g 扰动）
            if f < 0.4:
                g = np.clip(g + rng.normal(0, 0.1 * B, size=dim), -B, B)
                g_fit = mse(forward(X_fit, g, d_in), y_fit)
        elif variant == "DMPSO":
            # 多样性引导多变异：diversity 低时对 pbest/gbest 高斯/柯西扰动
            center = X.mean(axis=0)
            div = float(np.mean(np.linalg.norm(X - center, axis=1)))
            w = W_MIN + (W_MAX - W_MIN) * (1 - t / T_MAX) ** 2
            V = w * V + C1 * r1 * (pbest - X) + C2 * r2 * (g - X)
            if div < 1e-3:
                mu_idx = int(rng.integers(0, N_POP))
                pbest[mu_idx] = np.clip(
                    pbest[mu_idx] + rng.standard_cauchy(size=dim) * 0.01, -B, B)
        elif variant == "GLPSO":
            # 遗传学习 PSO：以 GA 交叉产生后代替换最差粒子
            w = W_MIN + (W_MAX - W_MIN) * (1 - t / T_MAX) ** 2
            V = w * V + C1 * r1 * (pbest - X) + C2 * r2 * (g - X)
        else:
            raise ValueError(variant)

        V = np.clip(V, -V_CLIP, V_CLIP)
        X = np.clip(X + V, -B, B)
        fit = np.array([mse(forward(X_fit, x, d_in), y_fit) for x in X])
        imp = fit < pbest_fit
        pbest[imp] = X[imp]; pbest_fit[imp] = fit[imp]
        gi = int(np.argmin(pbest_fit))
        if pbest_fit[gi] < g_fit:
            g = pbest[gi].copy(); g_fit = pbest_fit[gi]

        if variant == "GLPSO":
            # 每 5 代做一次 GA 交叉（均匀交叉 + 高斯变异）
            if t % 5 == 0:
                order = np.argsort(pbest_fit)
                elite = pbest[order[:2]]
                child = 0.5 * (elite[0] + elite[1]) + rng.normal(0, 0.05 * B, size=dim)
                child = np.clip(child, -B, B)
                worst_i = order[-1]
                X[worst_i] = child
                fit[worst_i] = mse(forward(X_fit, child, d_in), y_fit)
                pbest[worst_i] = child; pbest_fit[worst_i] = fit[worst_i]
        # AMPSO 变异
        if variant == "AMPSO":
            pm = 0.1 if f_evo < 0.4 else 0.05
            if rng.uniform() < pm:
                idx = int(rng.integers(0, N_POP))
                pbest[idx] = np.clip(pbest[idx] + rng.normal(0, 0.1 * B, size=dim), -B, B)
    return g, g_fit


# ----------------------------------------------------------------------------
# 差分进化：SHADE / jSO
# ----------------------------------------------------------------------------
def _run_de(X_fit, y_fit, d_in, rng, variant):
    dim = dim_of(d_in)
    N = N_POP
    X = rng.uniform(-B, B, size=(N, dim))
    fit = np.array([mse(forward(X_fit, x, d_in), y_fit) for x in X])
    H = 5
    M_CR = np.full(H, 0.5); M_F = np.full(H, 0.5)
    k = 0
    pbest_ratio = 0.1
    max_evals = N * T_MAX  # 与 PSO 预算对齐
    evals = N
    while evals < max_evals:
        S_CR, S_F, S_df = [], [], []
        for i in range(N):
            r = rng.integers(0, H)
            if variant == "SHADE":
                CR = np.clip(rng.normal(M_CR[r], 0.1), 0, 1)
                F = np.clip(M_F[r] + 0.1 * rng.standard_cauchy(), 0, 1)
            else:  # jSO：随评估进度动态
                F = 0.3 + 0.5 * (evals / max_evals)
                F = np.clip(F + 0.1 * rng.standard_cauchy(), 0, 1)
                CR = 0.9 if rng.uniform() < 0.25 else np.clip(rng.normal(0.8, 0.1), 0, 1)
            # current-to-pbest
            pnum = max(2, int(N * pbest_ratio))
            order = np.argsort(fit)
            pbest_pool = order[:pnum]
            p = int(rng.choice(pbest_pool))
            cands = [j for j in range(N) if j != i]
            r1 = int(rng.choice(cands))
            cands2 = [j for j in cands if j != r1]
            r2 = int(rng.choice(cands2))
            v = X[i] + F * (X[p] - X[i]) + F * (X[r1] - X[r2])
            v = np.clip(v, -B, B)
            jrand = int(rng.integers(0, dim))
            mask = rng.uniform(0, 1, size=dim) < CR
            mask[jrand] = True
            u = np.where(mask, v, X[i])
            fu = mse(forward(X_fit, u, d_in), y_fit)
            evals += 1
            if fu <= fit[i]:
                X[i] = u
                S_CR.append(CR); S_F.append(F); S_df.append(abs(fu - fit[i]))
                fit[i] = fu
            if evals >= max_evals:
                break
        if S_CR:
            weights = np.array(S_df) / (np.sum(S_df) + 1e-12)
            M_CR[k] = np.sum(weights * np.array(S_CR)) / np.sum(weights)
            M_F[k] = np.sum(weights * np.array(S_F) ** 2) / np.sum(weights * np.array(S_F) + 1e-12)
            k = (k + 1) % H
    gi = int(np.argmin(fit))
    return X[gi].copy(), fit[gi]


# ----------------------------------------------------------------------------
# CMA-ES（cma 库）
# ----------------------------------------------------------------------------
def _run_cmaes(X_fit, y_fit, d_in, rng):
    import cma
    dim = dim_of(d_in)
    x0 = rng.uniform(-B, B, size=dim)
    sigma0 = 0.3 * B

    def obj(x):
        return mse(forward(X_fit, x, d_in), y_fit)

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "popsize": N_POP, "maxfevals": N_POP * T_MAX, "seed": int(rng.integers(0, 2**31 - 1)),
        "bounds": [-B, B], "verbose": -9,
    })
    es.optimize(obj)
    return np.clip(es.result.xbest, -B, B), es.result.fbest


# ----------------------------------------------------------------------------
# 基线主流程
# ----------------------------------------------------------------------------
BASELINES = ["CLPSO", "APSO", "AMPSO", "DMPSO", "GLPSO", "SHADE", "jSO", "CMA-ES",
             "Xavier+Adam", "Kaiming+Adam"]
PSO_VARIANTS = {"CLPSO", "APSO", "AMPSO", "DMPSO", "GLPSO"}
DE_VARIANTS = {"SHADE", "jSO"}
GRAD_INIT = {"Xavier+Adam": "xavier", "Kaiming+Adam": "kaiming"}


def run_baseline(Xtr, ytr, Xva, yva, Xte, yte, d_in, method, seed, rng):
    X_fit = np.vstack([Xtr, Xva]); y_fit = np.concatenate([ytr, yva])
    if method in GRAD_INIT:
        w = adam_train(Xtr, ytr, d_in, GRAD_INIT[method], rng, epochs=30)
        acc, f1 = evaluate(w, Xte, yte, d_in)
        return acc, f1, float("nan")
    if method in PSO_VARIANTS:
        g, g_fit = _run_pso_variant(X_fit, y_fit, d_in, rng, method)
    elif method in DE_VARIANTS:
        g, g_fit = _run_de(X_fit, y_fit, d_in, rng, method)
    elif method == "CMA-ES":
        g, g_fit = _run_cmaes(X_fit, y_fit, d_in, rng)
    else:
        raise ValueError(method)
    w = bp_finetune(g, Xtr, ytr, d_in, epochs=30)
    acc, f1 = evaluate(w, Xte, yte, d_in)
    return acc, f1, g_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--datasets", default="LoL,Heart")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    args = ap.parse_args()

    heart_path = os.path.join(DATA_DIR, "heart.csv")
    lol_path = os.path.join(DATA_DIR, "lol.csv")
    Xh, yh = load_heart(heart_path)
    Xl, yl = load_lol(lol_path)
    datasets = {"Heart": (Xh, yh), "LoL": (Xl, yl)}

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    rows = []
    total = len(datasets) * len(baselines) * args.seeds
    done = 0
    for ds_name in args.datasets.split(","):
        ds_name = ds_name.strip()
        if ds_name not in datasets:
            continue
        X, y = datasets[ds_name]
        d_in = X.shape[1]
        for method in baselines:
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                Xtr, ytr, Xva, yva, Xte, yte = split_standardize(X, y, seed)
                acc, f1, sf = run_baseline(Xtr, ytr, Xva, yva, Xte, yte,
                                           d_in, method, seed, rng)
                rows.append([ds_name, method, seed, f"{acc:.6f}", f"{f1:.6f}",
                             ("" if sf != sf else f"{sf:.6f}"), "real_experiment"])
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  进度 {done}/{total}  [{method}@{ds_name} seed{seed} "
                          f"acc={acc:.4f} f1={f1:.4f}]")

    # append 到 raw_results.csv
    with open(RAW_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"\n完成：{len(rows)} 条基线真实结果已追加到 {RAW_PATH}")


if __name__ == "__main__":
    main()
