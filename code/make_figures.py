# -*- coding: utf-8 -*-
"""生成论文图1（LoL）与图2（Heart）的真实收敛曲线。

左：PSO/IPSO/CDPSO 搜索阶段 gbest 适应度（5 次重复均值±标准差带）
右：三者在 BP 微调阶段（30 epoch）的验证损失曲线
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_experiment as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
N_SEEDS = 5
MODES = [("PSO", "pso"), ("IPSO", "ipso"), ("CDPSO", "cdpso_full")]
COLORS = {"PSO": "#1f77b4", "IPSO": "#ff7f0e", "CDPSO": "#2ca02c"}


def load(ds_name):
    heart_cands, lol_cands = R.get_datasets()
    hp = next(p for p in heart_cands if os.path.exists(p))
    lp = next(p for p in lol_cands if os.path.exists(p))
    X, y = (R.load_heart(hp) if ds_name == "Heart" else R.load_lol(lp))
    return R.split_standardize(X, y, seed=0)


def collect(ds_name):
    Xtr, ytr, Xva, yva, Xte, yte = load(ds_name)
    d_in = Xtr.shape[1]
    X_fit = np.vstack([Xtr, Xva])
    y_fit = np.concatenate([ytr, yva])
    search = {m: [] for m, _ in MODES}
    valcur = {m: [] for m, _ in MODES}
    for name, mode in MODES:
        for k in range(N_SEEDS):
            rng = np.random.default_rng(3000 + k)
            g, _ = R.pso_search(X_fit, y_fit, d_in, 3000 + k, mode, rng=rng)
            search[name].append(list(R.LAST_SEARCH_CURVE))
            R.bp_finetune(g, Xtr, ytr, d_in, epochs=30, Xva=Xva, yva=yva)
            valcur[name].append(list(R.LAST_VAL_CURVE))
    return search, valcur


def plot(ds_name, search, valcur, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name, _ in MODES:
        arr = np.array(search[name], dtype=float)
        mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        x = np.arange(1, arr.shape[1] + 1)
        ax.plot(x, mu, label=name, color=COLORS[name], linewidth=1.8)
        ax.fill_between(x, mu - sd, mu + sd, color=COLORS[name], alpha=0.18)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (MSE)")
    ax.set_title("(a) PSO search stage")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, _ in MODES:
        arr = np.array(valcur[name], dtype=float)
        mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        x = np.arange(1, arr.shape[1] + 1)
        ax.plot(x, mu, label=name, color=COLORS[name], linewidth=1.8)
        ax.fill_between(x, mu - sd, mu + sd, color=COLORS[name], alpha=0.18)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (MSE)")
    ax.set_title("(b) BP fine-tuning stage (30 epochs)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("%s dataset" % ds_name, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for ds_name, fname in (("LoL", "fig1_lol.png"), ("Heart", "fig2_heart.png")):
        s, v = collect(ds_name)
        plot(ds_name, s, v, os.path.join(OUT_DIR, fname))
