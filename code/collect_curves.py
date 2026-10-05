"""采集搜索阶段收敛曲线（PSO / IPSO / CDPSO 三种策略），供正文 §3.2 使用。

同时采集多样性轨迹 d(t)/d(0)，用于说明多样性反馈的作用。
输出 results/search_curves.csv
"""
import os
import csv
import json

import numpy as np

import run_experiment as R

OUT = 'results/search_curves.csv'
N_SEEDS = 30
DATASETS = ['LoL', 'Heart', 'Sonar', 'Pima', 'WDBC', 'Banknote']
MODES = {'PSO': 'pso', 'IPSO': 'ipso', 'CDPSO': 'cdpso_full'}


def run_with_trace(X_fit, y_fit, d_in, seed, mode):
    """复制 pso_search 的核心逻辑，同时记录适应度与多样性轨迹。"""
    rng = np.random.default_rng(seed)
    dim = d_in * R.HIDDEN + R.HIDDEN + R.HIDDEN + 1
    N, T, B = R.N_POP, R.T_MAX, R.B

    use_obl = mode in ('cdpso_full',)
    use_chaos = mode != 'pso'
    use_diversity = mode in ('cdpso_full',)
    use_perturb = mode in ('cdpso_full',)
    use_restart = mode in ('cdpso_full',)
    use_schedule_w = mode != 'pso'

    if use_obl:
        Xc = (R.tent_sequence(rng, N, dim) * 2.0 - 1.0) * B
        cand = np.vstack([Xc, -Xc])
        fits = np.array([R.mse(R.forward(X_fit, c, d_in), y_fit) for c in cand])
        X = cand[np.argsort(fits)[:N]]
        V = rng.uniform(-B, B, size=(N, dim))
    elif use_chaos:
        X = (R.tent_sequence(rng, N, dim) * 2.0 - 1.0) * B
        V = rng.uniform(-B, B, size=(N, dim))
    else:
        X = rng.uniform(-B, B, size=(N, dim))
        V = rng.uniform(-B, B, size=(N, dim))

    fit = np.array([R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X])
    pbest, pbest_fit = X.copy(), fit.copy()
    gi = int(np.argmin(pbest_fit))
    g, g_fit = pbest[gi].copy(), pbest_fit[gi]
    c0 = X.mean(axis=0)
    d0 = float(np.mean(np.linalg.norm(X - c0, axis=1))) or 1.0

    stag, rho = 0, 1.0
    fit_trace, rho_trace, w_trace = [], [], []
    for t in range(1, T + 1):
        if use_schedule_w:
            w_base = R.W_MIN + (R.W_MAX - R.W_MIN) * (1.0 - t / T) ** 2
            if use_diversity:
                c = X.mean(axis=0)
                d_t = float(np.mean(np.linalg.norm(X - c, axis=1)))
                rho = min(d_t / d0, 1.0)
                w = w_base * (1.0 + rho) / 2.0
            else:
                w = w_base
        else:
            w = 0.8

        r1 = rng.uniform(0, 1, size=(N, dim))
        r2 = rng.uniform(0, 1, size=(N, dim))
        V = np.clip(w * V + R.C1 * r1 * (pbest - X) + R.C2 * r2 * (g - X),
                    -R.V_CLIP, R.V_CLIP)
        X = np.clip(X + V, -B, B)

        fit = np.array([R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X])
        imp = fit < pbest_fit
        pbest[imp], pbest_fit[imp] = X[imp], fit[imp]
        ni = int(np.argmin(pbest_fit))
        if pbest_fit[ni] < g_fit:
            g, g_fit, stag = pbest[ni].copy(), pbest_fit[ni], 0
        else:
            stag += 1

        if use_perturb and rng.uniform(0, 1) < R.P_M:
            gp = np.clip(g + rng.normal(0, R.SIGMA0 * (1 - t / T), size=dim), -B, B)
            fg = R.mse(R.forward(X_fit, gp, d_in), y_fit)
            if fg < g_fit:
                g, g_fit, stag = gp, fg, 0

        fit_trace.append(float(g_fit))
        w_trace.append(float(w))
        c = X.mean(axis=0)
        d_t = float(np.mean(np.linalg.norm(X - c, axis=1)))
        rho_trace.append(float(min(d_t / d0, 1.0)))

        if use_restart and (stag >= R.STAG_LIMIT or rho < R.RHO_MIN):
            k = max(1, int(R.K_RESET * N))
            wi = np.argsort(pbest_fit)[-k:]
            X[wi] = (R.tent_sequence(rng, k, dim) * 2.0 - 1.0) * B
            V[wi] = rng.uniform(-B, B, size=(k, dim))
            fit[wi] = [R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X[wi]]
            pbest[wi], pbest_fit[wi] = X[wi], fit[wi]
            stag = 0

    return fit_trace, rho_trace, w_trace


def main():
    ds = R.load_all_datasets()
    rows = []
    total = len(DATASETS) * len(MODES) * N_SEEDS
    done = 0
    for name in DATASETS:
        X, y = ds[name]
        d_in = X.shape[1]
        for label, mode in MODES.items():
            acc = {k: [] for k in ('fit', 'rho', 'w')}
            for s in range(N_SEEDS):
                Xtr, ytr, Xva, yva, _, _ = R.split_standardize(X, y, s)
                X_fit = np.vstack([Xtr, Xva])
                y_fit = np.concatenate([ytr, yva])
                f, r, w = run_with_trace(X_fit, y_fit, d_in, s, mode)
                for t in range(len(f)):
                    rows.append(dict(dataset=name, method=label, seed=3000 + s,
                                     gen=t + 1, gbest_fit=f[t], rho=r[t], inertia=w[t]))
                done += 1
            print(f'  {name:10s} {label:6s} 完成 ({done}/{total})')
    with open(OUT, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'共 {len(rows)} 行 -> {OUT}')


if __name__ == '__main__':
    main()
