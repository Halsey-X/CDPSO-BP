"""
扩展实验模块：在 run_experiment.py 基础上增加
  (1) AUC / 平衡准确率指标
  (2) 干净消融变体 cdpso_no_obl（去反向学习，保留混沌初始化+多样性+扰动+重启）
  (3) 预算–精度曲线：BP 在固定 epoch 网格上的表现
  (4) 控制信号尺度不变性：不同搜索边界 B 下惯性权重轨迹是否重合

以 `python extra_experiments.py <子命令>` 运行。
"""
import os
import sys
import csv
import json
import time

import numpy as np

import run_experiment as R

# ---------------------------------------------------------------------------
# 额外指标：AUC 与平衡准确率
# ---------------------------------------------------------------------------


def evaluate_full(w, Xte, yte, d_in):
    """返回 (acc, f1, auc, bal_acc)。AUC 用秩和公式（等价于 Mann–Whitney U）。"""
    yh = R.forward(Xte, w, d_in)
    pred = (yh >= 0.5).astype(int)
    y = yte.astype(int)

    acc = float(np.mean(pred == y))
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # 平衡准确率 = (TPR + TNR) / 2
    tnr = float(np.sum((pred == 0) & (y == 0))) / max(1.0, float(np.sum(y == 0)))
    bal = 0.5 * (rec + tnr)

    # AUC（秩和）
    pos = yh[y == 1]
    neg = yh[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        allv = np.concatenate([pos, neg])
        order = allv.argsort()
        ranks = np.empty(len(allv), float)
        ranks[order] = np.arange(1, len(allv) + 1)
        # 并列取平均秩
        _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
        if np.any(cnt > 1):
            sums = np.zeros(len(cnt))
            np.add.at(sums, inv, ranks)
            ranks = (sums / cnt)[inv]
        r_pos = ranks[:len(pos)].sum()
        auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

    return acc, f1, auc, bal


# ---------------------------------------------------------------------------
# 干净消融：CDPSO 去掉反向学习（保留混沌初始化 + 多样性 + 扰动 + 重启）
# ---------------------------------------------------------------------------


def pso_search_no_obl(X_fit, y_fit, d_in, seed, rng=None):
    """等价于 run_experiment.pso_search 的 cdpso_full，但 use_obl=False。

    即：Tent 混沌初始化（不含反向解）+ 多样性反馈 + 高斯扰动 + 停滞重启。
    为避免改动主实验脚本，这里复制其核心逻辑并仅去掉反向学习分支。
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    dim = d_in * R.HIDDEN + R.HIDDEN + R.HIDDEN + 1
    N, T = R.N_POP, R.T_MAX
    B = R.B

    use_chaos = True
    use_diversity = True
    use_perturb = True
    use_restart = True
    use_schedule_w = True

    if use_chaos:
        X = (R.tent_sequence(rng, N, dim) * 2.0 - 1.0) * B
        V = rng.uniform(-B, B, size=(N, dim))

    fit = np.array([R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X])
    pbest = X.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    g = pbest[g_idx].copy()
    g_fit = pbest_fit[g_idx]

    center0 = X.mean(axis=0)
    d0 = float(np.mean(np.linalg.norm(X - center0, axis=1)))
    d0 = d0 if d0 > 1e-12 else 1.0

    stag = 0
    rho_now = 1.0
    w_trace = []
    curve = []
    for t in range(1, T + 1):
        w_base = R.W_MIN + (R.W_MAX - R.W_MIN) * (1.0 - t / T) ** 2
        center = X.mean(axis=0)
        d_t = float(np.mean(np.linalg.norm(X - center, axis=1)))
        rho = min(d_t / d0, 1.0)
        rho_now = rho
        w = w_base * (1.0 + rho) / 2.0
        w_trace.append(float(w))

        r1 = rng.uniform(0, 1, size=(N, dim))
        r2 = rng.uniform(0, 1, size=(N, dim))
        V = w * V + R.C1 * r1 * (pbest - X) + R.C2 * r2 * (g - X)
        V = np.clip(V, -R.V_CLIP, R.V_CLIP)
        X = np.clip(X + V, -B, B)

        fit = np.array([R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X])
        improve = fit < pbest_fit
        pbest[improve] = X[improve]
        pbest_fit[improve] = fit[improve]
        new_g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[new_g_idx] < g_fit:
            g = pbest[new_g_idx].copy()
            g_fit = pbest_fit[new_g_idx]
            stag = 0
        else:
            stag += 1

        if use_perturb and rng.uniform(0, 1) < R.P_M:
            sigma = R.SIGMA0 * (1.0 - t / T)
            gp = np.clip(g + rng.normal(0, sigma, size=dim), -B, B)
            fgp = R.mse(R.forward(X_fit, gp, d_in), y_fit)
            if fgp < g_fit:
                g, g_fit, stag = gp, fgp, 0

        curve.append(float(g_fit))

        if use_restart and (stag >= R.STAG_LIMIT or rho_now < R.RHO_MIN):
            n_reset = max(1, int(R.K_RESET * N))
            worst = np.argsort(pbest_fit)[-n_reset:]
            X[worst] = (R.tent_sequence(rng, n_reset, dim) * 2.0 - 1.0) * B
            V[worst] = rng.uniform(-B, B, size=(n_reset, dim))
            fit[worst] = [R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X[worst]]
            pbest[worst] = X[worst]
            pbest_fit[worst] = fit[worst]
            stag = 0

    return g, g_fit, w_trace, curve


def run_no_obl(Xtr, ytr, Xva, yva, Xte, yte, d_in, seed, rng):
    X_fit = np.vstack([Xtr, Xva])
    y_fit = np.concatenate([ytr, yva])
    g, g_fit, w_trace, curve = pso_search_no_obl(X_fit, y_fit, d_in, seed, rng=rng)
    w = R.bp_finetune(g, Xtr, ytr, d_in, epochs=30)
    acc, f1, auc, bal = evaluate_full(w, Xte, yte, d_in)
    return acc, f1, auc, bal, g_fit, w_trace, curve


# ---------------------------------------------------------------------------
# 预算–精度曲线：BP 在固定 epoch 网格
# ---------------------------------------------------------------------------

EPOCH_GRID = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]


def budget_curve(out_csv, seeds=30, epochs_grid=None):
    epochs_grid = epochs_grid or EPOCH_GRID
    ds = R.load_all_datasets()
    names = list(ds.keys())
    rows = []
    total = len(names) * len(epochs_grid) * seeds
    done = 0
    t0 = time.time()
    print(f"[budget_curve] {len(names)} 数据集 x {len(epochs_grid)} 档 x {seeds} 种子 "
          f"= {total} 次运行")
    for name in names:
        X, y = ds[name]
        d_in = X.shape[1]
        for E in epochs_grid:
            for s in range(seeds):
                rng = np.random.default_rng(3000 + s)
                Xtr, ytr, Xva, yva, Xte, yte = R.split_standardize(X, y, s)
                w0 = rng.uniform(-R.B, R.B, size=d_in * R.HIDDEN + R.HIDDEN
                                 + R.HIDDEN + 1)
                if E == 0:
                    w = w0
                else:
                    w = R.bp_finetune(w0, Xtr, ytr, d_in, epochs=E)
                acc, f1, auc, bal = evaluate_full(w, Xte, yte, d_in)
                rows.append(dict(dataset=name, epoch=E, seed=3000 + s,
                                 accuracy=acc, f1=f1, auc=auc, bal_acc=bal))
                done += 1
            if done % 200 == 0:
                print(f"  进度 {done}/{total}  {time.time()-t0:.0f}s")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"[budget_curve] 完成，用时 {time.time()-t0:.0f}s -> {out_csv}")


# ---------------------------------------------------------------------------
# 尺度不变性：不同搜索边界 B 下，CDPSO 的惯性权重轨迹是否完全重合
# ---------------------------------------------------------------------------

B_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]


def inertia_trace(B, X_fit, y_fit, d_in, seed):
    """在给定搜索边界 B 下运行 CDPSO，返回逐代惯性权重序列。"""
    rng = np.random.default_rng(seed)
    dim = d_in * R.HIDDEN + R.HIDDEN + R.HIDDEN + 1
    N, T = R.N_POP, R.T_MAX

    Xc = (R.tent_sequence(rng, N, dim) * 2.0 - 1.0) * B
    cand = np.vstack([Xc, -Xc])
    fits = np.array([R.mse(R.forward(X_fit, c, d_in), y_fit) for c in cand])
    X = cand[np.argsort(fits)[:N]]
    V = rng.uniform(-B, B, size=(N, dim))

    fit = np.array([R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X])
    pbest, pbest_fit = X.copy(), fit.copy()
    gi = int(np.argmin(pbest_fit))
    g, g_fit = pbest[gi].copy(), pbest_fit[gi]
    c0 = X.mean(axis=0)
    d0 = float(np.mean(np.linalg.norm(X - c0, axis=1))) or 1.0
    stag, rho = 0, 1.0
    trace = []
    for t in range(1, T + 1):
        w_base = R.W_MIN + (R.W_MAX - R.W_MIN) * (1.0 - t / T) ** 2
        c = X.mean(axis=0)
        d_t = float(np.mean(np.linalg.norm(X - c, axis=1)))
        rho = min(d_t / d0, 1.0)
        w = w_base * (1.0 + rho) / 2.0
        trace.append(w)
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
        if rng.uniform(0, 1) < R.P_M:
            gp = np.clip(g + rng.normal(0, R.SIGMA0 * (1 - t / T), size=dim), -B, B)
            fg = R.mse(R.forward(X_fit, gp, d_in), y_fit)
            if fg < g_fit:
                g, g_fit, stag = gp, fg, 0
        if stag >= R.STAG_LIMIT or rho < R.RHO_MIN:
            k = max(1, int(R.K_RESET * N))
            wi = np.argsort(pbest_fit)[-k:]
            X[wi] = (R.tent_sequence(rng, k, dim) * 2.0 - 1.0) * B
            V[wi] = rng.uniform(-B, B, size=(k, dim))
            fit[wi] = [R.mse(R.forward(X_fit, x, d_in), y_fit) for x in X[wi]]
            pbest[wi], pbest_fit[wi] = X[wi], fit[wi]
            stag = 0
    return np.array(trace)


def scale_invariance(out_dir, seeds=5):
    """验证 P1：x -> c*x 时 rho(t) 与 w(t) 轨迹不变。

    注意：仅当 rho 不变时 w 才不变；而 d(t) 与 d(0) 在缩放下同乘 c，
    故 rho = d(t)/d(0) 严格不变 —— 但 V_CLIP 与 B 的比例会影响轨迹，
    因此这里固定速度钳制为相对形式 V_CLIP = 0.5*B（与主实验 B=0.5 时一致）。
    """
    ds = R.load_all_datasets()
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for name, (X, y) in ds.items():
        d_in = X.shape[1]
        traces = {}
        for sc in B_SCALES:
            s0 = 3000
            Xtr, ytr, Xva, yva, _, _ = R.split_standardize(X, y, s0)
            X_fit = np.vstack([Xtr, Xva])
            y_fit = np.concatenate([ytr, yva])
            old_B, old_VC = R.B, R.V_CLIP
            R.B = 0.5 * sc
            R.V_CLIP = 0.5 * R.B          # 相对钳制，保证尺度等变
            try:
                allw = np.array([inertia_trace(R.B, X_fit, y_fit, d_in, 3000 + s)
                                 for s in range(seeds)])
            finally:
                R.B, R.V_CLIP = old_B, old_VC
            traces[sc] = allw.mean(axis=0)
        ref = traces[1.0]
        maxdev = {sc: float(np.max(np.abs(traces[sc] - ref))) for sc in B_SCALES}
        report[name] = dict(max_dev=maxdev,
                            mean_abs_w=float(ref.mean()),
                            dim=d_in * R.HIDDEN + R.HIDDEN + R.HIDDEN + 1)
        print(f"  {name:10s} dim={report[name]['dim']:4d} "
              f"max|w(B)-w(B0)| = " +
              ", ".join(f"B×{k}:{v:.2e}" for k, v in maxdev.items()))
    with open(os.path.join(out_dir, "scale_invariance.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    worst = max(max(r["max_dev"].values()) for r in report.values())
    print(f"[scale_invariance] 全部数据集上的最大轨迹偏差 = {worst:.3e}")
    return report


# ---------------------------------------------------------------------------
# 干净消融批量运行
# ---------------------------------------------------------------------------


def ablation_no_obl(out_csv, seeds=30):
    ds = R.load_all_datasets()
    rows = []
    total = len(ds) * seeds
    done = 0
    t0 = time.time()
    for name, (X, y) in ds.items():
        d_in = X.shape[1]
        for s in range(seeds):
            rng = np.random.default_rng(s)
            Xtr, ytr, Xva, yva, Xte, yte = R.split_standardize(X, y, s)
            acc, f1, auc, bal, g_fit, w_trace, curve = run_no_obl(
                Xtr, ytr, Xva, yva, Xte, yte, d_in, s, rng)
            rows.append(dict(dataset=name, method="CDPSO w/o OBL",
                             seed=3000 + s, accuracy=acc, f1=f1, auc=auc,
                             bal_acc=bal, search_fitness=g_fit))
            done += 1
        print(f"  {name} 完成 ({done}/{total})  {time.time()-t0:.0f}s")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"[ablation_no_obl] {len(rows)} 行 -> {out_csv}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    RES = "results"
    os.makedirs(RES, exist_ok=True)
    if cmd in ("all", "curve"):
        budget_curve(os.path.join(RES, "budget_curve.csv"))
    if cmd in ("all", "noobl"):
        ablation_no_obl(os.path.join(RES, "ablation_no_obl.csv"))
    if cmd in ("all", "scale"):
        scale_invariance(os.path.join(RES, "scale"))
