# -*- coding: utf-8 -*-
"""
run_experiment.py —— CDPSO-BP 真实可复现实验脚本（数据驱动）
================================================================
严格按论文《融合混沌反向学习与多样性反馈的改进粒子群算法优化BP神经网络》§3 方法定义
与 §4.2 实验设置实现，在公开数据集上真跑 30 个独立种子，产出逐种子真实结果，
供 exp_harness.py 统计（表3显著性 / 表4消融 / 表5基线 / 表6敏感性）。

关键设计（与论文一致）：
  * 三层前馈网络：输入层 - 8 隐层节点(tanh) - 1 输出(sigmoid)；MSE 损失；全批梯度下降 + 动量。
  * 数据划分：70% 训练 / 10% 验证 / 20% 测试；用训练集统计量标准化。
  * PSO 系列：搜索阶段适应度在 train+val 合并(80%)上算 MSE；再固定 30 epoch 微调(train)。
  * BP(充分)：随机初始化 + 300 epoch + 早停(验证损失连续 30 epoch 无改善)。
  * BP-30：随机初始化 + 固定 30 epoch。
  * 搜索参数：N=20, T=40, c1=c2=2, b=0.5, 速度钳制[-0.25,0.25], 标准PSO w=0.8,
              IPSO/CDPSO 基准 w∈[0.9,0.4] 二次衰减, p_m=0.2, sigma0=0.2, 停滞阈值 3 代, k=0.2。
  * 微调：lr=0.05, momentum=0.9。

权值维度（验证过与论文一致）：
  * Heart: 19 特征 -> 19*8 + 8 + 8*1 + 1 = 169 维
  * LoL  : 38 特征 -> 38*8 + 8 + 8*1 + 1 = 321 维

输出：results/raw_results.csv（source=real_experiment）
用法：
  python run_experiment.py --methods main            # 5 主方法(表3)
  python run_experiment.py --methods ablation        # 6 消融变体(表4)
  python run_experiment.py --methods main,ablation   # 全跑(表3+表4)
  python run_experiment.py --seeds 3 --methods main  # 快速验证(3种子)
"""

import os
import csv
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)
RAW_PATH = os.path.join(OUT_DIR, "raw_results.csv")

HIDDEN = 8            # 隐层节点数
LR = 0.05             # 微调学习率
MOMENTUM = 0.9        # 微调动量

# PSO 搜索参数（论文 §4.2）
N_POP = 20
T_MAX = 40
C1 = C2 = 2.0
B = 0.5               # 搜索边界 [-b, b]
V_CLIP = 0.25         # 速度钳制
W_STD = 0.8           # 标准 PSO 固定惯性权重
W_MAX = 0.9
W_MIN = 0.4
P_M = 0.2             # 高斯扰动概率
SIGMA0 = 0.2          # 初始扰动强度
STAG_LIMIT = 3        # 停滞阈值（代）
K_RESET = 0.2         # 停滞重启重置比例


# ----------------------------------------------------------------------------
# 数据加载与预处理
# ----------------------------------------------------------------------------
def load_heart(path):
    """Heart Disease (Kaggle heart.csv, 918 样本)。类别变量独热编码 -> 19 特征。
    编码规则（可复现）：数值 6 个标准化；Sex 独热 2 列；ChestPainType 独热 4 列；
    RestingECG 独热 3 列；ExerciseAngina 0/1；ST_Slope 独热 3 列。合计 19 特征。"""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(d)
    X_num = np.array([[float(d["Age"]), float(d["RestingBP"]), float(d["Cholesterol"]),
                       float(d["FastingBS"]), float(d["MaxHR"]), float(d["Oldpeak"])]
                      for d in rows], float)
    y = np.array([1 if d["HeartDisease"] in ("1", "1.0", 1) else 0 for d in rows], float)

    def onehot(vals, cats):
        return np.array([[1.0 if v == c else 0.0 for c in cats] for v in vals], float)

    X_cat = np.hstack([
        onehot([d["Sex"] for d in rows], ["M", "F"]),
        onehot([d["ChestPainType"] for d in rows], ["ATA", "NAP", "ASY", "TA"]),
        onehot([d["RestingECG"] for d in rows], ["Normal", "ST", "LVH"]),
        np.array([[1.0 if d["ExerciseAngina"] == "Y" else 0.0] for d in rows], float),
        onehot([d["ST_Slope"] for d in rows], ["Up", "Flat", "Down"]),
    ])
    X = np.hstack([X_num, X_cat])
    return X.astype(float), y


def load_lol(path):
    """LoL (high_diamond_ranked_10min.csv, 9879 样本)。去掉 gameId 与目标 blueWins -> 38 特征。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = list(r)
    feat = [c for c in rows[0].keys() if c not in ("gameId", "blueWins")]
    X = np.array([[float(d[c]) for c in feat] for d in rows], float)
    y = np.array([1.0 if d["blueWins"] in ("1", "1.0", 1) else 0.0 for d in rows], float)
    return X, y


def split_standardize(X, y, seed, train_ratio=0.7, val_ratio=0.1):
    """随机 shuffle 划分 70/10/20，并用训练集统计量标准化。返回 (Xtr,ytr,Xva,yva,Xte,yte)。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    itr, iva, ite = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    Xtr, ytr = X[itr], y[itr]
    Xva, yva = X[iva], y[iva]
    Xte, yte = X[ite], y[ite]
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    return Xtr, ytr, Xva, yva, Xte, yte


# ----------------------------------------------------------------------------
# BP 网络前向 / 梯度
# ----------------------------------------------------------------------------
def decode(w, d_in):
    """权值向量 -> (W1, b1, W2, b2)。W1:(8,d_in) b1:(8,) W2:(1,8) b2:(1,)"""
    d_hid = HIDDEN
    n1 = d_in * d_hid
    W1 = w[:n1].reshape(d_hid, d_in)
    b1 = w[n1:n1 + d_hid]
    W2 = w[n1 + d_hid:n1 + d_hid + d_hid].reshape(1, d_hid)
    b2 = w[n1 + d_hid + d_hid:n1 + d_hid + d_hid + 1]
    return W1, b1, W2, b2


def forward(X, w, d_in):
    W1, b1, W2, b2 = decode(w, d_in)
    z1 = X @ W1.T + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2.T + b2
    yh = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))
    return yh.reshape(-1)


def mse(yh, y):
    return float(np.mean((yh - y) ** 2))


def bp_finetune(w0, X, y, d_in, epochs, lr=LR, momentum=MOMENTUM, Xva=None, yva=None,
                early_stop=False, es_patience=30):
    """全批梯度下降 + 动量。返回最终权值向量。若 early_stop 且给 Xva，则按验证损失早停。"""
    w = w0.copy()
    v = np.zeros_like(w)
    best_w = w.copy()
    best_va = float("inf")
    no_improve = 0
    n = X.shape[0]
    for _ in range(epochs):
        W1, b1, W2, b2 = decode(w, d_in)
        z1 = X @ W1.T + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2.T + b2
        yh = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))
        yh = yh.reshape(-1)
        # MSE 反向（sigmoid 输出 + tanh 隐层）
        dL = 2.0 * (yh - y) / n                     # dL/dyh
        delta2 = (dL * yh * (1.0 - yh)).reshape(-1, 1)   # (n,1)
        gW2 = (delta2.T @ a1)                        # (1,8)
        gb2 = delta2.sum(axis=0)                     # (1,)
        delta1 = (delta2 @ W2) * (1.0 - a1 ** 2)     # (n,8)
        gW1 = delta1.T @ X                           # (8,d_in)
        gb1 = delta1.sum(axis=0)                     # (8,)
        grad = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
        v = momentum * v - lr * grad
        w = w + v
        if early_stop and Xva is not None:
            lv = mse(forward(Xva, w, d_in), yva)
            if lv < best_va:
                best_va = lv
                best_w = w.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= es_patience:
                    return best_w
    return best_w if early_stop and Xva is not None else w


def evaluate(w, Xte, yte, d_in):
    yh = forward(Xte, w, d_in)
    pred = (yh >= 0.5).astype(int)
    yte = yte.astype(int)
    acc = float(np.mean(pred == yte))
    tp = float(np.sum((pred == 1) & (yte == 1)))
    fp = float(np.sum((pred == 1) & (yte == 0)))
    fn = float(np.sum((pred == 0) & (yte == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, f1


# ----------------------------------------------------------------------------
# PSO 初始化工具
# ----------------------------------------------------------------------------
def tent_sequence(rng, n, dim):
    """Tent 映射生成 [0,1] 内混沌序列，避免不动点 x=0 与 x=2/3。返回 (n,dim) 数组。"""
    out = np.zeros((n, dim))
    for i in range(n):
        x = rng.uniform(0, 1)
        for j in range(dim):
            if abs(x) < 0.01 or abs(x - 2 / 3) < 0.01:
                x = rng.uniform(0, 1)
            x = 1.0 - 2.0 * abs(x - 0.5)
            out[i, j] = x
    return out


def init_standard(rng, n, dim):
    """标准 PSO：位置/速度均匀随机。"""
    X = rng.uniform(-B, B, size=(n, dim))
    V = rng.uniform(-B, B, size=(n, dim))
    return X, V


def init_ipso(rng, n, dim):
    """IPSO：仅 Tent 混沌初始化（无反向学习）。速度均匀随机。"""
    X = (tent_sequence(rng, n, dim) * 2.0 - 1.0) * B  # [0,1]->[-b,b]
    V = rng.uniform(-B, B, size=(n, dim))
    return X, V


def init_cdpso(rng, n, dim):
    """CDPSO：Tent 混沌 N 个 + 反向解 N 个 -> 2N 候选择优 N。反向解 x'=a+b-x=-x（边界对称）。"""
    Xc = (tent_sequence(rng, n, dim) * 2.0 - 1.0) * B
    Xo = -Xc  # 反向解（a+b-x, a=-b,b=b -> -x）
    cand = np.vstack([Xc, Xo])  # (2n, dim)
    # 择优：无法预知最优，采用"混沌解与反向解交错择优"——为保证确定性，评估后取前 N（在搜索主循环内按适应度择优）
    # 这里返回 2N 候选，由调用方按适应度择优
    X = cand.copy()
    V = rng.uniform(-B, B, size=(X.shape[0], dim))
    return X, V, n  # 返回候选数与目标规模


# ----------------------------------------------------------------------------
# PSO 搜索主循环（统一框架，参数化变体）
# ----------------------------------------------------------------------------
def pso_search(X_fit, y_fit, d_in, seed, mode, rng=None):
    """
    mode 取值:
      pso                标准 PSO（w=0.8 固定，随机初始化）
      ipso               IPSO（Tent 混沌初始化 + 时间表二次衰减权重，ρ≡1）
      ipso_obl           IPSO+OBL（IPSO 基础上加反向学习初始化）
      cdpso_full         CDPSO(完整)：Tent+OBL + 多样性反馈权重 + 高斯扰动 + 停滞重启
      cdpso_no_restart   CDPSO 去掉停滞重启
      cdpso_no_perturb   CDPSO 去掉高斯扰动
      cdpso_no_diversity CDPSO 去掉多样性反馈（ρ≡1）
    返回 (gbest 权值向量, gbest 末适应度)。
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    dim = d_in * HIDDEN + HIDDEN + HIDDEN + 1

    use_obl = mode in ("cdpso_full", "cdpso_no_restart", "cdpso_no_perturb",
                       "cdpso_no_diversity", "ipso_obl")
    use_chaos = mode != "pso"
    use_diversity = mode in ("cdpso_full", "cdpso_no_restart", "cdpso_no_perturb")
    use_perturb = mode in ("cdpso_full", "cdpso_no_restart", "cdpso_no_diversity")
    use_restart = mode in ("cdpso_full", "cdpso_no_perturb", "cdpso_no_diversity")
    use_schedule_w = mode != "pso"  # 非标准 PSO 均用时间表递减权重基准

    if use_obl:
        Xc = (tent_sequence(rng, N_POP, dim) * 2.0 - 1.0) * B
        cand = np.vstack([Xc, -Xc])  # 2N 候选
        fits = np.array([mse(forward(X_fit, c, d_in), y_fit) for c in cand])
        order = np.argsort(fits)[:N_POP]  # 择优 N（MSE 越小越好）
        X = cand[order]
        V = rng.uniform(-B, B, size=(N_POP, dim))
    elif use_chaos:
        X = (tent_sequence(rng, N_POP, dim) * 2.0 - 1.0) * B
        V = rng.uniform(-B, B, size=(N_POP, dim))
    else:
        X = rng.uniform(-B, B, size=(N_POP, dim))
        V = rng.uniform(-B, B, size=(N_POP, dim))

    # 初始适应度
    fit = np.array([mse(forward(X_fit, x, d_in), y_fit) for x in X])
    pbest = X.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    g = pbest[g_idx].copy()
    g_fit = pbest_fit[g_idx]

    # 初始多样性参考 d(0)
    center0 = X.mean(axis=0)
    d0 = float(np.mean(np.linalg.norm(X - center0, axis=1)))
    d0 = d0 if d0 > 1e-12 else 1.0

    stag = 0
    for t in range(1, T_MAX + 1):
        # 多样性反馈权重
        if use_schedule_w:
            w_base = W_MIN + (W_MAX - W_MIN) * (1.0 - t / T_MAX) ** 2
            if use_diversity:
                center = X.mean(axis=0)
                d_t = float(np.mean(np.linalg.norm(X - center, axis=1)))
                rho = min(d_t / d0, 1.0)
                w = w_base * (1.0 + rho) / 2.0
            else:
                w = w_base
        else:
            w = W_STD

        # 速度/位置更新
        r1 = rng.uniform(0, 1, size=(N_POP, dim))
        r2 = rng.uniform(0, 1, size=(N_POP, dim))
        V = w * V + C1 * r1 * (pbest - X) + C2 * r2 * (g - X)
        V = np.clip(V, -V_CLIP, V_CLIP)
        X = np.clip(X + V, -B, B)

        # 重评估
        fit = np.array([mse(forward(X_fit, x, d_in), y_fit) for x in X])
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

        # 高斯扰动（贪心接受）
        if use_perturb and rng.uniform(0, 1) < P_M:
            sigma = SIGMA0 * (1.0 - t / T_MAX)
            gp = g + rng.normal(0, sigma, size=dim)
            gp = np.clip(gp, -B, B)
            fgp = mse(forward(X_fit, gp, d_in), y_fit)
            if fgp < g_fit:
                g = gp
                g_fit = fgp
                stag = 0

        # 停滞重启
        if use_restart and stag >= STAG_LIMIT:
            n_reset = max(1, int(K_RESET * N_POP))
            worst_idx = np.argsort(pbest_fit)[-n_reset:]
            X[worst_idx] = (tent_sequence(rng, n_reset, dim) * 2.0 - 1.0) * B
            V[worst_idx] = rng.uniform(-B, B, size=(n_reset, dim))
            fit[worst_idx] = [mse(forward(X_fit, x, d_in), y_fit) for x in X[worst_idx]]
            pbest[worst_idx] = X[worst_idx]
            pbest_fit[worst_idx] = fit[worst_idx]
            stag = 0

    return g, g_fit


# ----------------------------------------------------------------------------
# 方法主流程：返回 (acc, f1, search_fitness)
# ----------------------------------------------------------------------------
def run_method(Xtr, ytr, Xva, yva, Xte, yte, d_in, method, seed, rng):
    if method == "BP":
        w0 = rng.uniform(-B, B, size=d_in * HIDDEN + HIDDEN + HIDDEN + 1)
        w = bp_finetune(w0, Xtr, ytr, d_in, epochs=300, Xva=Xva, yva=yva, early_stop=True)
        acc, f1 = evaluate(w, Xte, yte, d_in)
        return acc, f1, float("nan")
    if method == "BP-30":
        w0 = rng.uniform(-B, B, size=d_in * HIDDEN + HIDDEN + HIDDEN + 1)
        w = bp_finetune(w0, Xtr, ytr, d_in, epochs=30)
        acc, f1 = evaluate(w, Xte, yte, d_in)
        return acc, f1, float("nan")
    # PSO 系列：搜索(train+val 80%) + 30 epoch 微调(train)
    X_fit = np.vstack([Xtr, Xva])
    y_fit = np.concatenate([ytr, yva])
    mode_map = {
        "PSO-BP": "pso", "IPSO-BP": "ipso", "CDPSO-BP": "cdpso_full",
        "IPSO": "ipso", "IPSO+OBL": "ipso_obl",
        "CDPSO-无重启": "cdpso_no_restart", "CDPSO-无扰动": "cdpso_no_perturb",
        "CDPSO-无多样性": "cdpso_no_diversity", "CDPSO(完整)": "cdpso_full",
    }
    g, g_fit = pso_search(X_fit, y_fit, d_in, seed, mode_map[method], rng=rng)
    w = bp_finetune(g, Xtr, ytr, d_in, epochs=30)
    acc, f1 = evaluate(w, Xte, yte, d_in)
    return acc, f1, g_fit


# ----------------------------------------------------------------------------
# 数据集注册
# ----------------------------------------------------------------------------
def get_datasets():
    ds = {}
    heart_path = os.path.join(HERE, "..", "..", "..", "tmp_heart.csv")
    # 数据文件放在脚本同目录 data/ 下
    data_dir = os.path.join(HERE, "data")
    heart_candidates = [
        os.path.join(data_dir, "heart.csv"),
        os.path.join(HERE, "heart.csv"),
    ]
    lol_candidates = [
        os.path.join(data_dir, "lol.csv"),
        os.path.join(HERE, "lol.csv"),
    ]
    return heart_candidates, lol_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="main,ablation",
                    help="main / ablation / main,ablation")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--datasets", default="LoL,Heart")
    args = ap.parse_args()

    heart_cands, lol_cands = get_datasets()
    heart_path = next((p for p in heart_cands if os.path.exists(p)), None)
    lol_path = next((p for p in lol_cands if os.path.exists(p)), None)
    if heart_path is None or lol_path is None:
        print("[错误] 找不到数据文件。请把 heart.csv 与 lol.csv 放到脚本同目录或 data/ 子目录。")
        print("  heart 候选:", heart_cands)
        print("  lol   候选:", lol_cands)
        return

    Xh, yh = load_heart(heart_path)
    Xl, yl = load_lol(lol_path)
    datasets = {"Heart": (Xh, yh), "LoL": (Xl, yl)}

    methods_main = ["BP", "BP-30", "PSO-BP", "IPSO-BP", "CDPSO-BP"]
    methods_ablation = ["IPSO", "IPSO+OBL", "CDPSO-无重启", "CDPSO-无扰动",
                        "CDPSO-无多样性", "CDPSO(完整)"]
    sel = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = []
    if "main" in sel:
        methods += methods_main
    if "ablation" in sel:
        methods += methods_ablation
    if not methods:
        methods = methods_main + methods_ablation

    n_seeds = args.seeds
    rows = []
    total = len(datasets) * len(methods) * n_seeds
    done = 0
    for ds_name in args.datasets.split(","):
        ds_name = ds_name.strip()
        if ds_name not in datasets:
            continue
        X, y = datasets[ds_name]
        d_in = X.shape[1]
        print(f"=== 数据集 {ds_name}: {X.shape[0]} 样本, {d_in} 特征, "
              f"权值维度 {d_in*HIDDEN+HIDDEN+HIDDEN+1} ===")
        for method in methods:
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                Xtr, ytr, Xva, yva, Xte, yte = split_standardize(X, y, seed)
                acc, f1, sf = run_method(Xtr, ytr, Xva, yva, Xte, yte,
                                         d_in, method, seed, rng)
                rows.append([ds_name, method, seed, f"{acc:.6f}", f"{f1:.6f}",
                             ("" if sf != sf else f"{sf:.6f}"), "real_experiment"])
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  进度 {done}/{total}  [{method}@{ds_name} seed{seed} "
                          f"acc={acc:.4f} f1={f1:.4f}]")

    # 写 CSV（保留表头注释）
    lines = ["# raw_results.csv —— exp_harness.py 输入（逐种子真实结果）",
             "# 本文件由 run_experiment.py 真实运行生成（source=real_experiment）。",
             "# 列: dataset,method,seed,accuracy,f1,search_fitness,source"]
    with open(RAW_PATH, "w", newline="", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")
        w = csv.writer(f)
        w.writerow(["dataset", "method", "seed", "accuracy", "f1", "search_fitness", "source"])
        for r in rows:
            w.writerow(r)
    print(f"\n完成：{len(rows)} 条真实结果已写入 {RAW_PATH}")
    print("下一步：运行 exp_harness.py 生成表3/4/5/6。")


if __name__ == "__main__":
    main()
