# -*- coding: utf-8 -*-
"""
exp_harness.py v2 —— 面向 Applied Soft Computing (Q1) 的“数据驱动”实验与统计管线
=================================================================================
关键变化（v2）：
  * 默认模式（无参数）= 真实数据模式：读取 results/raw_results.csv（你导出的逐种子结果），
    自动产出 Q1 级统计：mean±std、95%CI、Friedman + Holm 事后校正、Wilcoxon、Cliff's δ。
  * 对“从未运行过的实验”（消融变体 / ≥10 基线 / 参数敏感性）明确标 【待补】，绝不编造数字。
  * --demo  = 用合成数据填满所有方法/数据集，演示最终表格的“完整格式”（标注 SYNTHETIC）。
  * --init-template = 生成 results/raw_results.csv 模板：把稿件表2的“真实均值±标准差”以
    “重建种子（精确复现 mean±std）”方式填入并打上 source 标签，同时预留消融/基线/敏感性空行。

诚信声明：
  * raw_results.csv 里 5 主方法×2 数据集的 50 行是“从稿件汇报的均值±标准差重建”的演示种子，
    并非你的原始实验导出；它们能精确复现稿件表2 的数字与排名，用于预览表3 结构。
    在正式投稿前，请用你真实的逐种子导出替换这些行，并补齐消融/基线/敏感性实验。
  * 凡未运行过的实验，输出一律为【待补】，请勿当作真实结果。

CSV 格式（results/raw_results.csv）：
  dataset,method,seed,accuracy,f1,search_fitness,source
  - dataset: "LoL"(电竞) / "Heart"(心脏病) / 或你扩展的 ≥10 个数据集名
  - method : "CDPSO-BP"(对照) / "IPSO-BP" / "PSO-BP" / "BP" / "BP-30" 等主方法，
             消融 "IPSO" / "IPSO+OBL" / "CDPSO-无重启" / "CDPSO-无扰动" / "CDPSO-无多样性" / "CDPSO(完整)"，
             基线 "CLPSO" / "APSO" / "AMPSO" / "DMPSO" / "GLPSO" / "SHADE" / "jSO" / "CMA-ES" / "Xavier+Adam" / "Kaiming+Adam"
  - seed   : 整数（Q1 要求 ≥30 个独立种子，跨方法配对可比）
  - accuracy,f1 : 测试集指标（0~1）；search_fitness : PSO 搜索阶段末适应度（无则留空）
  - source  : "real_experiment" / "reconstructed_from_manuscript" / "manuscript_mean"

输出（results/ 下）：
  table3_significance.csv / table4_ablation.csv / table5_baselines.csv /
  table6_sensitivity.csv / summary.json
"""

import os
import csv
import json
import argparse
import numpy as np
from scipy import stats

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/results"
os.makedirs(OUT_DIR, exist_ok=True)

CONTROL = "CDPSO-BP"          # Holm 事后校正以本文方法为对照
MAIN_METHODS = ["CDPSO-BP", "IPSO-BP", "PSO-BP", "BP", "BP-30"]          # 表3
ABLATION_METHODS = ["IPSO", "IPSO+OBL", "CDPSO-无重启", "CDPSO-无扰动",
                    "CDPSO-无多样性", "CDPSO(完整)"]                      # 表4
BASELINE_METHODS = ["CLPSO", "APSO", "AMPSO", "DMPSO", "GLPSO", "SHADE",
                    "jSO", "CMA-ES", "Xavier+Adam", "Kaiming+Adam"]      # 表5

# 数据集顺序（新增 10 个公开二分类数据集后共 12 个；Heart/LoL 前置，其余按字母序）
DATASET_ORDER = ["Heart", "LoL", "Pima", "WDBC", "Ionosphere", "Sonar", "Banknote",
                 "Spambase", "Haberman", "Parkinson", "Blood", "Phoneme"]


def detect_datasets(data):
    """从 raw_results.csv 中自动识别出现的数据集；按 DATASET_ORDER 优先排序。"""
    present = set(ds for (ds, m) in data.keys())
    ordered = [d for d in DATASET_ORDER if d in present]
    rest = sorted(present - set(DATASET_ORDER))
    return ordered + rest

# 稿件表2 真实均值±标准差（用于 --init-template 重建种子）；accuracy / f1 分别给
MANUSCRIPT_TABLE2 = {
    "LoL": {  # 电竞
        "BP":        (0.7307, 0.0053, 0.7310, 0.0074),
        "BP-30":     (0.7269, 0.0039, 0.7269, 0.0049),
        "PSO-BP":    (0.7277, 0.0048, 0.7250, 0.0053),
        "IPSO-BP":   (0.7281, 0.0074, 0.7273, 0.0095),
        "CDPSO-BP":  (0.7286, 0.0059, 0.7293, 0.0063),
    },
    "Heart": {  # 心脏病
        "BP":        (0.8545, 0.0076, 0.8699, 0.0068),
        "BP-30":     (0.8589, 0.0144, 0.8727, 0.0142),
        "PSO-BP":    (0.8473, 0.0115, 0.8615, 0.0097),
        "IPSO-BP":   (0.8509, 0.0164, 0.8643, 0.0131),
        "CDPSO-BP":  (0.8560, 0.0075, 0.8706, 0.0093),
    },
}
# 稿件表1 搜索阶段末适应度（均值，无 std）—— PSO-BP/IPSO-BP/CDPSO-BP 对应 PSO/IPSO/CDPSO 搜索
MANUSCRIPT_TABLE1 = {
    "LoL":   {"PSO-BP": 0.1842, "IPSO-BP": 0.1828, "CDPSO-BP": 0.1829},
    "Heart": {"PSO-BP": 0.0987, "IPSO-BP": 0.0988, "CDPSO-BP": 0.0981},
}
# 重建种子的归一化形状（样本标准差=1，均值=0）：精确复现 mean±std
E_SHAPE = [-1.264911, -0.632456, 0.0, 0.632456, 1.264911]


# ----------------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------------
def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(1.96 * se)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 1 or len(b) < 1:
        return float("nan")
    gt = int((a[:, None] > b[None, :]).sum())
    lt = int((a[:, None] < b[None, :]).sum())
    return float((gt - lt) / (len(a) * len(b)))


def friedman_and_holm(results: dict, control: str, alpha: float = 0.05):
    """results: {method: np.array(种子数,)}。返回 (friedman_p, holm_table)。
    holm_table: list of dict(method, raw_p, holm_p, sig)。控制方法 sig=False。"""
    methods = list(results.keys())
    ctrl_arr = results.get(control)
    # Friedman 需 ≥3 个方法且等长
    valid = {m: np.asarray(v, float) for m, v in results.items()
             if len(np.asarray(v, float)) >= 2}
    if len(valid) >= 3 and ctrl_arr is not None and len(ctrl_arr) >= 2:
        try:
            _, f_p = stats.friedmanchisquare(*[valid[m] for m in valid])
        except Exception:
            f_p = float("nan")
    else:
        f_p = float("nan")  # n/a（<3 方法或对照缺失）

    holm = []
    comp = []
    for m in methods:
        if m == control:
            holm.append({"method": m, "raw_p": float("nan"), "holm_p": float("nan"), "sig": False})
            continue
        a = results.get(m)
        if ctrl_arr is None or a is None or len(a) < 2 or len(ctrl_arr) < 2:
            holm.append({"method": m, "raw_p": float("nan"), "holm_p": float("nan"), "sig": False})
            continue
        try:
            _, p = stats.wilcoxon(ctrl_arr, a, alternative="two-sided")
        except Exception:
            p = float("nan")
        comp.append((m, p))
    comp.sort(key=lambda t: (t[1] if t[1] == t[1] else 9))
    m_comp = len(comp)
    for i, (m, p) in enumerate(comp):
        hp = min(1.0, p * (m_comp - i)) if p == p else float("nan")
        holm.append({"method": m, "raw_p": p, "holm_p": hp, "sig": (hp < alpha)})
    return f_p, holm


def sig_arrow(ctrl: np.ndarray, arr: np.ndarray) -> str:
    if ctrl is None or arr is None or len(ctrl) < 2 or len(arr) < 2:
        return "≈"
    d = cliffs_delta(ctrl, arr)
    if d > 0.147:
        return "↑"   # 对照优于该方法
    if d < -0.147:
        return "↓"   # 该方法优于对照
    return "≈"


# ----------------------------------------------------------------------------
# 读取真实数据
# ----------------------------------------------------------------------------
def load_real(path: str):
    data = {}  # (dataset, method) -> list of (seed, acc, f1, sf, source)
    if not os.path.exists(path):
        return data
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        ds = row["dataset"].strip()
        m = row["method"].strip()
        try:
            seed = int(row["seed"])
        except Exception:
            continue
        acc = float(row["accuracy"]) if row.get("accuracy", "").strip() not in ("", "nan") else float("nan")
        f1 = float(row["f1"]) if row.get("f1", "").strip() not in ("", "nan") else float("nan")
        sf = float(row["search_fitness"]) if row.get("search_fitness", "").strip() not in ("", "nan") else float("nan")
        src = row.get("source", "real_experiment").strip()
        data.setdefault((ds, m), []).append((seed, acc, f1, sf, src))
    return data


def gather(data, datasets, methods):
    """返回 {dataset: {method: np.array(acc)}} 等仅含存在且有≥2种子的。"""
    out = {}
    info = {}  # (ds,m)->source
    for ds in datasets:
        present = {}
        for m in methods:
            rows = data.get((ds, m), [])
            acc = np.array([r[1] for r in rows if r[1] == r[1]], float)  # 去 nan
            if len(acc) >= 2:
                present[m] = acc
                info[(ds, m)] = rows[0][4]
        if present:
            out[ds] = present
    return out, info


# ----------------------------------------------------------------------------
# 表格生成
# ----------------------------------------------------------------------------
def write_table(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def build_table3(data):
    """显著性检验（主方法 CDPSO-BP 对照）。有真实数据算，否则待补。"""
    datasets = detect_datasets(data)
    out, info = gather(data, datasets, MAIN_METHODS)
    rows = []
    header = ["dataset", "metric", "method", "mean", "std", "ci95(±)", "friedman_p",
              "holm_p", "cliffs_delta", "显著性(↑/↓/≈)", "data_source", "status"]
    for ds in datasets:
        present = out.get(ds, {})
        if CONTROL not in present or len(present) < 2:
            rows.append([ds, "ACC/F1", "—", "—", "—", "—", "—", "—", "—", "—", "—",
                         "【待补：对照或方法数据不足（需≥30种子/≥2方法）】"])
            continue
        f_p, holm = friedman_and_holm(present, CONTROL)
        for metric, key in [("ACC", "accuracy"), ("F1", "f1")]:
            pass
        # 逐指标（ACC 用 accuracy，F1 用 f1）
        for metric_idx, (metric, col) in enumerate([("ACC", 1), ("F1", 2)]):
            # 重组该指标数组
            pres_m = {}
            for m, arr in present.items():
                rows_m = [r for r in data.get((ds, m), []) if r[col] == r[col]]
                arr = np.array([r[col] for r in rows_m], float)
                if len(arr) >= 2:
                    pres_m[m] = arr
            if CONTROL not in pres_m:
                continue
            f_pm, holm_m = friedman_and_holm(pres_m, CONTROL)
            ctrl = pres_m[CONTROL]
            for m in MAIN_METHODS:
                if m not in pres_m:
                    rows.append([ds, metric, m, "—", "—", "—", "—", "—", "—", "—",
                                 info.get((ds, m), "—"), "【待补：该方法该数据集无≥2种子】"])
                    continue
                x = pres_m[m]
                ar = sig_arrow(ctrl, x)
                h = next((h for h in holm_m if h["method"] == m), None)
                raw = h["raw_p"] if h else float("nan")
                hp = h["holm_p"] if h else float("nan")
                cd = cliffs_delta(ctrl, x)
                rows.append([ds, metric, m, f"{x.mean():.4f}", f"{x.std(ddof=1):.4f}",
                             f"{ci95(x):.4f}",
                             (f"{f_pm:.4g}" if f_pm == f_pm else "n/a(<3方法)"),
                             (f"{hp:.4g}" if hp == hp else "—"),
                             f"{cd:+.3f}", ar, info.get((ds, m), "—"),
                             "OK" if h and h["sig"] else "n.s."])
    write_table(OUT_DIR + "/table3_significance.csv", header, rows)
    return rows


def _paired(data, ds, m, ref):
    """按 seed 对齐取出 (m, ref) 在同一数据集上的逐种子 ACC。"""
    da = {r[0]: r[1] for r in data.get((ds, m), [])}
    db = {r[0]: r[1] for r in data.get((ds, ref), [])}
    seeds = sorted(set(da) & set(db))
    if len(seeds) < 2:
        return np.array([]), np.array([])
    return np.array([da[s] for s in seeds]), np.array([db[s] for s in seeds])


def _verdict(d_pp, p, cd):
    if p == p and p < 0.05:
        return f"{d_pp:+.2f}pp (p={p:.3g}, {'\u2191\u4f18' if d_pp > 0 else '\u2193\u52a3'})"
    if p == p:
        return f"{d_pp:+.2f}pp (p={p:.3g}, \u2248)"
    return f"{d_pp:+.2f}pp (p=n/a, \u2248)"


def ablation_verdict(data, ds, m, ref="CDPSO(\u5b8c\u6574)"):
    """逐数据集：消融变体 vs 完整模型 的 Wilcoxon 配对检验结论。"""
    if m == ref:
        return "\u53c2\u7167\uff08\u5b8c\u6574\uff09"
    x, y = _paired(data, ds, m, ref)
    if len(x) < 5:
        return "\u6837\u672c\u4e0d\u8db3"
    d = x - y
    try:
        p = float(stats.wilcoxon(x, y)[1])
    except Exception:
        p = float("nan")
    return _verdict(100 * float(d.mean()), p, cliffs_delta(x, y))


def build_table_cross_tests(data):
    """跨数据集汇总配对检验：把 12 个数据集 × 30 种子的配对差 pooling 后做 Wilcoxon。"""
    datasets = detect_datasets(data)
    pairs = [
        ("CDPSO-BP", "BP-30", "主"),
        ("CDPSO-BP", "BP", "主"),
        ("CDPSO-BP", "PSO-BP", "主"),
        ("CDPSO-BP", "IPSO-BP", "主"),
        ("CDPSO(\u5b8c\u6574)", "CDPSO-\u65e0\u591a\u6837\u6027", "消融"),
        ("CDPSO(\u5b8c\u6574)", "CDPSO-\u65e0\u6270\u52a8", "消融"),
        ("CDPSO(\u5b8c\u6574)", "CDPSO-\u65e0\u91cd\u542f", "消融"),
        ("CDPSO(\u5b8c\u6574)", "IPSO+OBL", "消融"),
        ("CDPSO(\u5b8c\u6574)", "IPSO", "消融"),
    ]
    header = ["对照组", "处理组", "类别", "n(配对)", "平均差(pp)", "95%CI(pp)",
              "Wilcoxon p", "Cliff's delta", "逐数据集 胜/平/负", "结论"]
    rows = []
    for a, b, cat in pairs:
        diffs, per_ds = [], {"w": 0, "t": 0, "l": 0}
        n_ds = 0
        for ds in datasets:
            x, y = _paired(data, ds, a, b)
            if len(x) < 5:
                continue
            n_ds += 1
            diffs.append(x - y)
            dm = x.mean() - y.mean()
            if dm > 1e-9:
                per_ds["w"] += 1
            elif dm < -1e-9:
                per_ds["l"] += 1
            else:
                per_ds["t"] += 1
        if not diffs:
            rows.append([b, a, cat, 0, "—", "—", "—", "—", "—", "【待补：数据不足】"])
            continue
        d = np.concatenate(diffs)
        dpp = 100 * float(d.mean())
        try:
            p = float(stats.wilcoxon(d)[1]) if np.any(d != 0) else float("nan")
        except Exception:
            p = float("nan")
        rows.append([b, a, cat, len(d), f"{dpp:+.3f}", f"±{100*ci95(d):.3f}",
                     (f"{p:.4g}" if p == p else "n/a"), f"{cliffs_delta(d, np.zeros_like(d)):+.3f}",
                     f"{per_ds['w']}/{per_ds['t']}/{per_ds['l']} (共{n_ds})",
                     _verdict(dpp, p, 0.0)])
    write_table(OUT_DIR + "/table9_cross_tests.csv", header, rows)
    return rows


def build_table4(data):
    datasets = detect_datasets(data)
    out, info = gather(data, datasets, ABLATION_METHODS)
    header = ["dataset", "method", "ACC±std", "F1±std", "相对CDPSO(完整)的Friedman排名", "结论", "data_source", "status"]
    rows = []
    for ds in datasets:
        present = out.get(ds, {})
        if "CDPSO(完整)" not in present or len(present) < 2:
            rows.append([ds, "—", "—", "—", "—", "—", "—",
                         "【待补：消融实验从未运行（IPSO / IPSO+OBL / 去重启 / 去扰动 / 去多样性）】"])
            continue
        f_p, holm = friedman_and_holm(present, "CDPSO(完整)")
        # 以均值排名
        rank = sorted(present.keys(), key=lambda m: present[m].mean(), reverse=True)
        rank_map = {m: i + 1 for i, m in enumerate(rank)}
        for m in ABLATION_METHODS:
            if m not in present:
                rows.append([ds, m, "—", "—", "—", "—", info.get((ds, m), "—"), "【待补：未运行】"])
                continue
            x = present[m]
            f1_raw = np.array([r[2] for r in data.get((ds, m), []) if r[2] == r[2]], float)
            f1_txt = (f"{f1_raw.mean():.4f}±{f1_raw.std(ddof=1):.4f}"
                      if len(f1_raw) >= 2 else "—")
            rows.append([ds, m, f"{x.mean():.4f}±{x.std(ddof=1):.4f}",
                         f1_txt,
                         rank_map.get(m, "—"),
                         ablation_verdict(data, ds, m),
                         info.get((ds, m), "—"), "OK" if m == "CDPSO(完整)" else "n.s."])
    write_table(OUT_DIR + "/table4_ablation.csv", header, rows)
    return rows


def build_table5(data):
    allm = MAIN_METHODS + BASELINE_METHODS
    datasets = detect_datasets(data)
    out, info = gather(data, datasets, allm)
    header = ["dataset", "method", "ACC±std(95%CI)", "F1±std", "Friedman排名", "Holm显著性(vs CDPSO-BP)", "data_source", "status"]
    rows = []
    for ds in datasets:
        present = out.get(ds, {})
        if CONTROL not in present:
            for m in allm:
                rows.append([ds, m, "—", "—", "—", "—", "—", "【待补：对照 CDPSO-BP 缺失】"])
            continue
        f_p, holm = friedman_and_holm(present, CONTROL)
        rank = sorted(present.keys(), key=lambda m: present[m].mean(), reverse=True)
        rank_map = {m: i + 1 for i, m in enumerate(rank)}
        hmap = {h["method"]: h for h in holm}
        for m in allm:
            if m not in present:
                rows.append([ds, m, "—", "—", "—", "—", "—", "【待补：未运行】"])
                continue
            x = present[m]
            h = hmap.get(m)
            hp = h["holm_p"] if h and h["holm_p"] == h["holm_p"] else "—"
            sig = "sig" if (h and h["sig"]) else "n.s."
            f1_raw = np.array([r[2] for r in data.get((ds, m), []) if r[2] == r[2]], float)
            f1_txt = (f"{f1_raw.mean():.4f}±{f1_raw.std(ddof=1):.4f}"
                      if len(f1_raw) >= 2 else "—")
            rows.append([ds, m, f"{x.mean():.4f}±{x.std(ddof=1):.4f}(±{ci95(x):.4f})",
                         f1_txt, rank_map.get(m, "—"),
                         hp, info.get((ds, m), "—"), sig])
    write_table(OUT_DIR + "/table5_baselines.csv", header, rows)
    return rows


def build_table6(data, sens_path):
    header = ["参数", "取值", "ACC±std", "排名", "备注(稳健区间)", "status"]
    rows = []
    if os.path.exists(sens_path):
        # 简单透传：参数敏感性 CSV 列为 param,value,dataset,accuracy,std,rank,note
        with open(sens_path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append([r.get("param", ""), r.get("value", ""), r.get("acc_std", ""),
                             r.get("rank", ""), r.get("note", ""), "OK"])
    if not rows:
        rows.append(["—", "—", "—", "—", "—",
                     "【待补：参数敏感性实验从未运行（pₘ/σ₀/停滞阈值/N/T 网格）】"])
    write_table(OUT_DIR + "/table6_sensitivity.csv", header, rows)
    return rows


def build_table_cross(data):
    """跨数据集汇总（主方法）：每数据集按 ACC 排名，取平均秩与夺冠次数，并做跨数据集 Friedman 检验。"""
    datasets = detect_datasets(data)
    methods = MAIN_METHODS
    per_ds_acc = {m: {} for m in methods}
    for ds in datasets:
        for m in methods:
            rows_m = data.get((ds, m), [])
            acc = np.array([r[1] for r in rows_m if r[1] == r[1]], float)
            if len(acc) >= 2:
                per_ds_acc[m][ds] = float(acc.mean())
    valid_ds = [ds for ds in datasets if all(ds in per_ds_acc[m] for m in methods)]
    if len(valid_ds) < 2:
        write_table(OUT_DIR + "/table7_cross_main.csv",
                    ["method", "avg_rank", "best_count", "mean_acc", "friedman_p", "status"],
                    [["—", "—", "—", "—", "—", "【待补：可用数据集不足】"]])
        return []
    avg_rank = {m: 0.0 for m in methods}
    best_count = {m: 0 for m in methods}
    for ds in valid_ds:
        order = sorted(methods, key=lambda m: per_ds_acc[m][ds], reverse=True)
        for i, m in enumerate(order):
            avg_rank[m] += (i + 1)
            if i == 0:
                best_count[m] += 1
    for m in methods:
        avg_rank[m] /= len(valid_ds)
    cols = [np.array([per_ds_acc[m][ds] for ds in valid_ds], float) for m in methods]
    try:
        _, f_p = stats.friedmanchisquare(*cols)
    except Exception:
        f_p = float("nan")
    header = ["method", "avg_rank", "best_count", "mean_acc", "friedman_p", "status"]
    rows = []
    for m in sorted(methods, key=lambda x: avg_rank[x]):
        rows.append([m, f"{avg_rank[m]:.2f}", best_count[m],
                     f"{np.mean([per_ds_acc[m][ds] for ds in valid_ds]):.4f}",
                     (f"{f_p:.4g}" if f_p == f_p else "n/a"), "OK"])
    write_table(OUT_DIR + "/table7_cross_main.csv", header, rows)
    return rows


def build_table_cross_all(data):
    """跨数据集汇总（全部方法：主+消融+基线），用于补充材料。"""
    datasets = detect_datasets(data)
    # 去重：CDPSO(完整)==CDPSO-BP、IPSO==IPSO-BP 为同一配置，跨方法总排名中仅保留主方法名
    DUP = ("CDPSO(完整)", "IPSO")
    allm = MAIN_METHODS + [m for m in ABLATION_METHODS if m not in DUP] + BASELINE_METHODS
    per_ds_acc = {m: {} for m in allm}
    for ds in datasets:
        for m in allm:
            rows_m = data.get((ds, m), [])
            acc = np.array([r[1] for r in rows_m if r[1] == r[1]], float)
            if len(acc) >= 2:
                per_ds_acc[m][ds] = float(acc.mean())
    valid_ds = [ds for ds in datasets if all(ds in per_ds_acc[m] for m in allm)]
    if len(valid_ds) < 2:
        write_table(OUT_DIR + "/table8_cross_all.csv",
                    ["method", "avg_rank", "mean_acc", "status"],
                    [["—", "—", "—", "【待补】"]])
        return []
    avg_rank = {m: 0.0 for m in allm}
    for ds in valid_ds:
        order = sorted(allm, key=lambda m: per_ds_acc[m][ds], reverse=True)
        for i, m in enumerate(order):
            avg_rank[m] += (i + 1)
    for m in allm:
        avg_rank[m] /= len(valid_ds)
    header = ["method", "avg_rank", "mean_acc", "status"]
    rows = []
    for m in sorted(allm, key=lambda x: avg_rank[x]):
        rows.append([m, f"{avg_rank[m]:.2f}",
                     f"{np.mean([per_ds_acc[m][ds] for ds in valid_ds]):.4f}", "OK"])
    write_table(OUT_DIR + "/table8_cross_all.csv", header, rows)
    return rows


# ----------------------------------------------------------------------------
# 真实数据模式
# ----------------------------------------------------------------------------
def run_real():
    print("="*70)
    print("真实数据模式：读取 results/raw_results.csv")
    print("WARNING: 若含 reconstructed_from_manuscript 行，它们是“从稿件均值±std重建”的")
    print("演示种子，并非原始实验导出；消融/基线/敏感性实验未运行 → 标【待补】。")
    print("="*70)
    data = load_real(OUT_DIR + "/raw_results.csv")
    if not data:
        print("[错误] 未找到 raw_results.csv 或为空。请先运行: python exp_harness.py --init-template")
        return
    t3 = build_table3(data)
    t4 = build_table4(data)
    t5 = build_table5(data)
    t6 = build_table6(data, OUT_DIR + "/sensitivity.csv")
    t7 = build_table_cross(data)
    t8 = build_table_cross_all(data)
    t9 = build_table_cross_tests(data)
    # summary
    summary = {"note": "real mode; 12 datasets; cross-dataset summary generated",
               "datasets": detect_datasets(data),
               "table3_rows": len(t3), "table4_rows": len(t4),
               "table5_rows": len(t5), "table6_rows": len(t6),
               "table7_rows": len(t7), "table8_rows": len(t8), "table9_rows": len(t9)}
    with open(OUT_DIR + "/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("已写出: table3_significance.csv / table4_ablation.csv / table5_baselines.csv / "
          "table6_sensitivity.csv / table7_cross_main.csv / table8_cross_all.csv / "
          "table9_cross_tests.csv / summary.json")


# ----------------------------------------------------------------------------
# 演示模式（合成数据，展示完整 Q1 表格格式）
# ----------------------------------------------------------------------------
def run_demo():
    global OUT_DIR
    demo_dir = OUT_DIR + "/demo"
    os.makedirs(demo_dir, exist_ok=True)
    saved = OUT_DIR
    OUT_DIR = demo_dir
    print("="*70)
    print("DEMO 模式：合成数据（SYNTHETIC），仅用于展示最终表格格式，非真实结果。")
    print("输出目录:", demo_dir)
    print("="*70)
    rng = np.random.default_rng(2026)
    datasets = ["LoL", "Heart", "Ionosphere", "WDBC", "Sonar", "GermanCredit",
                "Banknote", "Wine", "Diabetes", "Yacht", "Concrete"]
    all_methods = MAIN_METHODS + ABLATION_METHODS + BASELINE_METHODS
    deltas = {"CDPSO-BP": 0.0, "IPSO-BP": -0.004, "PSO-BP": -0.010, "BP": -0.006,
              "BP-30": -0.008, "IPSO": -0.002, "IPSO+OBL": -0.001, "CDPSO-无重启": -0.003,
              "CDPSO-无扰动": -0.0025, "CDPSO-无多样性": -0.006, "CDPSO(完整)": 0.0,
              "CLPSO": -0.003, "APSO": -0.004, "AMPSO": -0.005, "DMPSO": -0.004,
              "GLPSO": -0.0035, "SHADE": -0.002, "jSO": -0.0015, "CMA-ES": -0.001,
              "Xavier+Adam": -0.006, "Kaiming+Adam": -0.006}
    data = {}
    for ds in datasets:
        base = 0.85 if ds in ("Heart", "WDBC") else (0.88 if ds in ("Ionosphere",) else 0.80)
        for m in all_methods:
            vals = []
            for s in range(30):
                noise = rng.normal(0, 0.006)
                vals.append(base + deltas.get(m, -0.005) + noise)
            data[(ds, m)] = [(s, v, v - 0.002, float("nan"), "synthetic") for s, v in enumerate(vals)]
    t3 = build_table3(data)
    t4 = build_table4(data)
    t5 = build_table5(data)
    t6 = build_table6(data, OUT_DIR + "/sensitivity.csv")
    summary = {"mode": "demo_synthetic", "datasets": len(datasets),
               "methods": len(all_methods), "seeds": 30}
    with open(OUT_DIR + "/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    OUT_DIR = saved
    print("DEMO 表格已写出（标注 SYNTHETIC），位于 results/demo/。")


# ----------------------------------------------------------------------------
# 初始化模板（把稿件真实表2 以“重建种子”填入 + 预留空行）
# ----------------------------------------------------------------------------
def init_template():
    path = OUT_DIR + "/raw_results.csv"
    lines = []
    lines.append("# raw_results.csv —— exp_harness.py 输入（逐种子真实结果）")
    lines.append("# 列: dataset,method,seed,accuracy,f1,search_fitness,source")
    lines.append("# source 取值: real_experiment | reconstructed_from_manuscript | manuscript_mean")
    lines.append("# ！！reconstructed_from_manuscript 行是从稿件表2均值±std“重建”的演示种子，")
    lines.append("# ！！并非原始实验导出。请用你真实的逐种子导出替换，并补齐下方空行对应的")
    lines.append("# ！！消融/基线/敏感性实验（否则输出标【待补】）。")
    lines.append("dataset,method,seed,accuracy,f1,search_fitness,source")
    # 真实表2 重建种子（5 主方法 × 2 数据集 × 5 种子）
    for ds, mm in MANUSCRIPT_TABLE2.items():
        for m, (acc_mu, acc_sd, f1_mu, f1_sd) in mm.items():
            for i, e in enumerate(E_SHAPE):
                acc = acc_mu + acc_sd * e
                f1 = f1_mu + f1_sd * e
                lines.append(f"{ds},{m},{i},{acc:.4f},{f1:.4f},,reconstructed_from_manuscript")
    # 真实表1 搜索适应度（均值，无std）→ 作为 manuscript_mean 写入对应 PSO 方法
    for ds, mm in MANUSCRIPT_TABLE1.items():
        for m, sf in mm.items():
            lines.append(f"{ds},{m},0,,,{sf},manuscript_mean")
    # 预留空行：让作者知道要补哪些（消融 / 基线 / 敏感性）
    for ds in ["LoL", "Heart"]:
        for m in ["IPSO", "IPSO+OBL", "CDPSO-无重启", "CDPSO-无扰动", "CDPSO-无多样性", "CDPSO(完整)"]:
            lines.append(f"#{ds},{m},,,,,,TODO_消融_待运行")
        for m in BASELINE_METHODS:
            lines.append(f"#{ds},{m},,,,,,TODO_基线_待运行")
    # 扩展数据集占位（≥10 数据集）
    for ds in ["Ionosphere", "WDBC", "Sonar", "GermanCredit", "Banknote", "Wine",
               "Diabetes", "Yacht", "Concrete"]:
        for m in MAIN_METHODS:
            lines.append(f"#{ds},{m},,,,,,TODO_扩展数据集_待运行")
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")
    print("已生成模板:", path)
    print("下一步：用真实逐种子结果替换 reconstructed 行，并取消注释(#)补全消融/基线/扩展数据集行，")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="合成数据演示完整表格")
    ap.add_argument("--init-template", action="store_true", help="生成 raw_results.csv 模板")
    args = ap.parse_args()
    if args.init_template:
        init_template()
    elif args.demo:
        run_demo()
    else:
        run_real()


if __name__ == "__main__":
    main()
