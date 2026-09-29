# -*- coding: utf-8 -*-
"""下载 10 个公开二分类数据集到 data/ 目录，清洗为标准 CSV（数值特征 + 二值标签）。"""
import os, sys, io, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_openml

OUT = r"D:/02XIHO Files/Daily Work/03Scientific Research/00Article_File/07改进粒子群算法/data"
os.makedirs(OUT, exist_ok=True)

# (内部键, OpenML 数据集名, 备注)
SPEC = [
    ("Pima",       "diabetes",                       "Pima Indians Diabetes"),
    ("WDBC",       "Breast Cancer Wisconsin (Diagnostic)", "WDBC"),
    ("Ionosphere", "ionosphere",                     "Ionosphere"),
    ("Sonar",      "sonar",                           "Sonar Mines vs Rocks"),
    ("Banknote",   "banknote-authentication",         "Banknote Auth"),
    ("Spambase",   "spambase",                        "Spambase"),
    ("Haberman",   "haberman",                        "Haberman Survival"),
    ("Parkinson",  "parkinsons",                      "Parkinsons"),
    ("Blood",      "Blood Transfusion Service Center","Blood Transfusion"),
    ("Phoneme",    "phoneme",                         "Phoneme"),
]

ok = []
for key, name, note in SPEC:
    try:
        print(f"[fetch] {key} <- {name} ...", flush=True)
        ds = fetch_openml(name=name, as_frame=True, parser="auto")
        X = ds.data.copy()
        y = ds.target.copy()
        # 仅保留数值特征，去掉非数值/ID 列
        num_cols = []
        for c in X.columns:
            col = pd.to_numeric(X[c], errors="coerce")
            if col.notna().mean() > 0.99 and X[c].nunique() < len(X):  # 非 ID
                X[c] = col
                num_cols.append(c)
        X = X[num_cols].astype(float)
        # 标签二值化
        if y.dtype == object or str(y.dtype).startswith("category"):
            classes = sorted(y.unique().tolist())
            y = y.map({v: i for i, v in enumerate(classes)})
        else:
            y = pd.to_numeric(y, errors="coerce")
            uniq = sorted(pd.unique(y.dropna()))
            if set(uniq) == {1, 2}:
                y = (y - 1).astype(float)        # {1,2} -> {0,1}
            elif set(uniq) <= {0, 1}:
                y = y.astype(float)
            else:
                y = y.rank(method="dense").astype(int) - 1
        y = y.astype(float).values
        df = X.copy()
        df["label"] = y
        pos = int((y == 1).sum()); neg = int((y == 0).sum())
        path = os.path.join(OUT, f"{key}.csv")
        df.to_csv(path, index=False)
        ok.append((key, X.shape[1], len(df), pos, neg))
        print(f"  -> OK  {key}: 样本={len(df)} 特征={X.shape[1]} 正={pos} 负={neg}  -> {path}", flush=True)
    except Exception as e:
        print(f"  -> FAIL {key}: {type(e).__name__}: {e}", flush=True)

print("\n=== 汇总 ===")
for k, nfeat, n, p, ng in ok:
    print(f"{k:12s} feat={nfeat:3d} n={n:6d} pos={p:5d} neg={ng:5d}")
print(f"成功 {len(ok)}/10")
