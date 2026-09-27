"""诊断：停滞重启机制在 CDPSO 完整版中是否真的被触发。

用法: python diag_restart.py
"""
import os
import numpy as np
import run_experiment as R


_CACHE = {}


def build(ds_name, seed=0):
    if ds_name not in _CACHE:
        heart_cands, lol_cands = R.get_datasets()
        hp = next(p for p in heart_cands if os.path.exists(p))
        lp = next(p for p in lol_cands if os.path.exists(p))
        _CACHE["Heart"] = R.load_heart(hp)
        _CACHE["LoL"] = R.load_lol(lp)
    X, y = _CACHE[ds_name]
    return R.split_standardize(X, y, seed)


def count(ds_name, mode, seeds=5):
    Xtr, ytr, Xva, yva, Xte, yte = build(ds_name)
    d_in = Xtr.shape[1]
    R.RESTART_EVENTS = 0
    R.MAX_STAG = 0
    for k in range(seeds):
        rng = np.random.default_rng(3000 + k)
        R.pso_search(Xtr, ytr, d_in, 3000 + k, mode, rng=rng)
    return R.RESTART_EVENTS, R.MAX_STAG


if __name__ == "__main__":
    for ds in ("Heart", "LoL"):
        for mode in ("cdpso_full", "cdpso_no_perturb"):
            ev, ms = count(ds, mode, seeds=5)
            print(f"{ds:6s} {mode:18s} 5 seeds -> 重启触发 {ev} 次 | 最大连续停滞 {ms} 代")
