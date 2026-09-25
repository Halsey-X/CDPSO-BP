# CDPSO-BP: Chaos-Driven Particle Swarm Optimization with Diversity Feedback for BP Neural Network Training

Official reproduction code and results for the manuscript *"Improved Particle Swarm Optimization with Chaos-based Oppositional Learning and Diversity Feedback for BP Neural Network Training"* (under review).

## Overview

CDPSO is a low-budget PSO variant for training three-layer BP neural networks. It combines:

1. **Tent chaotic map + Opposition-Based Learning (OBL) initialization** — diversified initial swarm;
2. **Diversity-feedback inertial weight** — the mean-square displacement of the swarm adaptively adjusts the exploration/exploitation balance;
3. **Gaussian perturbation + stagnation restart** — escapes local optima under a strictly limited search budget (N=20 particles, T=40 iterations).

All experiments fix a *small, identical* training budget across methods, so differences reflect **initialization/optimization quality rather than extra compute**.

## Repository layout

```
CDPSO-BP/
├── code/
│   ├── run_experiment.py   # Main pipeline: BP, PSO-BP, IPSO-BP, CDPSO-BP + ablation variants (30 seeds)
│   ├── baselines.py        # 10 extra baselines: CLPSO, APSO, AMPSO, DMPSO, GLPSO, SHADE, jSO, CMA-ES, Xavier+Adam, Kaiming+Adam
│   ├── sensitivity.py      # Parameter sensitivity study (Table 6)
│   └── exp_harness.py      # Q1-grade statistics: Friedman + Holm correction, Wilcoxon, Cliff's delta, 95% CI
├── data/
│   ├── heart.csv           # Heart disease dataset (918 samples, 19 features) — see Data sources
│   └── lol.csv             # League of Legends ranked 10-min dataset (9,879 samples, 38 features) — see Data sources
└── results/                # Final tables produced by the real 30-seed runs
    ├── table3_significance.csv
    ├── table4_ablation.csv
    ├── table5_baselines.csv
    ├── table6_sensitivity.csv
    └── summary.json
```

## Requirements

- Python >= 3.12
- numpy, scipy, scikit-learn, cma

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

```bash
# 1) Main methods + ablation (30 independent seeds, both datasets)
python code/run_experiment.py --methods main,ablation --seeds 30 --datasets LoL,Heart

# 2) 10 supplementary baselines (30 seeds, both datasets)
python code/baselines.py --seeds 30 --datasets LoL,Heart

# 3) Parameter sensitivity (10 seeds per configuration)
python code/sensitivity.py --seeds 10

# 4) Generate Q1-grade statistical tables (Friedman + Holm, Cliff's delta, 95% CI)
python code/exp_harness.py
```

Each seed is deterministic (`seed = 3000 + k`), so the tables in `results/` are exactly reproducible.

## Data sources

| File | Origin | License note |
|------|--------|--------------|
| `heart.csv` | Heart Failure Prediction dataset (Kaggle), 918 samples, 55.3% positive | Used here for research reproduction only; original author rights apply |
| `lol.csv` | League of Legends Diamond ranked games, first 10 min (Kaggle), 9,879 samples | Used here for research reproduction only; original author rights apply |

Both datasets were obtained from public mirrors of the original Kaggle uploads. If you use them, please also cite the original dataset sources.

## Citing

If this code or the datasets' preprocessing helps your research, please cite the manuscript (citation to be added upon acceptance).

## License

MIT License — see [LICENSE](LICENSE). The datasets in `data/` are redistributed for reproducibility and remain under their original licenses.
