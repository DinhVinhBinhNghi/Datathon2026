# -*- coding: utf-8 -*-
"""Regenerate the precomputed feature-group importance figure from CSV values.

This script is intentionally lightweight. It does not retrain the forecasting model;
it only recreates the report-ready explainability plot from the saved importance table.
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/modeling/feature_group_importance_comparison.csv")
    parser.add_argument("--out", default="outputs/modeling/feature_group_importance_comparison.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    labels = df["feature_group"].tolist()
    a = df["model_a_2013_2018_pct"].to_numpy(dtype=float)
    b = df["model_b_2019_2022_pct"].to_numpy(dtype=float)

    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.barh(y + h/2, a, height=h, label="Model A (2013-2018)")
    ax.barh(y - h/2, b, height=h, label="Model B (2019-2022)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("% Total feature importance (LightGBM gain)", fontsize=11)
    ax.set_title("Feature Group Importance: A+B Regime Comparison (33 features, compliant)", fontsize=13, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    maxv = max(float(np.max(a)), float(np.max(b)))
    ax.set_xlim(0, maxv * 1.18)
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(va + maxv * 0.01, i + h/2, f"{va:.0f}%", va="center", fontsize=9)
        ax.text(vb + maxv * 0.01, i - h/2, f"{vb:.0f}%", va="center", fontsize=9)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved {args.out}")


if __name__ == "__main__":
    main()
