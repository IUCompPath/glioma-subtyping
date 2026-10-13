#!/usr/bin/env python3
"""Verify the artifacts of smoke_test/run.sh (training -> evaluation -> multi-scale fusion)."""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd


def check(cond, msg):
    if not cond:
        sys.exit(f"SMOKE CHECK FAILED: {msg}")
    print(f"ok  {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="smoke_test/data")
    p.add_argument("--backbones", nargs="+", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--mags", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=2)
    a = p.parse_args()

    for bb, model in itertools.product(a.backbones, a.models):
        base = os.path.join(a.out, "eval_results", "tcga_who2021", bb, model)
        for mag in a.mags:
            for k in range(a.folds):
                ckpt = os.path.join(a.out, "results", "tcga_who2021", bb, model, f"{mag}_s1", f"s_{k}_checkpoint.pt")
                check(os.path.isfile(ckpt), f"{bb}/{model}@{mag} fold {k}: checkpoint")
                split = pd.read_csv(os.path.join(a.data, "splits", f"splits_{k}.csv"))
                test_ids = set(split.test.dropna())
                df = pd.read_csv(os.path.join(base, mag, f"fold_{k}.csv"))
                probs = df[[f"p_{c}" for c in range(3)]].to_numpy()
                check(set(df.slide_id) == test_ids and np.allclose(probs.sum(1), 1, atol=1e-4),
                      f"{bb}/{model}@{mag} fold {k}: predictions for the {len(test_ids)} test slides, probs sum to 1")
        for r in range(2, len(a.mags) + 1):
            for combo in itertools.combinations(sorted(a.mags, key=lambda m: float(m[:-1])), r):
                g = "_".join(combo)
                for k in range(a.folds):
                    check(os.path.isfile(os.path.join(base, g, f"fold_{k}.csv")), f"{bb}/{model} fusion {g} fold {k}")
                check(os.path.isfile(os.path.join(base, g, "summary.csv")), f"{bb}/{model} fusion {g} summary")


if __name__ == "__main__":
    main()
