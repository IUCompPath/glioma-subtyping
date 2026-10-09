#!/usr/bin/env python3
"""Assert that every pipeline stage of scripts/smoke_test.sh produced sane output."""
import argparse
import glob
import itertools
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch


def check(cond, msg):
    if not cond:
        sys.exit(f"SMOKE CHECK FAILED: {msg}")
    print(f"ok  {msg}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--mags", nargs="+", required=True)
    p.add_argument("--backbone", default="imagenet")
    p.add_argument("--n_classes", type=int, default=3)
    a = p.parse_args(argv)

    labels = pd.read_csv(os.path.join(a.out, "labels.csv"))
    slide_ids = set(labels.slide_id)

    for mag in a.mags:
        patch_h5 = glob.glob(os.path.join(a.out, "patches", "smoke", mag, "patches", "*.h5"))
        check({os.path.basename(f)[:-3] for f in patch_h5} == slide_ids, f"{mag}: patch coordinates for every slide")
        feat_dir = os.path.join(a.out, "features", a.backbone, "smoke", mag)
        for sid in slide_ids:
            with h5py.File(os.path.join(a.out, "patches", "smoke", mag, "patches", sid + ".h5")) as f:
                n_patches = len(f["coords"])
            feats = torch.load(os.path.join(feat_dir, "pt_files", sid + ".pt"))
            check(feats.shape == (n_patches, 1024) and bool(torch.isfinite(feats).all()),
                  f"{mag}/{sid[:24]}: features {tuple(feats.shape)} finite, one per patch")

    for model in a.models:
        for mag in a.mags:
            exp = os.path.join(a.out, "results", "tcga_who2021", a.backbone, model, f"{mag}_s1")
            check(os.path.isfile(os.path.join(exp, "s_0_checkpoint.pt")), f"{model}@{mag}: checkpoint saved")
            fold = os.path.join(a.out, "eval_results", "tcga_who2021", a.backbone, model, mag, "fold_0.csv")
            df = pd.read_csv(fold)
            probs = df[[f"p_{c}" for c in range(a.n_classes)]].to_numpy()
            check(set(df.slide_id) == slide_ids and np.allclose(probs.sum(1), 1, atol=1e-4),
                  f"{model}@{mag}: predictions for every slide, probabilities sum to 1")
        if len(a.mags) > 1:
            group = "_".join(sorted(a.mags, key=lambda m: float(m[:-1])))
            base = os.path.join(a.out, "eval_results", "tcga_who2021", a.backbone, model, group)
            check(os.path.isfile(os.path.join(base, "fold_0.csv")) and os.path.isfile(os.path.join(base, "summary.csv")),
                  f"{model}: late-fusion {group} written")


if __name__ == "__main__":
    main()
