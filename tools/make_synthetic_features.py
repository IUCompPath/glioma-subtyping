#!/usr/bin/env python3
"""Create class-dependent synthetic patch features for a backbone we have no real features for.

Mirrors the layout of smoke_test/data/features/<backbone>/tcga/<mag>/pt_files, using the
slide list and labels of smoke_test/data/labels.csv. Each class gets its own random mean
vector (shared across magnifications), so models can learn something; purely a plumbing test.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

MAGS = ["2.5x", "5x", "10x", "20x"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="smoke_test/data")
    p.add_argument("--backbone", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--n_patches", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    labels = pd.read_csv(os.path.join(a.data, "labels.csv"))
    rng = np.random.default_rng(a.seed)
    class_means = rng.normal(0, 1, (labels.label.nunique(), a.dim))
    for m in MAGS:
        dst = os.path.join(a.data, "features", a.backbone, "tcga", m, "pt_files")
        os.makedirs(dst, exist_ok=True)
        for sid, y in zip(labels.slide_id, labels.label):
            x = class_means[y] * 0.5 + rng.normal(0, 1, (a.n_patches, a.dim))
            torch.save(torch.from_numpy(x.astype(np.float32)), os.path.join(dst, sid + ".pt"))
    print(f"synthetic {a.backbone} features (dim {a.dim}) for {len(labels)} slides x {len(MAGS)} mags")


if __name__ == "__main__":
    main()
