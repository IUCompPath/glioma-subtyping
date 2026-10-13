#!/usr/bin/env python3
"""Build the tiny, committed feature set used by smoke_test/ from real TCGA features.

Picks ``--per_class`` slides per WHO-2021 class that exist at every magnification,
keeps a fixed random subset of ``--n_patches`` patch embeddings per slide, and writes:

    <out>/features/<backbone>/tcga/<mag>/pt_files/<slide>.pt
    <out>/labels.csv                       slide_id,case_id,label
    <out>/splits/splits_{0,1}.csv          2 folds, train/val/test per fold

Slides are taken in order of smallest source file (cheap to read, deterministic).
The subsampled features are for pipeline testing only, not for any scientific result.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

MAGS = ["2.5x", "5x", "10x", "20x"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="dir containing <mag>/pt_files/*.pt (full-size features)")
    p.add_argument("--labels", default="dataset_csv/tcga_2021_who_labels.csv")
    p.add_argument("--out", default="smoke_test/data")
    p.add_argument("--backbone", default="gigapath")
    p.add_argument("--per_class", type=int, default=4)
    p.add_argument("--n_patches", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    df = pd.read_csv(a.labels)[["slide_id", "case_id", "label"]].drop_duplicates("slide_id")
    have = lambda m: {f[:-3] for f in os.listdir(os.path.join(a.src, m, "pt_files"))}
    common = set.intersection(*(have(m) for m in MAGS))
    df = df[df.slide_id.isin(common)].copy()
    df["size"] = [os.path.getsize(os.path.join(a.src, "20x", "pt_files", s + ".pt")) for s in df.slide_id]
    # one slide per patient, smallest first
    df = df.sort_values("size").drop_duplicates("case_id")
    chosen = pd.concat([df[df.label == c].head(a.per_class) for c in sorted(df.label.unique())])
    assert len(chosen) == a.per_class * 3, "not enough slides"

    rng = np.random.default_rng(a.seed)
    for m in MAGS:
        dst = os.path.join(a.out, "features", a.backbone, "tcga", m, "pt_files")
        os.makedirs(dst, exist_ok=True)
        for s in chosen.slide_id:
            x = torch.load(os.path.join(a.src, m, "pt_files", s + ".pt"), map_location="cpu").float()
            idx = np.sort(rng.choice(len(x), min(a.n_patches, len(x)), replace=False))
            torch.save(x[idx].clone(), os.path.join(dst, s + ".pt"))
    print("feature dim:", x.shape[1])

    chosen[["slide_id", "case_id", "label"]].to_csv(os.path.join(a.out, "labels.csv"), index=False)
    # 2 folds: within each class, rotate which slide is val / test; the rest train
    os.makedirs(os.path.join(a.out, "splits"), exist_ok=True)
    for fold in range(2):
        parts = {"train": [], "val": [], "test": []}
        for c in sorted(chosen.label.unique()):
            ids = list(chosen[chosen.label == c].slide_id)
            ids = ids[fold:] + ids[:fold]
            parts["test"].append(ids[0]); parts["val"].append(ids[1]); parts["train"] += ids[2:]
        pd.DataFrame({k: pd.Series(v) for k, v in parts.items()}).to_csv(
            os.path.join(a.out, "splits", f"splits_{fold}.csv"), index=False)
    print(chosen.groupby("label").size().to_dict(), "slides;", a.n_patches, "patches each")


if __name__ == "__main__":
    main()
