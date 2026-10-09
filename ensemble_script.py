#!/usr/bin/env python3
"""Late fusion across magnifications.

Reads the per-fold slide-level predictions written by ``eval.py``::

    <eval_dir>/<source>_<label>/<backbone>/<model>/<mag>/fold_<k>.csv

and, for every combination of >= 2 available magnifications (e.g. ``5x_10x_20x``),
averages the class probabilities per slide, takes the argmax, and writes::

    <eval_dir>/<source>_<label>/<backbone>/<model>/<mag1>_<mag2>_.../fold_<k>.csv
    <eval_dir>/<source>_<label>/<backbone>/<model>/<group>/summary.csv   (per-fold metrics)

``summary.csv`` is also written for each single magnification, so single-scale and fused
results can be compared directly.

Usage:
    python ensemble_script.py <label> <backbone> <model> [--eval_dir DIR] [--sources tcga,ebrains,ipd]
    python ensemble_script.py who2021 uni mamba_mil --sources tcga
"""
import argparse
import glob
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

MAG_ORDER = ["2.5x", "5x", "10x", "20x", "40x"]
N_CLASSES = 3


def average_predictions(dfs):
    """Average the ``p_*`` columns of several per-fold prediction frames by slide_id."""
    df = pd.concat(dfs, ignore_index=True)
    prob_cols = sorted(c for c in df.columns if c.startswith("p_"))
    if not prob_cols:
        raise ValueError("No probability columns (p_*) found.")
    out = df.groupby("slide_id", as_index=False).agg({"Y": "first", **{c: "mean" for c in prob_cols}})
    out["Y_hat"] = out[prob_cols].values.argmax(axis=1).astype(int)
    return out[["slide_id", "Y", "Y_hat"] + prob_cols]


def class_auc(df, n_classes=N_CLASSES):
    """One-vs-rest AUC averaged over the classes present in ``df``."""
    y = df["Y"].astype(int).to_numpy()
    aucs = []
    for c in range(n_classes):
        pos = (y == c).astype(int)
        if 0 < pos.sum() < len(pos):
            aucs.append(roc_auc_score(pos, df[f"p_{c}"].to_numpy()))
    return float(np.mean(aucs)) if aucs else float("nan")


def sens_spec(cm):
    sens, spec = [], []
    total = cm.sum()
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens.append(tp / (tp + fn) if tp + fn else 0.0)
        spec.append(tn / (tn + fp) if tn + fp else 0.0)
    return sens, spec


def summarize_fold(df, fold, n_classes=N_CLASSES):
    y, y_hat = df["Y"].astype(int), df["Y_hat"].astype(int)
    cm = confusion_matrix(y, y_hat, labels=list(range(n_classes)))
    sens, spec = sens_spec(cm)
    return {
        "fold": fold,
        "auc": class_auc(df, n_classes),
        "acc": accuracy_score(y, y_hat),
        "balanced_acc": balanced_accuracy_score(y, y_hat),
        "sensitivity": sens,
        "specificity": spec,
    }


def fold_files(mag_dir):
    """{fold_index: path} for every ``fold_<k>.csv`` in ``mag_dir``."""
    files = {}
    for path in glob.glob(os.path.join(mag_dir, "fold_*.csv")):
        m = re.fullmatch(r"fold_(\d+)\.csv", os.path.basename(path))
        if m:
            files[int(m.group(1))] = path
    return files


def available_mags(model_dir):
    return [m for m in MAG_ORDER if fold_files(os.path.join(model_dir, m))]


def write_summary(group_dir, folds_dfs):
    rows = [summarize_fold(df, k) for k, df in sorted(folds_dfs.items())]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(group_dir, "summary.csv"), index=False)


def run_model_dir(model_dir):
    mags = available_mags(model_dir)
    if not mags:
        print(f"[WARN] no per-magnification fold CSVs under {model_dir}")
        return []
    per_mag = {m: {k: pd.read_csv(p) for k, p in fold_files(os.path.join(model_dir, m)).items()} for m in mags}
    for m in mags:
        write_summary(os.path.join(model_dir, m), per_mag[m])

    written = []
    for r in range(2, len(mags) + 1):
        for combo in itertools.combinations(mags, r):
            group = "_".join(combo)
            group_dir = os.path.join(model_dir, group)
            os.makedirs(group_dir, exist_ok=True)
            common_folds = sorted(set.intersection(*(set(per_mag[m]) for m in combo)))
            fused = {}
            for k in common_folds:
                fused[k] = average_predictions([per_mag[m][k] for m in combo])
                fused[k].to_csv(os.path.join(group_dir, f"fold_{k}.csv"), index=False)
            write_summary(group_dir, fused)
            written.append(group)
            print(f"[OK] {group}: {len(fused)} fold(s)")
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label", help="label tag used in the experiment name, e.g. who2021")
    p.add_argument("backbone", help="feature extractor, e.g. uni")
    p.add_argument("model", help="MIL model, e.g. mamba_mil")
    p.add_argument("--eval_dir", default="./eval_results", help="root written by eval.py (default: ./eval_results)")
    p.add_argument("--sources", default="tcga,ebrains,ipd", help="comma-separated cohorts (default: tcga,ebrains,ipd)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    found = [(s, os.path.join(args.eval_dir, f"{s}_{args.label}", args.backbone, args.model)) for s in sources]
    found = [(s, d) for s, d in found if os.path.isdir(d)]
    if not found:
        print("[ERROR] no matching directories under", args.eval_dir, file=sys.stderr)
        return 2
    for src, model_dir in found:
        print(f"=== {src}: {model_dir}")
        run_model_dir(model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
