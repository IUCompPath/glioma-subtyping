#!/usr/bin/env python3
"""
ensemble_script.py

This script looks for evaluation outputs created by your eval.sh:
 - tcga_<LABEL>/<BACKBONE>/<MODEL>/<MAG>
 - ebrains_<LABEL>/<BACKBONE>/<MODEL>/<MAG>
 - ipd_<LABEL>/<BACKBONE>/<MODEL>/<MAG>

It builds ensembles (averages probs across magnifications) and summary CSVs for each SOURCE that exists.

Usage:
    python ensemble_script.py <label> <backbone> <model> [--out_root OUT_ROOT] [--sources tcga,ebrains,ipd]

Examples:
    python ensemble_script.py who2021 uni clam_sb
    python ensemble_script.py who2021 uni clam_sb --sources tcga
    python ensemble_script.py who2021 uni clam_sb --out_root /path/to/experiments --sources tcga,ipd
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# -----------------------
# Configuration / Defaults
# -----------------------
DEFAULT_TASK = "tcga_3_class"
COHORT = ["test"]
MAG_LIST_ALL = ["2.5x", "5x", "10x", "20x"]

ENSEMBLE_GROUPS_CREATE = [
    "5x_10x_20x",
    "10x_20x",
    "5x_10x",
    "5x_20x",
    "2.5x_20x",
    "2.5x_5x_10x",
    "2.5x_5x",
    "2.5x_10x",
    "2.5x_5x_20x",
    "2.5x_10x_20x",
    "2.5x_5x_10x_20x",
]

ENSEMBLE_GROUPS_SUMMARY = [
    "2.5x",
    "5x",
    "10x",
    "20x",
    "5x_10x_20x",
    "10x_20x",
    "5x_10x",
    "5x_20x",
    "2.5x_20x",
    "2.5x_5x_10x",
    "2.5x_5x",
    "2.5x_10x",
    "2.5x_5x_20x",
    "2.5x_10x_20x",
    "2.5x_5x_10x_20x",
]


# -----------------------
# Helpers
# -----------------------
def compute_prediction_dataframe_by_averaging(dfs):
    result_df = pd.concat(dfs, ignore_index=True)
    grouped = result_df.groupby("slide_id", as_index=False).mean()
    prob_cols = [c for c in grouped.columns if c.startswith("p_")]
    if len(prob_cols) == 0:
        raise ValueError("No probability columns (p_*) found in dataframes.")
    grouped["Y_hat"] = grouped[prob_cols].values.argmax(axis=1).astype(int)
    return grouped


def compute_auc_for_df(df, n_classes=3):
    labels = df["Y"].to_list()
    binary = label_binarize(labels, classes=list(range(n_classes)))
    aucs = []
    for class_idx in range(n_classes):
        if class_idx in labels:
            fpr, tpr, _ = roc_curve(binary[:, class_idx], df[f"p_{class_idx}"].to_list())
            aucs.append(auc(fpr, tpr))
        else:
            aucs.append(np.nan)
    return float(np.nanmean(np.array(aucs)))


def compute_sens_spec_from_cm(cm, n_classes=3):
    sens = []
    spec = []
    for class_idx in range(n_classes):
        true_positive = cm[class_idx, class_idx]
        false_negative = cm[class_idx, :].sum() - true_positive
        false_positive = cm[:, class_idx].sum() - true_positive
        true_negative = cm.sum() - (true_positive + false_negative + false_positive)
        sensitivity = (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0
        specificity = (true_negative / (true_negative + false_positive)) if (true_negative + false_positive) > 0 else 0.0
        sens.append(sensitivity)
        spec.append(specificity)
    return sens, spec


# -----------------------
# Ensemble creation & summary (per dir_path)
# -----------------------
def create_ensemble_csvs(dir_path, task=DEFAULT_TASK, cohort=COHORT):
    for q in tqdm(ENSEMBLE_GROUPS_CREATE, desc=f"Creating ensemble CSVs for {dir_path}"):
        mags_to_consider = q.split("_")
        results_dir = os.path.join(dir_path, q)
        os.makedirs(results_dir, exist_ok=True)
        for m in cohort:
            out_eval_dir = os.path.join(results_dir, f"EVAL_tcga_2021_{task}_{q}_{m}")
            os.makedirs(out_eval_dir, exist_ok=True)
            for l in range(10):
                dfs = []
                missing = False
                for mag in mags_to_consider:
                    file_path = os.path.join(dir_path, mag, f"EVAL_tcga_2021_{task}_{mag}_{m}", f"fold_{l}.csv")
                    if not os.path.exists(file_path):
                        # missing -> skip this fold for this ensemble group
                        print(f"[WARN] missing {file_path} -> skipping fold {l} for ensemble {q} at {dir_path}")
                        missing = True
                        break
                    dfs.append(pd.read_csv(file_path))
                if missing or len(dfs) == 0:
                    continue
                df_avg = compute_prediction_dataframe_by_averaging(dfs)
                out_file = os.path.join(out_eval_dir, f"fold_{l}.csv")
                df_avg.to_csv(out_file, index=False)


def create_summary_csvs(dir_path, task=DEFAULT_TASK):
    for q in tqdm(ENSEMBLE_GROUPS_SUMMARY, desc=f"Creating summary CSVs for {dir_path}"):
        results_dir = os.path.join(dir_path, q)
        df_summary = pd.DataFrame(columns=[
            "model", "test_auc", "test_acc", "balanced_test_accuracy", "test_sen", "test_spec"
        ])
        for l in range(10):
            test_path = os.path.join(results_dir, f"EVAL_tcga_2021_{task}_{q}_test", f"fold_{l}.csv")
            if not os.path.exists(test_path):
                print(f"[WARN] missing test file {test_path} -> skipping fold {l} for summary {q} at {dir_path}")
                continue
            df_test = pd.read_csv(test_path)
            test_acc = accuracy_score(df_test["Y"].to_list(), df_test["Y_hat"].to_list())
            test_acc_b = balanced_accuracy_score(df_test["Y"].to_list(), df_test["Y_hat"].to_list())
            test_auc_score = compute_auc_for_df(df_test, n_classes=3)
            cm_test = confusion_matrix(df_test["Y"].to_list(), df_test["Y_hat"].to_list())
            test_sen, test_spec = compute_sens_spec_from_cm(cm_test, n_classes=3)
            df_summary = df_summary.append({
                "model": l,
                "test_auc": test_auc_score,
                "test_acc": test_acc,
                "balanced_test_accuracy": test_acc_b,
                "test_sen": test_sen,
                "test_spec": test_spec,
            }, ignore_index=True)
        out_summary = os.path.join(results_dir, f"EVAL_tcga_idh_{q}_eval_results_detailed.csv")
        if not df_summary.empty:
            os.makedirs(results_dir, exist_ok=True)
            df_summary.to_csv(out_summary, index=False)
        else:
            print(f"[INFO] No summary rows for {results_dir}; no CSV written.")


# -----------------------
# CLI + driver
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create ensemble CSVs and summaries by averaging magnification predictions.")
    p.add_argument("label", type=str, help="Label part used in directories (e.g. tcga)")
    p.add_argument("backbone", type=str, help="Backbone name (e.g. uni)")
    p.add_argument("model", type=str, help="Model name (e.g. mamba_mil)")
    p.add_argument("--out_root", type=str, default=".", help="Root folder where source_* directories live (default=.)")
    p.add_argument("--sources", type=str, default="tcga,ebrains,ipd", help="Comma-separated sources to check (default=tcga,ebrains,ipd)")
    return p.parse_args()


def main():
    args = parse_args()
    label = args.label
    backbone = args.backbone
    model = args.model
    out_root = args.out_root
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    # Build candidate base directories for each source
    found = []
    for src in sources:
        candidate = os.path.join(out_root, f"{src}_{label}", backbone, model)
        if os.path.isdir(candidate):
            found.append((src, candidate))
        else:
            print(f"[INFO] Not found: {candidate}")

    if not found:
        print("[ERROR] No matching source directories found. Checked:")
        for src in sources:
            print(f"  - {os.path.join(out_root, f'{src}_{label}', backbone, model)}")
        sys.exit(2)

    # For each found source dir, verify magnification subfolders and run ensembles
    for src, dir_path in found:
        print(f"\n=== Processing source: {src}  dir: {dir_path} ===")
        missing_mags = [mag for mag in MAG_LIST_ALL if not os.path.isdir(os.path.join(dir_path, mag))]
        if missing_mags:
            print(f"[WARN] Missing magnification folders under {dir_path}: {missing_mags}")
            # still attempt to run; some ensembles may still be possible if required mags exist
        # create ensemble CSVs and summaries
        create_ensemble_csvs(dir_path, task=DEFAULT_TASK, cohort=COHORT)
        create_summary_csvs(dir_path, task=DEFAULT_TASK)
        print(f"[DONE] Completed for {src} ({dir_path})")

    print("\nAll processing finished.")


if __name__ == "__main__":
    main()
