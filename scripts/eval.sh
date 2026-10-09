#!/usr/bin/env bash
# Evaluate trained TCGA models on TCGA (held-out test folds) and the external cohorts.
#
# Usage: scripts/eval.sh <MODEL> <BACKBONE> <MAG> [COHORTS]
#   COHORTS  space-separated subset of "tcga ebrains ipd" (default: all three)
#
# Environment (all optional): FEAT_ROOT (data/features), RESULTS_DIR (results),
#   EVAL_DIR (eval_results), K (10), CSV_PATH / SPLIT_DIR (override the tcga labels / splits), PYTHON.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

[ "$#" -ge 3 ] || { sed -n '2,10p' "$0"; exit 1; }
MODEL=$1; BACKBONE=$2; MAG=$3
COHORTS=${4:-"tcga ebrains ipd"}

TASK=tcga_3_class
LABEL=who2021
RESULTS_DIR=${RESULTS_DIR:-results}
EVAL_DIR=${EVAL_DIR:-eval_results}
IN_DIM=$(backbone_dim "$BACKBONE")
MODELS_EXP="tcga_${LABEL}/${BACKBONE}/${MODEL}/${MAG}_s1"

for COHORT in $COHORTS; do
    case "$COHORT" in
        tcga)    CSV=${CSV_PATH:-dataset_csv/tcga_2021_who_labels.csv}; SPLITS=${SPLIT_DIR:-splits/tcga_who_2021_100} ;;
        ebrains) CSV=dataset_csv/ebrains_2021_who_labels.csv;         SPLITS=splits/ebrain_who_2021_100 ;;
        ipd)     CSV=dataset_csv/ipd_2021_who_labels_slidewise.csv;   SPLITS=splits/ipd_who_2021_100 ;;
        *) echo "Error: unknown cohort '$COHORT'" >&2; exit 1 ;;
    esac
    echo "Evaluating ${COHORT}: ${BACKBONE} | ${MODEL} | ${MAG}"
    "$PY" eval.py \
        --k "${K:-10}" \
        --models_exp_code "$MODELS_EXP" \
        --save_exp_code "${COHORT}_${LABEL}/${BACKBONE}/${MODEL}/${MAG}" \
        --task "$TASK" --model_type "$MODEL" \
        --results_dir "$RESULTS_DIR" --eval_dir "$EVAL_DIR" --split test \
        --features_dir "${FEAT_ROOT:-data/features}/${BACKBONE}/${COHORT}/${MAG}" \
        --csv_path "$CSV" --splits_dir "$SPLITS" --embed_dim "$IN_DIM"
done
