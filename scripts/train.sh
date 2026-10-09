#!/usr/bin/env bash
# Train one MIL model on TCGA (10-fold cross-validation, 3-class WHO 2021).
#
# Usage: scripts/train.sh <BACKBONE> <MODEL> <MAG>
#   BACKBONE  uni | imagenet | hibou | ctranspath | lunit | conch_v1 | gigapath | optimus | virchow
#   MODEL     mean_mil | max_mil | att_mil | trans_mil | clam_sb | mamba_mil | dsmil | wikgmil | rrtmil
#   MAG       2.5x | 5x | 10x | 20x
#
# Environment (all optional): FEAT_ROOT (data/features), RESULTS_DIR (results), K (10),
#   MAX_EPOCHS (200), CSV_PATH, SPLIT_DIR, PYTHON, EXTRA_ARGS.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

[ "$#" -eq 3 ] || { sed -n '2,11p' "$0"; exit 1; }
BACKBONE=$1; MODEL_TYPE=$2; MAG=$3

TASK=tcga_3_class
LABEL=who2021
CSV_PATH=${CSV_PATH:-dataset_csv/tcga_2021_who_labels.csv}
SPLIT_DIR=${SPLIT_DIR:-splits/tcga_who_2021_100}
RESULTS_DIR=${RESULTS_DIR:-results}
IN_DIM=$(backbone_dim "$BACKBONE")

SAVE_EXP="tcga_${LABEL}/${BACKBONE}/${MODEL_TYPE}/${MAG}"
FEAT_DIR="${FEAT_ROOT:-data/features}/${BACKBONE}/tcga/${MAG}"

echo "Training: backbone=$BACKBONE (dim $IN_DIM) model=$MODEL_TYPE mag=$MAG -> ${RESULTS_DIR}/${SAVE_EXP}_s1"

COMMON=(--early_stopping --lr 1e-4 --k "${K:-10}" --max_epochs "${MAX_EPOCHS:-200}"
        --exp_code "$SAVE_EXP" --results_dir "$RESULTS_DIR" --task "$TASK"
        --embed_dim "$IN_DIM" --weighted_sample --log_data
        --features_dir "$FEAT_DIR" --csv_path "$CSV_PATH" --split_dir "$SPLIT_DIR")

if [ "$MODEL_TYPE" = "clam_sb" ]; then
    # CLAM has its own trainer with instance-level clustering options.
    "$PY" pipeline/main_clam.py "${COMMON[@]}" --model_type clam_sb --bag_loss ce --inst_loss svm \
        --subtyping --no_inst_cluster ${EXTRA_ARGS:-}
else
    "$PY" pipeline/main.py "${COMMON[@]}" --model_type "$MODEL_TYPE" ${EXTRA_ARGS:-}
fi
