#!/bin/bash

# --- 1. Fixed Variable Assignments (No spaces around '=') ---
TASK='tcga_3_class'
CSV_PATH='tcga_2021_who_labels.csv'
LABEL='who2021'

# Positional Arguments
BACKBONE=$1    # e.g., uni, virchow, gigapath
MODEL_TYPE=$2  # e.g., clam_sb, trans_mil, mamba_mil
MAG=$3         # e.g., 20x, 10x, 5x, 2.5x

# --- 2. Corrected Case Statement ---
# The 'in' acts as the opening and 'esac' as the closing.
case "$BACKBONE" in 
    "uni"|"imagenet"|"hibou")
        in_dim=1024
        ;;
    "ctranspath")
        in_dim=768
        ;;
    "lunit")
        in_dim=384
        ;;
    "conch_v1")
        in_dim=512
        ;;
    "gigapath"|"optimus")
        in_dim=1536
        ;;
    "virchow")
        in_dim=2560
        ;;
    *)
        echo "Error: Unsupported backbone '$BACKBONE'"
        echo "Usage: ./train.sh <backbone> <model_type> <mag>"
        exit 1
        ;;
esac

# --- 3. Define Paths ---
SAVE_EXP="tcga_${LABEL}/${BACKBONE}/${MODEL_TYPE}/${MAG}"
FEAT_DIR="data/features/${BACKBONE}/${TASK}/${MAG}/pt_files/"

# --- 4. Print Status for Debugging ---
echo "-------------------------------------------------------"
echo "Training Workflow Initialized"
echo "Task:       $TASK"
echo "Backbone:   $BACKBONE (Resolved Dim: $in_dim)"
echo "Model:      $MODEL_TYPE"
echo "Mag:        $MAG"
echo "Exp Code:   $SAVE_EXP"
echo "Feature Dir: $FEAT_DIR"
echo "-------------------------------------------------------"

# --- 5. Execution ---
python main.py \
    --early_stopping \
    --lr 1e-4 \
    --k 10 \
    --label_frac 1.0 \
    --exp_code "$SAVE_EXP" \
    --bag_loss ce \
    --inst_loss svm \
    --task "$TASK" \
    --model_type "$MODEL_TYPE" \
    --log_data  \
    --embed_dim "$in_dim" \
    --weighted_sample \
    --features_dir "$FEAT_DIR" \
    --subtyping \
    --no_inst_cluster \
    --csv_path "dataset_csv/$CSV_PATH" \
    --split_dir task_who_2021_100 