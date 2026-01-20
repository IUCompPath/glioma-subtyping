#!/bin/bash

# Positional Arguments
MODEL=$1        # e.g., mamba_mil
BACKBONE=$2     # e.g., uni
MAG=$3          # e.g., 20x

# Constant Metadata
TASK='tcga_3_class'
LABEL='who2021'
mags=($MAG) # Array for loop compatibility

# --- 1. Dynamic Input Dimension ---
case "$BACKBONE" in
    "uni"|"imagenet"|"hibou") in_dim=1024 ;;
    "ctranspath")            in_dim=768  ;;
    "lunit")                 in_dim=384  ;;
    "conch_v1")              in_dim=512  ;;
    "gigapath"|"optimus")    in_dim=1536 ;;
    "virchow")               in_dim=2560 ;;
    *) echo "Error: Unsupported backbone '$BACKBONE'"; exit 1 ;;
esac

# --- 2. TCGA Evaluation ---
echo "Starting TCGA Evaluation for $BACKBONE | $MODEL | $MAG"
for mag in "${mags[@]}"
do
   save_exp="tcga_${LABEL}/${BACKBONE}/${MODEL}/${mag}"
   python eval.py \
    --k 10 \
    --models_exp_code "${save_exp}_s1" \
    --save_exp_code "${save_exp}" \
    --task "$TASK" \
    --model_type "$MODEL" \
    --results_dir results \
    --split test \
    --features_dir "data/tcga/features/${BACKBONE}/${TASK}/${mag}/pt_files/" \
    --csv_path dataset_csv/df_2021_histology_labels.csv \
    --splits_dir splits/task_who_2021_100/ \
    --embed_dim ${in_dim}
done

# --- 3. EBRAINS Evaluation (External Validation) ---
echo "Starting EBRAINS Evaluation..."
for mag in "${mags[@]}"
do
    save_exp="ebrains_${LABEL}/${BACKBONE}/${MODEL}/${mag}"
    models_ckpt="tcga_${LABEL}/${BACKBONE}/${MODEL}/${mag}_s1"
    
    python eval.py \
     --k 10 \
     --models_exp_code "$models_ckpt" \
     --save_exp_code "${save_exp}" \
     --task "$TASK" \
     --model_type "$MODEL" \
     --results_dir results \
     --split test \
     --features_dir "data/ebrains/features/${BACKBONE}/${TASK}/${mag}/pt_files/" \
     --csv_path dataset_csv/ebrain_df_label_histology.csv \
     --splits_dir splits/ebrain_who_2021_100/ \
     --embed_dim ${in_dim}
done

# --- 4. IPD Evaluation (External Validation) ---
echo "Starting IPD Evaluation..."
for mag in "${mags[@]}"
do
    save_exp="ipd_${LABEL}/${BACKBONE}/${MODEL}/${mag}"
    models_ckpt="tcga_${LABEL}/${BACKBONE}/${MODEL}/${mag}_s1"

    python eval.py \
     --k 10 \
     --models_exp_code "$models_ckpt" \
     --save_exp_code "${save_exp}" \
     --task "$TASK" \
     --model_type "$MODEL" \
     --results_dir results \
     --split test \
     --features_dir "data/ipd/features/${BACKBONE}/${TASK}/${mag}/pt_files/" \
     --csv_path dataset_csv/ipd_labels_refined.csv \
     --splits_dir splits/ipd_who_2021_100/ \
     --embed_dim ${in_dim}
done