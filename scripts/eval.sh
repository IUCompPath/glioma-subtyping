#!/bin/bash

WANDB_MODE=dryrun

# Positional Arguments
TASK=$1         # e.g., tcga_3_class
LABEL=$2        # e.g., histology
MODEL=$3        # e.g., mamba_mil
BACKBONE=$4     # e.g., uni

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

# Define Magnifications
mags=("20x" "10x" "5x" "2.5x")

# --- 2. Iterate over TCGA Evaluation ---
echo "Starting TCGA Evaluation Grid..."
for mag in "${mags[@]}"
do
   save_exp="tcga_${LABEL}/${BACKBONE}/${MODEL}/${mag}"
   echo "Running: $save_exp"
   
   python eval_miccai.py \
    --k 10 \
    --models_exp_code "${save_exp}_s1" \
    --save_exp_code "${save_exp}" \
    --task "$TASK" \
    --model_type "$MODEL" \
    --results_dir results \
    --data_root_dir /N/slate/sinnani/clam \
    --split test \
    --features_dir "/N/project/histopath/tcga_gbm_lgg_features/features_${BACKBONE}/tcga_gbm_lgg_${mag}/" \
    --csv_path dataset_csv/df_2021_histology_labels.csv \
    --splits_dir splits/task_who_2021_100/ \
    --embed_dim ${in_dim}
done
SAVE_EXP="tcga_${LABEL}/${BACKBONE}/${MODEL_TYPE}/${MAG}"
# --- 3. Iterate over EBRAINS Evaluation ---
echo "Starting EBRAINS Evaluation Grid..."
for mag in "${mags[@]}"
do
    save_exp="ebrains_${LABEL}/${BACKBONE}/${MODEL}/${mag}"
    models_ckpt="tcga_${LABEL}/${BACKBONE}/${MODEL}/${mag}_s1"
    
    python eval_miccai.py \
     --k 10 \
     --models_exp_code "$models_ckpt" \
     --save_exp_code "${save_exp}" \
     --task "$TASK" \
     --model_type "$MODEL" \
     --results_dir results \
     --data_root_dir /N/slate/sinnani/clam \
     --split test \
     --features_dir "/N/project/histopath/tcga_gbm_lgg_features/features_${BACKBONE}/ebrains_${mag}/" \
     --csv_path dataset_csv/ebrain_df_label_histology.csv \
     --splits_dir splits/ebrain_who_2021_100/ \
     --embed_dim ${in_dim}
done

# --- 4. Iterate over IPD Evaluation ---
echo "Starting IPD Evaluation Grid..."
for mag in "${mags[@]}"
do
    save_exp="ipd_${LABEL}/${BACKBONE}/${MODEL}/${mag}"
    # Clean mag string if necessary (preserving your original substitution logic)
    model_mag=${mag//_1/} 
    models_ckpt="tcga_${LABEL}/${BACKBONE}/${MODEL}/${model_mag}_s1"

    python eval_miccai.py \
     --k 10 \
     --models_exp_code "$models_ckpt" \
     --save_exp_code "${save_exp}" \
     --task "$TASK" \
     --model_type "$MODEL" \
     --results_dir results \
     --data_root_dir /N/slate/sinnani/clam \
     --split test \
     --features_dir "/N/project/histopath/ipd_dataset/features/${BACKBONE}/${mag}/" \
     --csv_path dataset_csv/ipd_labels_refined.csv \
     --splits_dir splits/ipd_who_2021_100/ \
     --embed_dim ${in_dim}
done