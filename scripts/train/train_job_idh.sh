#!/bin/bash

#SBATCH -p gpu
#SBATCH -A r00917
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=sinnani@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00

nvidia-smi

module load conda
source activate /N/u/sinnani/BigRed200/MambaMIL/mambamil
cd /N/slate/sinnani/MambaMIL

WANDB_MODE=dryrun

# Define your variables

model=$1
split_dir=$2

mag="20x"
organ="gbm_lgg"
backbone="uni"
label="idh"
task="tcga_2_class"

if [ "$backbone" = "resnet" ]; then
    in_dim=1024
elif [ "$backbone" = "imagenet" ]; then
    in_dim=1024
elif [ "$backbone" = "ctranspath" ]; then
    in_dim=768
elif [ "$backbone" = "lunit" ]; then
    in_dim=384
elif [ "$backbone" = "conch_v1" ]; then
    in_dim=512
elif [ "$backbone" = "uni" ]; then
    in_dim=1024
elif [ "$backbone" = "gigapath" ]; then
    in_dim=1536
elif [ "$backbone" = "virchow" ]; then
    in_dim=2560
elif [ "$backbone" = "hibou" ]; then
    in_dim=1024
elif [ "$backbone" = "optimus" ]; then
    in_dim=1536
else
    echo "Error: Unsupported backbone '$backbone'"
    exit 1
fi

# Define save_exp for each combination
save_exp="tcga_${label}/${backbone}/${split_dir}/${model}/${mag}"
echo "$save_exp"

# python main.py --drop_out 0 --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code "$save_exp" --task "$task" --patch_size 256 --backbone "$backbone" --results_dir results --model_type "$model" \
#     --log_data --in_dim "$in_dim" --split_dir splits/${split_dir} --csv_path dataset_csv/df_idh_label.csv --features_dir /N/project/histopath/tcga_${organ}_features/features_${backbone}_new/tcga_${organ}_${mag} \
#     --data_root_dir /N/slate/sinnani/MambaMIL/


#python main.py --drop_out 0 --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code tcga_histology_uni_wikgmil_20x_s1 --task tcga_3_class --patch_size 256 --backbone uni --results_dir results --model_type wikgmil --log_data --in_dim 1024 --split_dir splits/task_who_2021_100 --csv_path dataset_csv/df_2021_histology_labels.csv --features_dir /mnt/e/ssl/MambaMIL/features/tcga_20x_uni/ --data_root_dir /mnt/e/ssl/MambaMIL 


python eval_miccai.py --k 10 --models_exp_code "tcga_${label}/${backbone}/${split_dir}/${model}/${mag}_s1" --save_exp_code tcga_idh/${backbone}/${split_dir}/${model}/${mag} --task tcga_2_class --model_type ${model} \
    --results_dir results --data_root_dir /N/slate/sinnani/MambaMIL/ --split test --features_dir /N/project/histopath/tcga_${organ}_features/features_${backbone}_new/tcga_${organ}_${mag}/ \
    --csv_path dataset_csv/df_idh_label.csv --splits_dir splits/${split_dir} --in_dim ${in_dim} --embed_dim ${in_dim}

python eval_miccai.py --k 10 --models_exp_code "tcga_${label}/${backbone}/${split_dir}/${model}/${mag}_s1" --save_exp_code ebrains_idh/${backbone}/${split_dir}/${model}/${mag} --task tcga_2_class --model_type ${model} \
    --results_dir results --data_root_dir /N/slate/sinnani/MambaMIL/ --split test --features_dir /N/project/histopath/tcga_gbm_lgg_features/features_uni/ebrain_20x_uni/ \
    --csv_path dataset_csv/ebrain_df_label_idh.csv --splits_dir splits/ebrain_who_2021_100/ --in_dim ${in_dim} --embed_dim ${in_dim}

python eval_miccai.py --k 10 --models_exp_code "tcga_${label}/${backbone}/${split_dir}/${model}/${mag}_s1" --save_exp_code penn_idh/${backbone}/${split_dir}/${model}/${mag} --task tcga_2_class --model_type ${model} \
    --results_dir results --data_root_dir /N/slate/sinnani/MambaMIL/ --split test --features_dir /N/project/histopath/tcga_gbm_lgg_features/features_uni/penn_20x_uni/features_20x_256_uni/ \
    --csv_path dataset_csv/idh_upenn_data_curated_data.csv --splits_dir splits/task_upenn_idh_100/ --in_dim ${in_dim} --embed_dim ${in_dim}