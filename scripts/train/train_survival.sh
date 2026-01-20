#!/bin/bash

#SBATCH -J surv
#SBATCH -p gpu
#SBATCH -A r00917
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=sinnani@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00

# Load any modules that your program needs
module load conda

# Activate the conda environment
conda activate /N/u/sinnani/BigRed200/MambaMIL/mambamil

# Change directory to the project folder
cd /N/slate/sinnani/MambaMIL

backbone=$1
model=$2
if [[ "$backbone" = "uni_new" || "$backbone" = "imagenet" || "$backbone" = "hibou" ]]; then
    in_dim=1024
elif [[ "$backbone" = "gigapath" || "$backbone" = "optimus" || "$backbone" = "hoptimus1" || "$backbone" = "uni_v2" ]]; then
    in_dim=1536
elif [ "$backbone" = "lunit" ]; then
    in_dim=384
elif [[ "$backbone" = "ctranspath" || "$backbone" = "conch_v15" || "$backbone" = "titan" || "$backbone" = "chief" ]]; then
    in_dim=768
elif [[ "$backbone" = "virchow" || "$backbone" = "virchow2" ]]; then
    in_dim=2560
elif [ "$backbone" = "conch_v1" ]; then
    in_dim=512
elif [ "$backbone" = "prism" ]; then
    in_dim=1280
else
    echo "Error: Unsupported backbone '$backbone'"
    exit 1
fi


# Run your program
python main_survival.py --drop_out 0.0 --data_root_dir /N/slate/sinnani/MambaMIL --k_start $3 \
--early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code tcga_gbmlgg_survival/${model}/${backbone} \
--patch_size 256 --task TCGA_gbmlgg_survival --backbone tcga_20x --results_dir results --model_type ${model} \
--log_data --split_dir splits/tcga_gbmlgg_survival_kfold --in_dim ${in_dim} --feat_dir /N/project/histopath/tcga_gbm_lgg_features/features_${backbone}/tcga_gbm_lgg_20x/pt_files
