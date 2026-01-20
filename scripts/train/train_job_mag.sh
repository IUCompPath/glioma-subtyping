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
#SBATCH --time=48:00:00

module load anaconda
conda activate /N/u/sinnani/BigRed200/MambaMIL/mambamil
cd /N/slate/sinnani/MambaMIL

WANDB_MODE=dryrun
# Define your variables
task=$1
csv_path=$2
label=$3
model=$4
backbone=$5

mags=("2.5x" "10x")
#model_names=("max_mil" "mean_mil" "att_mil" "trans_mil" "mamba_mil") # Replace with your actual model names
#backbone="gigapath" # Replace with your actual backbones
organ="gbm_lgg"
# Set in_dim based on backbone
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


# Iterate over the combinations

for mag in "${mags[@]}"
do
    # Define save_exp for each combination
    save_exp="tcga_${label}/${backbone}/${model}/${mag}"
    echo "$save_exp"
    python main.py \
        --drop_out 0.0 \
        --early_stopping \
        --lr 2e-4 \
        --k 10 \
        --label_frac 1.0 \
        --exp_code "$save_exp" \
        --task "$task" \
        --patch_size 256 \
        --backbone "$backbone" \
        --results_dir results \
        --model_type "$model" \
        --log_data \
    	--in_dim "$in_dim" \
        --split_dir splits/task_who_2021_100 \
        --csv_path dataset_csv/${csv_path} \
        --features_dir /N/project/histopath/tcga_${organ}_features/features_${backbone}/tcga_${organ}_${mag} \
        --data_root_dir /N/u/sinnani/BigRed200/MambaMIL/
done
