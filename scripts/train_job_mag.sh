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
source activate /N/u/sinnani/BigRed200/clam/clam_latest

# Navigate to the project directory
cd /N/slate/sinnani/clam


WANDB_MODE=dryrun
# Get command-line arguments
task=$1
csv_path=$2
label=$3

mags=("10x" "2.5x")
backbone=$4 # Replace with your actual backbones

if [ "$backbone" = "uni" ]; then
    in_dim=1024
elif [ "$backbone" = "ctranspath" ]; then
    in_dim=768
elif [ "$backbone" = "lunit" ]; then
    in_dim=384
elif [ "$backbone" = "imagenet" ]; then
    in_dim=1024
elif [ "$backbone" = "gigapath" ]; then
    in_dim=1536
elif [ "$backbone" = "virchow" ]; then
    in_dim=2560
elif [ "$backbone" = "conch_v1" ]; then
    in_dim=512
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
    save_exp="tcga_${label}/${backbone}/clam_sb/${mag}"
    echo "$save_exp"
    # Run your program with command-line arguments
    python main.py \
        --early_stopping \
        --lr 1e-4 \
        --k 10 \
        --label_frac 1.0 \
        --exp_code "$save_exp" \
        --bag_loss ce \
        --inst_loss svm \
        --task $task \
        --model_type clam_sb \
        --log_data \
        --data_root_dir /N/u/project/histopath/ \
        --embed_dim "$in_dim" \
        --weighted_sample \
        --features_dir /N/project/histopath/tcga_gbm_lgg_features/features_${backbone}/tcga_gbm_lgg_${mag}/ \
        --subtyping \
        --no_inst_cluster \
        --csv_path dataset_csv/$csv_path \
        --split_dir task_who_2021_100
done
