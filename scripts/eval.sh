#!/bin/bash

#SBATCH -p gpu
#SBATCH -A r00917
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=sinnani@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00

module load conda
conda activate /N/u/sinnani/BigRed200/clam/clam_latest

# Navigate to the project directory
cd /N/slate/sinnani/clam

WANDB_MODE=dryrun
# Define your variables
task=$1
label=$2
model=$3
mags=("20x_1" "20x")
# mags=("20x_1" "20x" "10x" "5x" "2.5x" "10x_1" "5x_1" "2.5x_1")
#model_names=("max_mil" "mean_mil" "att_mil" "trans_mil" "mamba_mil") # Replace with your actual model names
backbone=$4 # Replace with your actual backbones

if [ "$backbone" = "uni" ]; then
    in_dim=1024
elif [ "$backbone" = "imagenet" ]; then
    in_dim=1024
elif [ "$backbone" = "lunit" ]; then
    in_dim=384
elif [ "$backbone" = "ctranspath" ]; then
    in_dim=768
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
# for mag in "${mags[@]}"
# do
#    # Define save_exp for each combination
#    save_exp="tcga_${label}/${backbone}/${model}/${mag}"
#    echo "$save_exp"
#    echo "tcga_histology_${backbone}_${model}_${mag}"
#    python eval_miccai.py --k 10 --models_exp_code ${save_exp}_s1 --save_exp_code ${save_exp} --task tcga_3_class --model_type clam_sb \
#    --results_dir results --data_root_dir /N/slate/sinnani/clam --split test --features_dir /N/project/histopath/tcga_gbm_lgg_features/features_${backbone}/tcga_gbm_lgg_${mag}/ \
#    --csv_path dataset_csv/df_2021_histology_labels.csv --splits_dir splits/task_who_2021_100/ --embed_dim ${in_dim}
# done

# for mag in "${mags[@]}"
# do
#     # Define save_exp for each combination
#     save_exp="ebrains_${label}/${backbone}/${model}/${mag}"
#     echo "$save_exp"
#     python eval_miccai.py --k 10 --models_exp_code tcga_${label}/${backbone}/${model}/${mag}_s1 --save_exp_code ${save_exp} --task tcga_3_class --model_type clam_sb \
#     --results_dir results --data_root_dir /N/slate/sinnani/clam --split test --features_dir /N/project/histopath/tcga_gbm_lgg_features/features_${backbone}/ebrains_${mag}/ \
#     --csv_path dataset_csv/ebrain_df_label_histology.csv --splits_dir splits/ebrain_who_2021_100/ --embed_dim ${in_dim}
# done


for mag in "${mags[@]}"
do
    # Define save_exp for each combination
    save_exp="ipd_${label}/${backbone}/${model}/${mag}"
    echo "$save_exp"
    model_mag=${mag//_1/}
    python eval_miccai.py --k 10 --models_exp_code tcga_${label}/${backbone}/${model}/${model_mag}_s1 --save_exp_code ${save_exp} --task tcga_3_class --model_type clam_sb \
    --results_dir results --data_root_dir /N/slate/sinnani/clam --split test --features_dir /N/project/histopath/ipd_dataset/features/${backbone}/${mag}/ \
    --csv_path dataset_csv/ipd_labels_refined.csv --splits_dir splits/ipd_who_2021_100/ --embed_dim ${in_dim}
done

    # python eval_miccai.py --k 10 --models_exp_code tcga_histology/optimus/att_mil/2.5x_s1 --save_exp_code ipd_histology/optimus/att_mil/2.5x --task tcga_3_class --model_type att_mil
    # --results_dir results --data_root_dir /N/slate/sinnani/clam/ --split test --features_dir /N/project/histopath/ipd_dataset/features/optimus/2.5x/ 
    # --csv_path dataset_csv/ipd_labels_refined.csv --splits_dir splits/ipd_who_2021_100/ --in_dim 1536 --embed_dim 1536

