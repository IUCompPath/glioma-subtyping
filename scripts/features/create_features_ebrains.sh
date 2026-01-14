#!/bin/bash

#SBATCH -p gpu
#SBATCH -A r00917
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sinnani@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00

nvidia-smi

module load anaconda
export HF_TOKEN='' # replace with your actual token

conda activate /N/u/sinnani/BigRed200/clam/clam_latest
cd /N/slate/sinnani/clam
python extract_features_fp.py --data_h5_dir /N/project/histopath/tcga_gbm_lgg/patches_ebrain/patches_$1 \
--data_slide_dir /N/project/histopath/ebrains --csv_path dataset_csv/ebrain_data_info.csv \
--feat_dir /N/project/histopath/tcga_gbm_lgg_features/features_lunit/ebrains_new_$1 --batch_size $2 --slide_ext .ndpi --target_patch_size 224 --model_name lunit

