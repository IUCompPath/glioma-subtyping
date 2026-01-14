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

conda activate /N/slate/sinnani/prov-gigapath/env_gigapath
cd /N/u/sinnani/BigRed200/clam
python extract_features_fp.py --data_h5_dir /N/project/histopath/tcga_gbm_lgg/patches/tcga_gbm_lgg_40x_$1_all \
--data_slide_dir /N/project/histopath/tcga_gbm_lgg_40x_wsi --csv_path dataset_csv/$3.csv \
--feat_dir /N/project/histopath/tcga_gbm_lgg_features/features_virchow/tcga_gbm_lgg_$1 --batch_size $2 \
--slide_ext .svs --target_patch_size 224 --model_name virchow
