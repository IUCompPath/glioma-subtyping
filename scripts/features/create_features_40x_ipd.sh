#!/bin/bash

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

nvidia-smi

module load conda

mag=$1
bs=$2
csv=$3
backbone=$4
organ=$5
export HF_TOKEN='' # replace with your actual token
export UNI_CKPT_PATH='checkpoints/pytorch_model.bin'

if [[ "$backbone" == "resnet" || "$backbone" == "uni" || "$backbone" == "conch_v1" || "$backbone" == "lunit" ]]; then
    conda activate /N/u/sinnani/BigRed200/clam/clam_latest
    echo "activated clam_latest"
elif [[ "$backbone" == "gigapath" || "$backbone" == "optimus" || "$backbone" == "virchow" || "$backbone" == "hibou" ]]; then
    conda activate /N/slate/sinnani/prov-gigapath/env_gigapath
else
    echo "Unsupported backbone: $backbone"
    exit 1
fi

cd /N/slate/sinnani/clam


if [[ "$backbone" == "virchow" ]]; then
    python extract_features_fp_virchow.py --data_h5_dir /N/project/histopath/patches/tcga_${organ}_40x_${mag} \
    --data_slide_dir /N/slate/sinnani/clam/tcga_${organ}_40x_wsi --csv_path dataset_csv/${organ}/tcga_${organ}_40x_remaining.csv \
    --feat_dir /N/project/histopath/tcga_${organ}_features/features_${backbone}/tcga_${organ}_${mag} --batch_size ${bs} \
    --slide_ext .svs --target_patch_size 224 --model_name ${backbone}
elif [[ "$backbone" == "hibou" ]]; then
    python extract_features_fp_hibou.py --data_h5_dir /N/project/histopath/patches/tcga_${organ}_40x_${mag} \
    --data_slide_dir /N/slate/sinnani/clam/tcga_${organ}_40x_wsi --csv_path dataset_csv/${organ}/tcga_${organ}_40x_remaining.csv \
    --feat_dir /N/project/histopath/tcga_${organ}_features/features_${backbone}/tcga_${organ}_${mag} --batch_size ${bs} \
    --slide_ext .svs --target_patch_size 224 --model_name ${backbone}
else
    python extract_features_fp.py --data_h5_dir /N/project/histopath/patches/tcga_${organ}_40x_${mag} \
    --data_slide_dir /N/slate/sinnani/clam/tcga_${organ}_40x_wsi --csv_path dataset_csv/${organ}/tcga_${organ}_40x_remaining.csv \
    --feat_dir /N/project/histopath/tcga_${organ}_features/features_${backbone}/tcga_${organ}_${mag} --batch_size ${bs} \
    --slide_ext .svs --target_patch_size 224 --model_name ${backbone}
fi

