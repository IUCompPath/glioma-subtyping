#!/bin/bash

# Activate environment
conda activate glioma_subtyping

# Arguments
dataset=$1     # tcga | ebrains | ipd
mag=$2         # e.g. 40x, 20x
ps=$3          # patch size / step size
pl=$4          # patch level

echo "Dataset: ${dataset}"
echo "Magnification: ${mag}"

python create_patches_fp.py \
  --source data/slides/${dataset} \
  --step_size ${ps} \
  --patch_size ${ps} \
  --patch \
  --seg \
  --save_dir data/patches/${dataset}/${mag} \
  --preset ${dataset}.csv \
  --patch_level ${pl}
