#!/bin/bash
#$ -S /bin/bash
nvidia-smi

# Input Arguments
MAG=$1          # e.g., 20x
BS=$2           # Batch Size, e.g., 128
CSV=$3          # CSV filename, e.g., slides.csv
BACKBONE=$4     # e.g., uni, virchow, hibou
DATASET=$5      # e.g., tcga_gbm
export HF_TOKEN='your_actual_token_here' # Ensure this is set for gated models

module load anaconda

# 1. Environment and Backbone Validation
# Added missing || and fixed spacing in the conditional check
if [[ "$BACKBONE" == "resnet" || "$BACKBONE" == "uni" || \
      "$BACKBONE" == "conch_v1" || "$BACKBONE" == "lunit" || \
      "$BACKBONE" == "ctranspath" || "$BACKBONE" == "gigapath" || \
      "$BACKBONE" == "optimus" || "$BACKBONE" == "virchow" || \
      "$BACKBONE" == "hibou" ]]; then
    
    source activate glioma_subtyping
else
    echo "Error: Unsupported backbone '$BACKBONE'"
    echo "Choose from: resnet, uni, conch_v1, lunit, ctranspath, gigapath, optimus, virchow, hibou"
    exit 1
fi

# 2. Define Directory Paths (Cleaned up for clarity)
H5_DIR="data/patches/${DATASET}/${MAG}"
SLIDE_DIR="data/wsi/${DATASET}"
CSV_PATH="dataset_csv/${CSV}"
FEAT_DIR="data/features/${BACKBONE}/${DATASET}/${MAG}"

echo "-------------------------------------------------------"
echo "Processing Dataset: $DATASET at $MAG"
echo "Model:             $BACKBONE"
echo "Output Directory:  $FEAT_DIR"
echo "-------------------------------------------------------"

# 3. Execution Logic
# Virchow and Hibou often require specific preprocessing wrappers (fp_virchow/fp_hibou)
if [[ "$BACKBONE" == "virchow" ]]; then
    SCRIPT="extract_features_fp_virchow.py"
elif [[ "$BACKBONE" == "hibou" ]]; then
    SCRIPT="extract_features_fp_hibou.py"
else
    SCRIPT="extract_features_fp.py"
fi

python "$SCRIPT" \
    --data_h5_dir "$H5_DIR" \
    --data_slide_dir "$SLIDE_DIR" \
    --csv_path "$CSV_PATH" \
    --feat_dir "$FEAT_DIR" \
    --batch_size "$BS" \
    --slide_ext .svs \
    --target_patch_size 224 \
    --model_name "$BACKBONE"