#!/usr/bin/env bash
# Extract patch-level features for one dataset / magnification / backbone.
#
# Usage: scripts/features/create_features.sh <MAG> <BATCH_SIZE> <CSV_FILE> <BACKBONE> <DATASET>
#   CSV_FILE  file in dataset_csv/ (or an explicit path) listing the slides to process
#
# Gated models (uni, conch_v1, virchow, hibou, ...) need a Hugging Face token:
#   export HF_TOKEN=<your token>        # never commit it
#
# Environment: WSI_ROOT, PATCH_ROOT, FEAT_ROOT (default data/{wsi,patches,features}),
#              SLIDE_EXT (default: .svs, .ndpi for ebrains, .tiff for ipd), PYTHON.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

[ "$#" -eq 5 ] || { sed -n '2,12p' "$0"; exit 1; }
MAG=$1; BS=$2; CSV=$3; BACKBONE=$4; DATASET=$5

backbone_dim "$BACKBONE" > /dev/null
MODEL_NAME=$(backbone_model_name "$BACKBONE")
case "$DATASET" in
    ebrains) DEFAULT_EXT=.ndpi ;;
    ipd)     DEFAULT_EXT=.tiff ;;
    *)       DEFAULT_EXT=.svs ;;
esac
SLIDE_EXT=${SLIDE_EXT:-$DEFAULT_EXT}
[ -f "$CSV" ] || CSV="dataset_csv/${CSV}"

H5_DIR="${PATCH_ROOT:-data/patches}/${DATASET}/${MAG}"
SLIDE_DIR="${WSI_ROOT:-data/wsi}/${DATASET}"
FEAT_DIR="${FEAT_ROOT:-data/features}/${BACKBONE}/${DATASET}/${MAG}"

case "$BACKBONE" in
    virchow) SCRIPT=extract_features_fp_virchow.py ;;
    hibou)   SCRIPT=extract_features_fp_hibou.py ;;
    *)       SCRIPT=extract_features_fp.py ;;
esac

echo "Dataset: $DATASET @ $MAG | backbone: $BACKBONE | output: $FEAT_DIR"
"$PY" "$SCRIPT" \
    --data_h5_dir "$H5_DIR" \
    --data_slide_dir "$SLIDE_DIR" \
    --csv_path "$CSV" \
    --feat_dir "$FEAT_DIR" \
    --batch_size "$BS" \
    --slide_ext "$SLIDE_EXT" \
    --target_patch_size 224 \
    --model_name "$MODEL_NAME"
