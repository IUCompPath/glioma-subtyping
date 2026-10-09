#!/usr/bin/env bash
# Segment tissue and extract patch coordinates at a target magnification.
#
# Usage: scripts/patches/create_patches.sh <DATASET> <MAG> [PRESET]
#   DATASET  tcga | ebrains | ipd (any folder under $WSI_ROOT)
#   MAG      2.5x | 5x | 10x | 20x   (per-slide pyramid level is resolved automatically;
#                                     slides scanned at a finer level are patched with a
#                                     proportionally larger window and resized later)
#   PRESET   CSV in presets/ (default: <DATASET>.csv)
#
# Environment: WSI_ROOT (default data/wsi), PATCH_ROOT (default data/patches), PYTHON.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

[ "$#" -ge 2 ] || { sed -n '2,12p' "$0"; exit 1; }
DATASET=$1
MAG=$2
PRESET=${3:-${DATASET}.csv}
WSI_ROOT=${WSI_ROOT:-data/wsi}
PATCH_ROOT=${PATCH_ROOT:-data/patches}

"$PY" create_patches_fp.py \
    --source "${WSI_ROOT}/${DATASET}" \
    --save_dir "${PATCH_ROOT}/${DATASET}/${MAG}" \
    --preset "$PRESET" \
    --target_mag "${MAG%x}" \
    --step_size 256 --patch_size 256 \
    --seg --patch --stitch
