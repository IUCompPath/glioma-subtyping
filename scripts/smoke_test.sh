#!/usr/bin/env bash
# End-to-end smoke test on a couple of real slides:
#   patching -> feature extraction -> training -> evaluation -> multi-magnification late fusion
# using the ImageNet ResNet-50 backbone (no gated weights / token needed).
#
# Usage: scripts/smoke_test.sh <SLIDE_DIR_OR_FILES...>
#   e.g. scripts/smoke_test.sh /path/to/slides            # takes the first 2 slides
#        scripts/smoke_test.sh a.svs b.svs
#
# Environment (optional): OUT (smoke_out), MODELS ("att_mil mamba_mil clam_sb"),
#   MAGS ("20x 10x"), PRESET (bwh_biopsy.csv: permissive tissue filter), PYTHON.
# Labels/splits are dummies (see tools/prepare_smoke_data.py): only plumbing is checked.
set -euo pipefail
[ "$#" -ge 1 ] || { sed -n '2,13p' "$0"; exit 1; }
SLIDES=(); for p in "$@"; do SLIDES+=("$(realpath "$p")"); done   # absolute, since we cd below
cd "$(dirname "$0")/.."
source scripts/lib.sh
OUT=${OUT:-smoke_out}
MODELS=${MODELS:-"att_mil mamba_mil clam_sb"}
MAGS=${MAGS:-"20x 10x"}
PRESET=${PRESET:-bwh_biopsy.csv}
BACKBONE=imagenet
rm -rf "$OUT"

"$PY" tools/prepare_smoke_data.py --slides "${SLIDES[@]}" --out "$OUT"
SLIDE_EXT=$(ls "$OUT/wsi/smoke" | head -1 | sed 's/.*\(\.[^.]*\)$/\1/')
export WSI_ROOT="$OUT/wsi" PATCH_ROOT="$OUT/patches" FEAT_ROOT="$OUT/features" RESULTS_DIR="$OUT/results" \
       EVAL_DIR="$OUT/eval_results" SLIDE_EXT CSV_PATH="$OUT/labels.csv" SPLIT_DIR="$OUT/splits" K=1 MAX_EPOCHS=3

for MAG in $MAGS; do
    scripts/patches/create_patches.sh smoke "$MAG" "$PRESET"
    scripts/features/create_features.sh "$MAG" 64 "$OUT/labels.csv" "$BACKBONE" smoke
done

# The training/eval scripts address cohorts by name; point the smoke features at 'tcga'.
for MAG in $MAGS; do
    mkdir -p "$OUT/features/$BACKBONE/tcga"
    ln -sfn "$(pwd)/$OUT/features/$BACKBONE/smoke/$MAG" "$OUT/features/$BACKBONE/tcga/$MAG"
done

for MODEL in $MODELS; do
    for MAG in $MAGS; do
        scripts/train.sh "$BACKBONE" "$MODEL" "$MAG"
        scripts/eval.sh "$MODEL" "$BACKBONE" "$MAG" tcga
    done
done

for MODEL in $MODELS; do
    "$PY" pipeline/ensemble_script.py who2021 "$BACKBONE" "$MODEL" --eval_dir "$OUT/eval_results" --sources tcga
done

# Verify every stage produced its artifacts.
"$PY" tools/check_smoke_outputs.py --out "$OUT" --models $MODELS --mags $MAGS
echo "SMOKE TEST PASSED"
