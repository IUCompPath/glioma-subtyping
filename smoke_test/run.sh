#!/usr/bin/env bash
# Fast smoke test of training -> evaluation -> multi-scale late fusion on pre-extracted features.
# No slides, no GPU downloads, no token: runs in a few minutes.
#
#   * gigapath: REAL TCGA features (12 slides, 32 sampled patches each), committed in smoke_test/data
#   * other backbones (default: uni): synthetic class-dependent features generated on the fly
#
# Usage: smoke_test/run.sh            # from the repository root
# Environment (optional): BACKBONES ("gigapath uni"), MODELS ("att_mil mamba_mil clam_sb"),
#   MAGS ("2.5x 5x 10x 20x"), OUT (smoke_test/out), PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

BACKBONES=${BACKBONES:-"gigapath uni"}
MODELS=${MODELS:-"att_mil mamba_mil clam_sb"}
MAGS=${MAGS:-"2.5x 5x 10x 20x"}
OUT=${OUT:-smoke_test/out}
DATA=smoke_test/data
FOLDS=2
rm -rf "$OUT"

for BB in $BACKBONES; do
    if [ ! -d "$DATA/features/$BB" ]; then
        "$PY" tools/make_synthetic_features.py --data "$DATA" --backbone "$BB" --dim "$(backbone_dim "$BB")"
    fi
done

export FEAT_ROOT="$DATA/features" RESULTS_DIR="$OUT/results" EVAL_DIR="$OUT/eval_results" \
       CSV_PATH="$DATA/labels.csv" SPLIT_DIR="$DATA/splits" K=$FOLDS MAX_EPOCHS=5

for BB in $BACKBONES; do
    for MODEL in $MODELS; do
        for MAG in $MAGS; do
            scripts/train.sh "$BB" "$MODEL" "$MAG"
            scripts/eval.sh "$MODEL" "$BB" "$MAG" tcga
        done
        "$PY" pipeline/ensemble_script.py who2021 "$BB" "$MODEL" --eval_dir "$EVAL_DIR" --sources tcga
    done
done

"$PY" smoke_test/check.py --out "$OUT" --data "$DATA" --backbones $BACKBONES --models $MODELS --mags $MAGS --folds $FOLDS
echo "SMOKE TEST PASSED"
