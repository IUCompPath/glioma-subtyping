#!/usr/bin/env bash
# Late fusion (multi-magnification ensembles) for every backbone x model x cohort.
#
# Usage: scripts/eval_ensemble.sh <LABEL> [EVAL_DIR]
#   e.g. scripts/eval_ensemble.sh who2021
set -uo pipefail
source "$(dirname "$0")/lib.sh"

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { sed -n '2,6p' "$0"; exit 1; }
LABEL=$1
EVAL_DIR=${2:-eval_results}

BACKBONES=(uni imagenet hibou ctranspath lunit conch_v1 gigapath optimus virchow)
MODELS=(mean_mil max_mil att_mil trans_mil clam_sb mamba_mil dsmil wikgmil rrtmil)

for bb in "${BACKBONES[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "--- $bb | $model | $LABEL"
        "$PY" pipeline/ensemble_script.py "$LABEL" "$bb" "$model" --eval_dir "$EVAL_DIR" \
            || echo "(skipped: no results for $bb / $model)"
    done
done
