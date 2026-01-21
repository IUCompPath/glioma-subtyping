#!/usr/bin/env bash
set -euo pipefail

# run_all_ensembles.sh
# Usage:
#   ./run_all_ensembles.sh <LABEL> [OUT_ROOT]
# Example:
#   ./run_all_ensembles.sh who2021
#   ./run_all_ensembles.sh who2021 /home/user/experiments

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <LABEL> [OUT_ROOT]"
  exit 1
fi

LABEL="$1"
OUT_ROOT="${2:-.}"

# Paths & script
PY_SCRIPT="ensemble_script.py"

# Sources to iterate (these correspond to directory prefixes: tcga_<LABEL>, ebrains_<LABEL>, ipd_<LABEL>)
SOURCES=( "tcga" "ebrains" "ipd" )

# Backbones to iterate
BACKBONES=( uni imagenet hibou ctranspath lunit conch_v1 gigapath optimus virchow )

# Models to iterate (included from your list + clam_sb)
MODELS=( mean_mil max_mil att_mil trans_mil clam_sb mamba_mil dsmil wikgmil rrtmil )

# Check python script exists
if [ ! -f "$PY_SCRIPT" ]; then
  echo "Warning: $PY_SCRIPT not found in current directory. Ensure it's present or adjust PY_SCRIPT variable."
fi

echo "Running ensembles for label='$LABEL' under out_root='$OUT_ROOT'"
echo "Backbones: ${BACKBONES[*]}"
echo "Models:    ${MODELS[*]}"
echo "Sources:   ${SOURCES[*]}"
echo

# Loop over backbones, models and sources
for bb in "${BACKBONES[@]}"; do
  for model in "${MODELS[@]}"; do
    for src in "${SOURCES[@]}"; do
      echo "------------------------------------------------"
      echo "RUNNING: Source: $src | Backbone: $bb | Model: $model | Label: $LABEL"
      # Call the python ensemble script for this single source
      # ensemble_script.py expects: <label> <backbone> <model> [--out_root OUT_ROOT] [--sources ...]
      # We'll pass --out_root and set --sources to the current src so the script processes only that source
      python "$PY_SCRIPT" "$LABEL" "$bb" "$model" --out_root "$OUT_ROOT" --sources "$src"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "Warning: ensemble_script.py exited with code $rc for $src | $bb | $model"
        # continue to next run rather than aborting everything
      fi
      echo
    done
  done
done

echo "All done."
