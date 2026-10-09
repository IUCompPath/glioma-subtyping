#!/usr/bin/env bash
# Shared helpers for the pipeline scripts. Source this file: `source "$(dirname "$0")/lib.sh"`.

# Feature dimension produced by each backbone (must match models/builder.py).
backbone_dim() {
    case "$1" in
        uni|imagenet|resnet|hibou) echo 1024 ;;
        ctranspath)                echo 768  ;;
        lunit)                     echo 384  ;;
        conch_v1)                  echo 512  ;;
        gigapath|optimus)          echo 1536 ;;
        virchow)                   echo 2560 ;;
        *) echo "Error: unsupported backbone '$1'" >&2; return 1 ;;
    esac
}

# Name understood by extract_features_fp*.py for a backbone directory name.
backbone_model_name() {
    case "$1" in
        imagenet) echo resnet ;;
        *)        echo "$1" ;;
    esac
}

# Python entry point: respects $PYTHON, defaults to the active interpreter.
PY="${PYTHON:-python}"

# Repo root on PYTHONPATH so entry points in pipeline/ and tools/ can import utils, models, ...
# Run all scripts from the repository root.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
