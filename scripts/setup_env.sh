#!/usr/bin/env bash
# Create the `glioma_subtyping` conda env and compile the Mamba CUDA kernels.
# Requires conda/mamba and an NVIDIA GPU + driver. Usage: scripts/setup_env.sh [ARCH]
#   ARCH = CUDA compute capability for the kernels, e.g. 8.0 (A100), 8.6 (A6000), 8.9 (RTX 6000 Ada)
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=${1:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}
SOLVER=$(command -v mamba || command -v conda)

"$SOLVER" env create -f environment.yml -y
PREFIX=$(conda env list | awk '$1=="glioma_subtyping"{print $NF}')
export PATH="$PREFIX/bin:$PATH" CUDA_HOME="$PREFIX" TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS=${MAX_JOBS:-8}
export MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE

pip install --no-build-isolation causal-conv1d==1.1.1
(cd mamba && pip install --no-build-isolation --no-deps .)
python - <<'PY'
import torch
from mamba.mamba_ssm import SRMamba
print("mamba OK:", SRMamba(d_model=64).cuda()(torch.randn(1, 8, 64).cuda()).shape)
PY
echo "Done. Activate with: conda activate glioma_subtyping"
