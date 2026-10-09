#!/bin/bash
# Example SLURM wrapper around scripts/train.sh. Fill in the placeholders for your cluster.
#
# Usage: sbatch scripts/slurm/train_job.sbatch <BACKBONE> <MODEL> <MAG>
#SBATCH -p <PARTITION>
#SBATCH -A <ACCOUNT>
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glioma_subtyping

export WANDB_MODE=offline
bash scripts/train.sh "$@"
