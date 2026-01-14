#!/bin/bash

#SBATCH -A r00917
#SBATCH --job-name=unzip_file
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sinnani@iu.edu
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

cd /N/slate/sinnani/clam
module load conda
conda activate /N/u/sinnani/BigRed200/clam/clam_latest

mag=$1
ps=$2
pl=$3

echo "Magnification "${mag}
python create_patches_fp.py --source /N/project/histopath/ipd_dataset/slides --step_size ${ps} --patch_size ${ps} --patch --seg --save_dir /N/project/histopath/ipd_dataset/patches/${mag} --preset ipd_preset.csv --patch_level ${pl}