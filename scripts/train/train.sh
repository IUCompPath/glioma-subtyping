#!/bin/bash

#SBATCH -J grade_10x
#SBATCH -p gpu
#SBATCH -A r00917
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sinnani@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00

#Load any modules that your program needs
module load anaconda

#Run your program
source activate /N/u/sinnani/BigRed200/MambaMIL/mambamil
cd /N/u/sinnani/BigRed200/MambaMIL
python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code tcga_idh --task tcga_2_class --results_dir results \
--model_type mamba_mil --log_data --split_dir splits/tcga_100 --csv_path dataset_csv/df.csv --features_dir /N/slate/features/