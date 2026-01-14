################################## START OF EMBEDDED SGE COMMANDS ##########################
#$ -S /bin/bash
##$ -cwd
#$ -N 20x_patch
#$ -M shubhinnani@gmail.com #### email to nofity with following options/scenarios
#$ -m a #### send mail in case the job is aborted
#$ -m b #### send mail when job begins
#$ -m e #### send mail when job ends
#$ -m s #### send mail when job is suspended
#$ -l h_vmem=128G
#$ -l A40=1
#$ -pe threaded 4
############# ################# END OF DEFAULT EMBEDDED SGE COMMANDS #######################

##module load cuda/11.3
nvidia-smi
source activate clama
cd /cbica/home/innanis/comp_space/clam_idh
python main.py --early_stopping --lr 1e-4 --k 10 --label_frac 1.0 --exp_code tcga_gbm_idh_$1_orig --bag_loss ce --inst_loss svm --task tcga_gbm_idh --model_type clam_sb --log_data --data_root_dir /cbica/home/innanis/clam_idh --weighted_sample --features_dir features/gbm_lgg_$1_orig
