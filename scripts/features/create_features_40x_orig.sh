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
#$ -l gpu=1
#$ -pe threaded 4
############# ################# END OF DEFAULT EMBEDDED SGE COMMANDS #######################

##module load cuda/11.3
nvidia-smi
source activate clama
cd /cbica/home/innanis/ssl/CLAM_vit
python extract_features_fp_vit_orig.py --data_h5_dir patches/tcga_gbm_lgg_40x_$1 --data_slide_dir tcga_gbm_lgg_40x_wsi_all --csv_path dataset_csv/all_slides_gbm_lgg_40x.csv --feat_dir features/retccl_gbm_lgg_$1_orig --batch_size $2 --slide_ext .svs --custom_downsample 2 --target_patch_size 224
