for i in tqdm(coad_read_slides):
    try:
        slide = openslide.OpenSlide(i)
    except:
        print(i)
    mag = slide.properties.get('aperio.AppMag')
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    slide_id = i.split('/')[-1]
    slidename = slide_id.split('.svs')[0]
    case_id = i.split('-01Z')[0].split('/')[-1]
    level_downsamples = slide.level_downsamples
    data = {'slide_id': slide_id,'slidename': slidename,'case_id': case_id,'mag': mag,'mpp_x': mpp_x,'mpp_y': mpp_y,'path':i,'level_downsamples':level_downsamples}
    data_list.append(data)


for i in tqdm(brca_slides):
    try:
        slide = openslide.OpenSlide(i)
    except:
        print(i)
    mag = slide.properties.get('aperio.AppMag')
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    slide_id = i.split('/')[-1]
    slidename = slide_id.split('.svs')[0]
    case_id = i.split('-01Z')[0].split('/')[-1]
    level_downsamples = slide.level_downsamples
    data = {'slide_id': slide_id,'slidename': slidename,'case_id': slidename,'mag': mag,'mpp_x': mpp_x,'mpp_y': mpp_y,'path':i,'level_downsamples':level_downsamples}
    data_list.append(data)

sbatch -J patches_40x_20x patches_40x.sh 20x 448 0
sbatch -J patches_40x_10x patches_40x.sh 10x 224 1
sbatch -J patches_40x_5x patches_40x.sh 5x 448 1
sbatch -J patches_40x_2.5x patches_40x.sh 2.5x 224 2

sbatch -J patches_20x_20x patches_20x.sh 20x 224 0
sbatch -J patches_20x_10x patches_20x.sh 10x 448 0
sbatch -J patches_20x_5x patches_20x.sh 5x 224 1
sbatch -J patches_20x_2.5x patches_20x.sh 2.5x 448 1


for i in if "resnet" "uni" "conch_v1""lunit" "gigapath" "optimus" "virchow" "hibou":
sbatch -J 40x20x create_features_40x.sh 20x 128 brca_csv $i brca
sbatch -J b40x10x create_features_40x.sh 10x 64 brca_csv $i brca
sbatch -J b40x5x create_features_40x.sh 5x 16 brca_csv $i brca
sbatch -J b40x2.5x create_features_40x.sh 2.5x 4 brca_csv $i brca
sbatch -J bct20x20x create_features_20x.sh 20x 128 brca_csv ctranspath brca
sbatch -J bct20x10x create_features_20x.sh 10x 64 brca_csv ctranspath brca
sbatch -J bct20x5x create_features_20x.sh 5x 16 brca_csv ctranspath brca
sbatch -J bct20x2.5x create_features_20x.sh 2.5x 4 brca_csv ctranspath brca
 
sbatch -J el20x create_features_ebrains.sh 20x 128
sbatch -J el10x create_features_ebrains.sh 10x 64
sbatch -J el5x create_features_ebrains.sh 5x 16
sbatch -J el2.5x create_features_ebrains.sh 2.5x 4

for i in "resnet" "uni" "conch_v1" "lunit" "gigapath" "optimus" "virchow" "hibou"; do
    first_char=${i:0:1}  
    sbatch -J ${first_char}4020x create_features_40x.sh 20x 128 brca_csv $i brca
    sbatch -J ${first_char}4010x create_features_40x.sh 10x 64 brca_csv $i brca
    sbatch -J ${first_char}b405x create_features_40x.sh 5x 16 brca_csv $i brca
    sbatch -J ${first_char}b402.5x create_features_40x.sh 2.5x 4 brca_csv $i brca
    sbatch -J ${first_char}b2020x create_features_20x.sh 20x 128 brca_csv $i brca
    sbatch -J ${first_char}b2010x create_features_20x.sh 10x 64 brca_csv $i brca
    sbatch -J ${first_char}b205x create_features_20x.sh 5x 16 brca_csv $i brca
    sbatch -J ${first_char}b202.5x create_features_20x.sh 2.5x 4 brca_csv $i brca
done



sbatch -J cct40x20x create_features_40x.sh 20x 128 coad_read_csv ctranspath coad_read
sbatch -J cct40x10x create_features_40x.sh 10x 64 coad_read_csv ctranspath coad_read
sbatch -J cct40x5x create_features_40x.sh 5x 16 coad_read_csv ctranspath coad_read
sbatch -J cct40x2.5x create_features_40x.sh 2.5x 4 coad_read_csv ctranspath coad_read



sbatch -J cct20x20x create_features_20x.sh 20x 128 coad_read_csv ctranspath coad_read
sbatch -J cct20x10x create_features_10x.sh 10x 64 coad_read_csv ctranspath coad_read
sbatch -J cct20x5x create_features_20x.sh 5x 16 coad_read_csv optimus coad_read
sbatch -J cct20x2.5x create_features_20x.sh 2.5x 4 coad_read_csv optimus coad_read

for i in "resnet" "uni" "conch_v1" "lunit" "gigapath" "optimus" "virchow" "hibou"; do
    first_char=${i:0:1}  
    sbatch -J ${first_char}c4020x create_features_40x.sh 20x 128 coad_read_csv $i coad_read
    sbatch -J ${first_char}c4010x create_features_40x.sh 10x 64 coad_read_csv $i coad_read
    sbatch -J ${first_char}c405x create_features_40x.sh 5x 16 coad_read_csv $i coad_read
    sbatch -J ${first_char}c402.5x create_features_40x.sh 2.5x 4 coad_read_csv $i coad_read
    sbatch -J ${first_char}c2020x create_features_20x.sh 20x 128 coad_read_csv $i coad_read
    sbatch -J ${first_char}c2010x create_features_20x.sh 10x 64 coad_read_csv $i coad_read
    sbatch -J ${first_char}c205x create_features_20x.sh 5x 16 coad_read_csv $i coad_read
    sbatch -J ${first_char}c202.5x create_features_20x.sh 2.5x 4 coad_read_csv $i coad_read
done


for i in "resnet" "ctranspath" "uni" "conch_v1" "lunit" "gigapath" "optimus" "virchow" "hibou"; do
    for j in "mean_mil" "max_mil" "att_mil" "trans_mil" "dsmil" "wikgmil" "mamba_mil" "rrtmil"; do
        first_char=${i:0:2}
        second_char=${j:0:3}
        sbatch -J b${first_char}${second_char} train.sh tcga_2_class brca $i $j
        sbatch -J mb${first_char}${second_char} train_job_mag.sh tcga_2_class brca $i $j
    done
done

for i in "resnet" "ctranspath" "uni" "conch_v1" "lunit" "gigapath" "optimus" "virchow" "hibou"; do
    for j in "mean_mil" "max_mil" "att_mil" "trans_mil" "dsmil" "wikgmil" "mamba_mil" "rrtmil"; do
        first_char=${i:0:2}
        second_char=${j:0:3}
        sbatch -J c${first_char}${second_char} train.sh tcga_2_class coad_read $i $j
        sbatch -J mc${first_char}${second_char} train_job_mag.sh tcga_2_class coad_read $i $j
    done
done

for i in "resnet" "ctranspath" "uni" "conch_v1" "lunit" "gigapath" "optimus" "virchow" "hibou"; do
    for j in "clam_sb"; do
        first_char=${i:0:2}
        second_char=${j:0:3}
        sbatch -J b${first_char}${second_char} train.sh tcga_2_class brca $i $j
        sbatch -J mb${first_char}${second_char} train_job_mag.sh tcga_2_class brca $i $j
    done
done

for i in "resnet" "ctranspath" "uni" "conch_v1" "lunit" "gigapath" "optimus" "virchow" "hibou"; do
    for j in "clam_sb"; do
        first_char=${i:0:2}
        second_char=${j:0:3}
        sbatch -J c${first_char}${second_char} train.sh tcga_2_class coad_read $i $j
        sbatch -J mc${first_char}${second_char} train_job_mag.sh tcga_2_class coad_read $i $j
    done
done


for i in "resnet" "uni" "conch_v1" "lunit" "gigapath" "optimus" "virchow" "hibou"; do
    first_char=${i:0:1}
    sbatch -J ${first_char}20x create_features_ebrains.sh 20x 128 ipd_csv $i
    sbatch -J ${first_char}10x create_features_ebrains.sh 10x 64 ipd_csv $i
    sbatch -J ${first_char}5x create_features_ebrains.sh 5x 16 ipd_csv $i
    sbatch -J ${first_char}2.5x create_features_ebrains.sh 2.5x 4 ipd_csv $i
    sbatch -J ${first_char}20x_1 create_features_ebrains.sh 20x_1 128 ipd_csv $i
    sbatch -J ${first_char}10x_1 create_features_ebrains.sh 10x_1 64 ipd_csv $i
    sbatch -J ${first_char}5x_1 create_features_ebrains.sh 5x_1 16 ipd_csv $i
    sbatch -J ${first_char}2.5x_1 create_features_ebrains.sh 2.5x_1 4 ipd_csv $i
done

create_features_ebrains_orig.sh 20x 128 ipd_csv ctranspath

sbatch -J ct20x create_features_ebrains_orig.sh 20x 128 ipd_csv ctranspath
sbatch -J ct10x create_features_ebrains_orig.sh 10x 64 ipd_csv ctranspath
sbatch -J ct5x create_features_ebrains_orig.sh 5x 16 ipd_csv ctranspath
sbatch -J ct2.5x create_features_ebrains_orig.sh 2.5x 4 ipd_csv ctranspath
sbatch -J ct20x_1 create_features_ebrains_orig.sh 20x_1 128 ipd_csv ctranspath
sbatch -J ct10x_1 create_features_ebrains_orig.sh 10x_1 64 ipd_csv ctranspath
sbatch -J ct5x_1 create_features_ebrains_orig.sh 5x_1 16 ipd_csv ctranspath
sbatch -J ct2.5x_1 create_features_ebrains_orig.sh 2.5x_1 4 ipd_csv ctranspath
