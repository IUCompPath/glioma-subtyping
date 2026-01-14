
import pandas as pd
import numpy as np
import shutil
import os
from tqdm import tqdm
import sklearn
from glob import glob
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

task ='histology'

dir_path = '/home/shubham/ssl/CLAM_lunit/eval_results_tcga/'+task
cohort = ['train','val','test']
ensemble_list = ['5x_10x_20x','10x_20x','5x_10x','5x_20x','2.5x_20x','2.5x_5x_10x','2.5x_5x','2.5x_10x','2.5x_5x_20x','2.5x_10x_20x','2.5x_5x_10x_20x']
len(ensemble_list)

for q in tqdm(ensemble_list):
    mags_to_consider = q.split('_')
    results_dir  = dir_path+'/'+q
    try:
        os.mkdir(results_dir)
    except:
        pass
    for m in cohort:
        try:
            os.mkdir(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_'+m)
        except:
            continue
        
        for l in range(0,10):
            dfs = []
            for mag in mags_to_consider:
                file_path = dir_path+'/'+mag+'/EVAL_tcga_2021_'+task+'_'+(mag)+'_'+m+'/fold_'+str(l)+'.csv'
                df_1 = pd.read_csv(file_path)
                dfs.append(df_1)
            
            result_df = pd.concat(dfs, ignore_index=True)
            df = result_df.groupby('slide_id', as_index=False).mean()

            for i in tqdm(df['slide_id'].to_list()):
                # Use argmax to get the index of the maximum probability among p_0, p_1, p_2
                predicted_class = df.loc[df['slide_id'] == i, ['p_0', 'p_1', 'p_2']].values.argmax(axis=1)[0]
                
                # Assign the predicted class to the 'Y_hat' column
                df.loc[df['slide_id'] == i, 'Y_hat'] = predicted_class
            df.to_csv(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_'+m+'/fold_'+str(l)+'.csv',index=False)




ensemble_list = ['2.5x','5x','10x','20x','5x_10x_20x','10x_20x','5x_10x','5x_20x','2.5x_20x','2.5x_5x_10x','2.5x_5x','2.5x_10x','2.5x_5x_20x','2.5x_10x_20x','2.5x_5x_10x_20x']

for q in tqdm(ensemble_list):
    mags_to_consider = q.split('_')
    results_dir  = dir_path+'/'+q
    df_summary = pd.DataFrame(columns = ['model','train_auc','train_acc','balanced_train_acc','val_auc','val_acc','balanced_val_accuracy',\
            'test_auc','test_acc','balanced_test_accuracy','train_sen', 'train_spec', 'val_sen', 'val_spec','test_sen', 'test_spec'])
    for l in range(0,10):
        df_train = pd.read_csv(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_train/fold_'+str(l)+'.csv')
        df_val = pd.read_csv(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_val/fold_'+str(l)+'.csv')
        df_test = pd.read_csv(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_test/fold_'+str(l)+'.csv')

        train_acc = accuracy_score(df_train['Y'].to_list(),df_train['Y_hat'].to_list())
        val_acc =accuracy_score(df_val['Y'].to_list(),df_val['Y_hat'].to_list())
        test_acc = accuracy_score(df_test['Y'].to_list(),df_test['Y_hat'].to_list())

        train_acc_b =balanced_accuracy_score(df_train['Y'].to_list(),df_train['Y_hat'].to_list())
        val_acc_b = balanced_accuracy_score(df_val['Y'].to_list(),df_val['Y_hat'].to_list())
        test_acc_b =balanced_accuracy_score(df_test['Y'].to_list(),df_test['Y_hat'].to_list())

        all_labels_train = df_train['Y'].to_list()
        binary_labels = label_binarize(all_labels_train, classes=[i for i in range(3)])
        aucs = []
        for class_idx in range(3):
            if class_idx in all_labels_train:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], df_train['p_'+str(class_idx)].to_list())
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        train_auc_score = np.nanmean(np.array(aucs))

        all_labels_val = df_val['Y'].to_list()
        binary_labels_val = label_binarize(all_labels_val, classes=[i for i in range(3)])
        aucs = []
        for class_idx in range(3):
            if class_idx in all_labels_val:
                fpr, tpr, _ = roc_curve(binary_labels_val[:, class_idx], df_val['p_'+str(class_idx)].to_list())
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        val_auc_score = np.nanmean(np.array(aucs))

        aucs = []
        all_labels_test = df_test['Y'].to_list()
        binary_labels_test = label_binarize(all_labels_test, classes=[i for i in range(3)])
    
        for class_idx in range(3):
            if class_idx in all_labels_test:
                fpr, tpr, _ = roc_curve(binary_labels_test[:, class_idx], df_test['p_'+str(class_idx)].to_list())
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        test_auc_score = np.nanmean(np.array(aucs))
        

        cm = confusion_matrix(df_train['Y'].to_list(),df_train['Y_hat'].to_list())
        num_classes = len(cm)
        train_sen = []
        train_spec = []

        for class_idx in range(3):
            true_positive = cm[class_idx, class_idx]
            false_negative = sum(cm[class_idx, :]) - true_positive
            false_positive = sum(cm[:, class_idx]) - true_positive
            true_negative = sum(sum(cm)) - (true_positive + false_negative + false_positive)

            sensitivity_class = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specificity_class = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

            train_sen.append(sensitivity_class)
            train_spec.append(specificity_class)


        cm = confusion_matrix(df_val['Y'].to_list(),df_val['Y_hat'].to_list())
        num_classes = len(cm)
        val_sen = []
        val_spec = []

        for class_idx in range(3):
            true_positive = cm[class_idx, class_idx]
            false_negative = sum(cm[class_idx, :]) - true_positive
            false_positive = sum(cm[:, class_idx]) - true_positive
            true_negative = sum(sum(cm)) - (true_positive + false_negative + false_positive)

            sensitivity_class = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specificity_class = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

            val_sen.append(sensitivity_class)
            val_spec.append(specificity_class)

        cm = confusion_matrix(df_test['Y'].to_list(),df_test['Y_hat'].to_list())
        num_classes = len(cm)
        test_sen = []
        test_spec = []

        for class_idx in range(3):
            true_positive = cm[class_idx, class_idx]
            false_negative = sum(cm[class_idx, :]) - true_positive
            false_positive = sum(cm[:, class_idx]) - true_positive
            true_negative = sum(sum(cm)) - (true_positive + false_negative + false_positive)

            sensitivity_class = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specificity_class = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

            test_sen.append(sensitivity_class)
            test_spec.append(specificity_class)

        df_summary = df_summary.append({'model':l,'train_auc':train_auc_score,'train_acc':train_acc,'balanced_train_acc':train_acc_b,
            'val_acc':val_acc,'val_auc':val_auc_score,'balanced_val_accuracy':val_acc_b, 'test_acc':test_acc, 'test_auc':test_auc_score,'balanced_test_accuracy':test_acc_b,
                'train_sen':train_sen, 'train_spec':train_spec, 'val_sen':val_sen, 'val_spec':val_spec,'test_sen':test_sen, 'test_spec':test_spec},ignore_index=True)
    df_summary.to_csv(results_dir+'/EVAL_tcga_idh_'+q+'_eval_results_detailed.csv',index=False)






"""Test data magnification wise 10-fold Detailed Evaluation"""

task = 'histology'

dir_path = '/home/shubham/ssl/CLAM_lunit/eval_results_ebrains'
m = 'test'
ensemble_list = ['5x_10x_20x','10x_20x','5x_10x','5x_20x','2.5x_20x','2.5x_5x_10x','2.5x_5x','2.5x_10x','2.5x_5x_20x','2.5x_10x_20x','2.5x_5x_10x_20x']
len(ensemble_list)


for q in tqdm(ensemble_list):
    mags_to_consider = q.split('_')
    results_dir  = dir_path+'/'+task
    
    try:
        os.mkdir(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_'+m)
    except:
        continue
    
    for l in range(0,10):
        dfs = []
        for mag in mags_to_consider:
            file_path = dir_path+'/'+task+'/'+'EVAL_tcga_2021_'+task+'_'+(mag)+'_'+m+'/fold_'+str(l)+'.csv'
            df_1 = pd.read_csv(file_path)
            dfs.append(df_1)
        
        result_df = pd.concat(dfs, ignore_index=True)
        df = result_df.groupby('slide_id', as_index=False).mean()

        for i in tqdm(df['slide_id'].to_list()):
            # Use argmax to get the index of the maximum probability among p_0, p_1, p_2
            predicted_class = df.loc[df['slide_id'] == i, ['p_0', 'p_1', 'p_2']].values.argmax(axis=1)[0]
            
            # Assign the predicted class to the 'Y_hat' column
            df.loc[df['slide_id'] == i, 'Y_hat'] = predicted_class
        df.to_csv(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_'+m+'/fold_'+str(l)+'.csv',index=False)



ensemble_list = ['2.5x','5x','10x','20x','5x_10x_20x','10x_20x','5x_10x','5x_20x','2.5x_20x','2.5x_5x_10x','2.5x_5x','2.5x_10x','2.5x_5x_20x','2.5x_10x_20x','2.5x_5x_10x_20x']
task = 'histology'
m='test'
aucs = []


for q in tqdm(ensemble_list):
    mags_to_consider = q.split('_')
    results_dir  = dir_path+'/'+task
    df_summary = pd.DataFrame(columns = ['fold','test_auc','test_acc','balanced_test_accuracy','test_sen', 'test_spec'])
    for l in range(0,10):
        df_test = pd.read_csv(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_test/fold_'+str(l)+'.csv')
        test_acc = accuracy_score(df_test['Y'].to_list(),df_test['Y_hat'].to_list())
        test_acc_b =balanced_accuracy_score(df_test['Y'].to_list(),df_test['Y_hat'].to_list())


        aucs = []
        all_labels_test = df_test['Y'].to_list()
        binary_labels_test = label_binarize(all_labels_test, classes=[i for i in range(3)])
    
        for class_idx in range(3):
            if class_idx in all_labels_test:
                fpr, tpr, _ = roc_curve(binary_labels_test[:, class_idx], df_test['p_'+str(class_idx)].to_list())
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        test_auc_score = np.nanmean(np.array(aucs))
        

        cm = confusion_matrix(df_test['Y'].to_list(),df_test['Y_hat'].to_list())
        num_classes = len(cm)
        test_sen = []
        test_spec = []

        for class_idx in range(3):
            true_positive = cm[class_idx, class_idx]
            false_negative = sum(cm[class_idx, :]) - true_positive
            false_positive = sum(cm[:, class_idx]) - true_positive
            true_negative = sum(sum(cm)) - (true_positive + false_negative + false_positive)

            sensitivity_class = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specificity_class = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

            test_sen.append(sensitivity_class)
            test_spec.append(specificity_class)

        df_summary = df_summary.append({'fold':l,'test_auc':test_auc_score,'test_acc':test_acc,'balanced_test_accuracy':test_acc_b,
                'test_sen':test_sen, 'test_spec':test_spec},ignore_index=True)
    df_summary.to_csv(results_dir+'/EVAL_tcga_2021_'+task+'_'+q+'_test'+'/EVAL_tcga_idh_'+q+'_eval_results_detailed.csv',index=False)





df_summary = pd.DataFrame(columns = ['fold','test_auc','test_acc','balanced_test_accuracy','test_sen', 'test_spec'])
for l in range(0,10):
    df_test = pd.read_csv('/home/shubham/clam_idh/eval_results/EVAL_renal_test_new_data/fold_'+str(l)+'.csv')
    test_acc = accuracy_score(df_test['Y'].to_list(),df_test['Y_hat'].to_list())
    test_acc_b =balanced_accuracy_score(df_test['Y'].to_list(),df_test['Y_hat'].to_list())


    aucs = []
    all_labels_test = df_test['Y'].to_list()
    binary_labels_test = label_binarize(all_labels_test, classes=[i for i in range(3)])

    for class_idx in range(3):
        if class_idx in all_labels_test:
            fpr, tpr, _ = roc_curve(binary_labels_test[:, class_idx], df_test['p_'+str(class_idx)].to_list())
            aucs.append(auc(fpr, tpr))
        else:
            aucs.append(float('nan'))
    test_auc_score = np.nanmean(np.array(aucs))
    

    cm = confusion_matrix(df_test['Y'].to_list(),df_test['Y_hat'].to_list())
    num_classes = len(cm)
    test_sen = []
    test_spec = []

    for class_idx in range(3):
        true_positive = cm[class_idx, class_idx]
        false_negative = sum(cm[class_idx, :]) - true_positive
        false_positive = sum(cm[:, class_idx]) - true_positive
        true_negative = sum(sum(cm)) - (true_positive + false_negative + false_positive)

        sensitivity_class = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity_class = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

        test_sen.append(sensitivity_class)
        test_spec.append(specificity_class)

    df_summary = df_summary.append({'fold':l,'test_auc':test_auc_score,'all_auc' :aucs,'test_acc':test_acc,'balanced_test_accuracy':test_acc_b,
            'test_sen':test_sen, 'test_spec':test_spec},ignore_index=True)
df_summary.to_csv('/home/shubham/clam_idh/eval_results/EVAL_renal_test_new_data/eval_results_detailed.csv',index=False)

cases_0 = os.listdir('/home/shubham/ffpe_ml/split_simple/splits_8/test/0')
cases_1 = os.listdir('/home/shubham/ffpe_ml/split_simple/splits_8/test/1')

for i in cases_1:
    cases_0.append(i)
cases = cases_0.append(cases_1)
df = pd.DataFrame(cases_0)
df['label']=0
df.to_csv('/home/shubham/ffpe_ml/split_simple/df_test.csv',index=False)
