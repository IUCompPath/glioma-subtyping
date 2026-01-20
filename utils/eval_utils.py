import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Internal imports (ensure these paths exist in your project)
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from models.model_clam import CLAM_SB, CLAM_MB  # Required for Script 2 models
from models.model_mil import MIL_fc, MIL_fc_mc # Required for Script 2 models

def initiate_model(args, ckpt_path, device='cuda'):
    print(f'Init Model: {args.model_type}')    
    
    # Base model dictionary
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    # Handle CLAM/MIL-fc logic from Script 2
    if args.model_type in ['clam_sb', 'clam_mb', 'mil']:
        if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
            model_dict.update({"size_arg": args.model_size})
        
        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict)
        else: # args.model_type == 'mil'
            if args.n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)
    
    # Handle MIL variants from Script 1
    elif args.model_type == 'mean_mil':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'max_mil':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'att_mil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim=args.in_dim, n_classes=args.n_classes, act='gelu', dropout=args.drop_out)
    elif args.model_type == 'dsmil':
        from models.DSMIL import FCLayer, BClassifier, MILNet
        i_classifier = FCLayer(in_size=args.in_dim, out_size=args.n_classes)
        b_classifier = BClassifier(input_size=args.in_dim, output_class=args.n_classes, dropout_v=args.drop_out)
        model = MILNet(i_classifier, b_classifier) 
    elif args.model_type in ['wikgmil', 'wikgmil_1']:
        from models.WiKGMIL import WiKG
        dim_hidden = 512 if args.model_type == 'wikgmil_1' else 128
        model = WiKG(dim_in=args.in_dim, dim_hidden=dim_hidden, topk=2, n_classes=args.n_classes, 
                     agg_type='bi-interaction', dropout=args.drop_out, pool='attn')
    elif args.model_type == 'rrtmil':
        from modules.rrt import RRTMIL
        model_params = {
            'input_dim': args.in_dim, 'n_classes': args.n_classes, 'dropout': args.drop_out,
            'act': 'gelu', 'region_num': 32, 'pool': 'attn', 'peg_k': 7, 'n_layers': 2,
            'n_heads': 8, 'attn': 'rmsa', 'trans_dim': 64, 'ffn': True, 'mlp_ratio': 4.
        }
        model = RRTMIL(**model_params)
    elif args.model_type == 'mamba_mil':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim=args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, 
                         act='gelu', layer=args.mambamil_layer, rate=args.mambamil_rate, type=args.mambamil_type)
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')

    # Load Checkpoint
    print_network(model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''): ckpt[key]})
    
    model.load_state_dict(ckpt_clean, strict=True)
    model = model.to(device)
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print(f'Test Error: {test_error:.4f}')
    print(f'AUC: {auc:.4f}')
    return model, patient_results, test_error, auc, df

def eval_survival(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print('Init Survival Loaders')
    loader = get_simple_loader_survival(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print(f'Survival Test Error: {test_error:.4f}')
    print(f'Survival AUC: {auc:.4f}')
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(tqdm(loader)):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        
        with torch.no_grad():
            # Standard MIL/CLAM output signature: logits, Y_prob, Y_hat, A_raw, results_dict
            logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        test_error += calculate_error(Y_hat, label)

    test_error /= len(loader)

    # AUC Calculation
    if len(np.unique(all_labels)) <= 1:
        auc_score = -1
    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            
            if hasattr(args, 'micro_average') and args.micro_average:
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({f'p_{c}': all_probs[:, c]})
    
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger