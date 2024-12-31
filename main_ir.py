import os
import json
import warnings
import argparse
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils.util_func import get_peak_number, has_element_in_range
from utils.eval_func import Peak_for_draw
from utils.load_pth import load_freeze
from utils.compound_tools import get_atom_feature_dims, get_bond_feature_dims
from utils.draw_ir import draw_ir_spectra

from dataloaders.IRDataloader import GetAtomBondAngleDataloader

from models.GNN_IR import GNNIR_Model, train_GNNIR, eval_GNNIR, test_GNNIR, pred_real_GNNIR

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

# ----------------Commonly-used Parameters----------------
atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = ["bond_dir", "bond_type", "is_in_ring"]
bond_angle_float_names = ['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']
column_specify={
    'ADH':[1,5,0,0],'ODH':[1,5,0,1],'IC':[0,5,1,2],'IA':[0,5,1,3],'OJH':[1,5,0,4],
    'ASH':[1,5,0,5],'IC3':[0,3,1,6],'IE':[0,5,1,7],'ID':[0,5,1,8],'OD3':[1,3,0,9],
    'IB':[0,5,1,10],'AD':[1,10,0,11],'AD3':[1,3,0,12],'IF':[0,5,1,13],'OD':[1,10,0,14],
    'AS':[1,10,0,15],'OJ3':[1,3,0,16],'IG':[0,5,1,17],'AZ':[1,10,0,18],'IAH':[0,5,1,19],
    'OJ':[1,10,0,20],'ICH':[0,5,1,21],'OZ3':[1,3,0,22],'IF3':[0,3,1,23],'IAU':[0,1.6,1,24]
}
bond_float_names = []
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)

# ---------------- Hyper-Parameters ----------------
def parse_args():
    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--model_name', type=str, default='gnn_decoder', help="model name")
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_root', type=str, default="/remote-home1/lihao/ai4s/ECDFormer/", help='dataset root')
    parser.add_argument('--rand_seed', type=int, default=1101)
    parser.add_argument('--sample_path', type=str, default='./dataset/ECD/')
    parser.add_argument('--mode', type=str, default='Train')
    parser.add_argument('--loss_type', type=str, default='MSE')
    
    # argument for visualization
    parser.add_argument('--visual_epoch', type=int, default=0, help="the model epoch for visualization")
    parser.add_argument('--NotPrintSave', action='store_true', help="print and save")
    args = parser.parse_args()
    
    args.MODE = args.mode
    assert args.MODE in ['Train', 'Test', 'Visual', 'Real', "multi_carbon"]
    args.test_mode = 'fixed'            # fixed or random or enantiomer(extract enantimoers)
    args.transfer_target = 'All_column' # trail name
    args.Use_geometry_enhanced = True   # default: True
    args.Use_column_info = False         # default: True
    
    if args.Use_geometry_enhanced:
        # bond_float_names = ["bond_length", 'prop']  # from HPLC
        bond_float_names=['bond_length']
    else:
        bond_float_names=['bond_length']

    if args.Use_column_info==True:
        bond_id_names.extend(['coated', 'immobilized'])
        bond_float_names.extend(['diameter'])
        if args.Use_geometry_enhanced==True:
            bond_angle_float_names.extend(['column_TPSA', 'column_TPSA', 'column_TPSA', 'column_MDEC', 'column_MATS'])
        else:
            bond_float_names.extend(['column_TPSA', 'column_TPSA', 'column_TPSA', 'column_MDEC', 'column_MATS'])
        full_bond_feature_dims.extend([2,2])

    args.train_ratio, args.valid_ratio, args.test_ratio = 0.90, 0.05, 0.05
    
    # add the global information into the args
    args.bond_float_names = bond_float_names
    args.atom_id_names = atom_id_names
    args.bond_id_names = bond_id_names
    args.bond_angle_float_names = bond_angle_float_names
    args.column_specify = column_specify
    args.full_atom_feature_dims = full_atom_feature_dims
    args.full_bond_feature_dims = full_bond_feature_dims

    # 兼容 ddp
    if not args.__contains__("local_rank"):
        args.local_rank = 0
    print(args, flush=True)
    return args

if __name__ == "__main__":
    args = parse_args()
    
    #-------------Construct data----------------
    IR_dataset = np.load(args.dataset_root + 'dataset/IR/ir_column_charity_10000.npy',allow_pickle=True).tolist()
    dataset_all, smiles_all, index_all = IR_dataset['dataset_all'], IR_dataset['smiles_all'], IR_dataset['index_all'], 

    atom_bond_dataloaders, bond_angle_dataloaders, smiles_datasets = GetAtomBondAngleDataloader(args, dataset_all, smiles_all, index_all)
    
    (train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond) = atom_bond_dataloaders
    (train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle) = bond_angle_dataloaders
    (train_smiles, valid_smiles, test_smiles) = smiles_datasets

    nn_params = {
        'args': args, 
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
    }
    model_device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    if args.model_name == 'gnn_ir':
        model = GNNIR_Model(**nn_params).to(model_device)
        train_model, eval_model, test_model = train_GNNIR, eval_GNNIR, test_GNNIR
        pred_model = pred_real_GNNIR
    else: assert("model name not find")
    
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=200, gamma=0.25)

    if args.MODE == 'Train':
        print('================== Start training =================\n')
        if not args.NotPrintSave: # 需要保留中间结果
            now = datetime.now()
            file_name = f'ckpts/ir_model_{args.model_name}/{now.month}-{now.day}-{now.hour}-{now.minute}/'
            # file_name = f'saves_lh/ecd/model_{args.model_name}/only_test/'
            try: os.makedirs(file_name)
            except OSError: pass
            train_log_txt, test_log_txt = file_name+'train_log.txt', file_name+'test_log.txt'
            config_log_txt = file_name+'config_log.txt'

            with open(config_log_txt, 'w') as f: # save the model args
                json.dump(args.__dict__, f, indent=2)

        for epoch in tqdm(range(args.epochs)):
            train_mae = train_model(args, model, model_device, train_loader_atom_bond, train_loader_bond_angle, optimizer)
            lr_scheduler.step()
            if (epoch + 1) % 40 == 0:
                print("the %d epoch's learning rate = %f" % (epoch, optimizer.param_groups[0]['lr']))
                valid_mae = eval_model(args, model, model_device, valid_loader_atom_bond, valid_loader_bond_angle)
                print("epoch: {}, train = {}, valid = {}\n".format((epoch + 1), train_mae, valid_mae))
                if not args.NotPrintSave:
                    with open(train_log_txt, "a") as f:
                        f.write("epoch: {}, train = {}, valid = {}\n".format((epoch + 1), train_mae, valid_mae))
                    torch.save(model.state_dict(), file_name+f'/model_save_{epoch + 1}.pth')
                
                gt, pred, result_info = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, epoch)
        
        # print('================== Start testing ==============\n')
        # for epoch in range(int(args.epochs/100)-1):
        #     model.load_state_dict(
        #         torch.load(
        #             file_name+'/model_save_{}.pth'.format(str(epoch+1)+'00'),
        #             map_location=model_device,)
        #     )
        #     gt, pred, result_info = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, epoch)

    if args.MODE == 'Test':
        print('================== Only Start testing ==============\n')
        # now = datetime.now()
        # save_path = f'saves_lh/ecd/model_{args.model_name}/10-16-16-14/'
        save_path = f'ckpts/ir_model_{args.model_name}/best/'
        # log_txt = save_path + 'test_log.txt'
        # json_path = save_path + 'test_log.json'
        # pred_dict = {}
        for epoch in range(1, int(args.epochs/40)):
            model.load_state_dict(
                torch.load(
                    save_path+'model_save_{}.pth'.format(int(epoch)*40),
                    map_location=model_device,)
            )
            # valid_mae = eval_model(args, model, model_device, valid_loader_atom_bond, valid_loader_bond_angle)
            gt, pred, results = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, epoch)
    
    if args.MODE == 'Visual':
        # 这个函数包含: (1). 可视化IR光谱预测 (2). 统计不同官能团是否和IR预测的峰出现的一致
        print('================== Visualizing IR ==================\n')
        save_path = f'saves_lh/ir/model_{args.model_name}/9-1-19-38/'
        visual_path = f'fig_lh/ir/{args.model_name}'
        analyse_dir_path = visual_path + f'/analyse_visualization/' # 存放对结果的统计图片
        case_dir_path = visual_path + f'/case_visualization/'       # 存放prediction和gt的对比图
        try: 
            os.makedirs(analyse_dir_path)
            os.makedirs(case_dir_path)
        except OSError: pass
        
        model.load_state_dict(torch.load(save_path+'/model_save_{}.pth'.format(str(args.visual_epoch))))
        gt, pred, results = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, args.visual_epoch)
        
        # 待检查的官能团集合
        from rdkit import Chem
        from rdkit.Chem import AllChem
        func_groups = {
            "C=O": Chem.MolFromSmarts("C=O"),       # 羰基
            "C=C": Chem.MolFromSmarts("C=C"),       # CC双键 烯烃, 芳烃
            "[OX2H]": Chem.MolFromSmarts("[OX2H]"), # 羟基
            "[NX3;!$(NC=O)]": Chem.MolFromSmarts("[NX3;!$(NC=O)]"), # 一级，二级，三级胺
            # '[NX3;H2]': Chem.MolFromSmarts('[NX3;H2]'),  # 一级胺 (-NH2)
            # '[NX3;H1]': Chem.MolFromSmarts('[NX3;H1]'),  # 二级胺 (-NH)
            # '[NX3;H0]': Chem.MolFromSmarts('[NX3;H0]'),  # 三级胺 (-N)
            "[F,Cl,Br,I]": Chem.MolFromSmarts("[F,Cl,Br,I]"), # 卤素化合物
        }
        func_groups_pos = {
            "C=O": [1650, 1850], "C=C": [1600, 1680], "[OX2H]": [2500, 3300], "[NX3;!$(NC=O)]": [3000, 3500],
            # '[NX3;H2]': [3300, 3500], '[NX3;H1]': [3100, 3400], '[NX3;H0]': [3000, 3300], 
            "[F,Cl,Br,I]": [600, 800]
        }
        func_group_info = {
            "C=O": [0, 0], "C=C": [0, 0], "[OX2H]": [0, 0], "[NX3;!$(NC=O)]":[0, 0],
            # '[NX3;H2]': [0, 0],'[NX3;H1]': [0, 0],'[NX3;H0]': [0, 0]
            "[F,Cl,Br,I]": [0, 0],
        }
        # func_group_info[fg][0]:具有基团的分子数量, func_group_info[fg][1]: 峰也对的上的分子数量
        func_group_distribution = {
            "C=O": 0, "C=C": 0, "[OX2H]": 0, "[NX3;!$(NC=O)]": 0, "[F,Cl,Br,I]": 0
            # '[NX3;H2]': 0,'[NX3;H1]': 0,'[NX3;H0]': 0
        } # 数据集中官能团的数量占比
        
        save_info = dict()

        for batch_id in range(len(gt)):
            batch_gt, batch_pred = gt[batch_id], pred[batch_id]
            assert len(batch_gt['pos']) == len(batch_pred['pos'])
            for case_id in tqdm(range(len(batch_gt['pos']))):
                case_gt = dict(pos=batch_gt['pos'][case_id], height=batch_gt['height'][case_id])
                case_pred = dict(pos=batch_pred['pos'][case_id], height=batch_pred['height'][case_id])
                
                total_id = batch_id*len(gt[0]['pos']) + case_id # 计算case对应的smiles_id
                mol_smiles = test_smiles[total_id]
                mol = AllChem.MolFromSmiles(mol_smiles)

                if abs(len(batch_gt['pos'][case_id])-len(batch_pred['pos'][case_id])) <= 2:
                    save_info[mol_smiles] = {'pred': case_pred, 'gt': case_gt}
                    continue
                    ## 绘制IR spectra的预测图片
                    draw_ir_spectra(
                        case_gt=case_gt, case_pred=case_pred, 
                        save_path=case_dir_path,
                        fig_name=f"{batch_id}_{case_id}"
                    )
                    with open(case_dir_path+f"{batch_id}_{case_id}.txt", "w") as f:
                        f.write(mol_smiles)

                    ## 统计预测的光谱有多少结构和分子对的上
                    seq_peak = batch_pred['pos'][case_id]
                    real_peak = [500+pos*100 for pos in seq_peak]
                    for fg, fg_smarts in func_groups.items():
                        # if mol.HasSubstructMatch(fg_smarts):
                        #     func_group_info[fg][0] += 1
                        #     find = has_element_in_range(
                        #         lst=real_peak, 
                        #         lower_bound=func_groups_pos[fg][0],
                        #         upper_bound=func_groups_pos[fg][1],
                        #     )
                        #     if find: func_group_info[fg][1] += 1

                        find = has_element_in_range(real_peak, func_groups_pos[fg][0], func_groups_pos[fg][1])
                        if find: 
                            func_group_info[fg][0] += 1
                            if mol.HasSubstructMatch(fg_smarts):
                                func_group_info[fg][1] += 1

        json.dump(save_info, open(visual_path+"/ir_good_case.json", "w"))
        
        for fg, info in func_group_info.items():
            print(f"Func_Group={fg}, Acc={func_group_info[fg][1]/func_group_info[fg][0]}") 

        for smiles in test_smiles:
            mol = AllChem.MolFromSmiles(smiles)
            for fg, fg_smarts in func_groups.items():
                if mol.HasSubstructMatch(fg_smarts):
                    func_group_distribution[fg] += 1
        for fg, num in func_group_distribution.items():
            print(f"Func_Group={fg}, distribution={num/len(test_smiles)}")