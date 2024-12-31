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

from utils.util_func import get_peak_number
from utils.eval_func import Peak_for_draw
from utils.draw_func import draw_scatter_diagram, draw_number_scatter, draw_prediction_and_gt, draw_bin_graph, draw_position_scatter
from utils.load_pth import load_freeze
from utils.compound_tools import get_atom_feature_dims, get_bond_feature_dims
# from dataloaders.PosDataloader import GetAtomBondAngleDataloader
from dataloaders.FinalDataloader import GetAtomBondAngleDataloader
from dataloaders.DrugDataloader import GetDrugDataloader
from dataloaders.MultiCarbonDataloader import GetMultiCarbonDataloader

# from models.GNN_PeakCLS import GNNPeakCLS_Model, train_GNNPeakCLS, eval_GNNPeakCLS, test_GNNPeakCLS
from models.GNN_AllThree import GNNAllThree_Model, train_GNNAllThree, eval_GNNAllThree, test_GNNAllThree, pred_real_GNNAllThree

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
    
    # model parameter
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
    parser.add_argument('--activation', type=str, default='relu',
                        help="activation function type, for ablation study")
    
    # dataset parameter
    parser.add_argument('--dataset_root', type=str, default="/remote-home1/lihao/ai4s/ECDFormer/", help='dataset root')
    parser.add_argument('--rand_seed', type=int, default=1101)
    parser.add_argument('--sample_path', type=str, default='/remote-home1/lihao/ai4s/ECDFormer/dataset/ECD/')

    # training parameter
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop (default: 10)')
    parser.add_argument('--mode', type=str, default='Train')
    parser.add_argument('--loss_type', type=str, default='MSE')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers (default: 0)')
    
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
    
    if args.MODE in ['Train', 'Test', 'Visual']:
        #-------------Construct data----------------
        ECD_all = pd.read_csv(args.dataset_root+'dataset/ECD/ecd_info.csv', encoding='gbk')
        ECD_smile_all = ECD_all['SMILES'].values
        index_all = ECD_all['Unnamed: 0'].values
        T1_all = ECD_all['RT'].values
        Speed_all = ECD_all['Speed'].values
        Prop_all = ECD_all['i-PrOH_proportion'].values
        Column_info = ECD_all['Column'].values
        ECD_dataset = np.load(args.dataset_root+'dataset/ECD/ecd_column_charity_new_smiles.npy', allow_pickle=True).tolist()
        dataset_all, smiles_all = [], []
        for itm in ECD_dataset:
            dataset_all.append(itm['info'])
            smiles_all.append(itm['smiles'])
        
        # get dict to match the (linenumber - unnamed_idx - hand_idx) :
        unnamed_idx_dict, hand_idx_dict, line_idx_dict = {}, {}, {}
        for i in range(len(ECD_dataset)):
            # line_number, unnamed index(itm['id']), hand_id
            itm = ECD_dataset[i]
            line_idx_dict[i] = {'hand_id':itm['hand_id'], 'unnamed_id':itm['id'], 'smiles':itm['smiles']}
            if itm['id'] not in unnamed_idx_dict.keys():
                unnamed_idx_dict[itm['id']] = {'line_number': i, 'hand_id':itm['hand_id'], 'smiles':itm['smiles']}
            else: assert "unnamed id repeat error"
            if itm['hand_id'] not in hand_idx_dict.keys():
                hand_idx_dict[itm['hand_id']] = [{'line_number': i, 'unnamed_id': itm['id'], 'smiles':itm['smiles']}]
            else:
                hand_idx_dict[itm['hand_id']].append(
                    {'line_number': i, 'unnamed_id': itm['id'], 'smiles':itm['smiles']}
                ) 

        using_data_percent = 1.0 # ablation study, using small percent for training 
        atom_bond_dataloaders, bond_angle_dataloaders, smiles_datasets = GetAtomBondAngleDataloader(
            args, dataset_all, smiles_all, index_all, T1_all, Speed_all, Prop_all, args.transfer_target, Column_info,
            unnamed_idx_dict, hand_idx_dict, line_idx_dict, using_data_percent
        )
        
        (train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond) = atom_bond_dataloaders
        (train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle) = bond_angle_dataloaders
        (train_smiles, valid_smiles, test_smiles) = smiles_datasets

    nn_params = {
        'args': args, 
        'num_tasks': 9,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    model_device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    if args.model_name == 'gnn_posenc':
        model = GNNPosEncoder_Model(**nn_params).to(model_device)
        train_model, eval_model, test_model = train_GNNPosEncoder, eval_GNNPosEncoder, test_GNNPosEncoder
    elif args.model_name == 'gnn_posheight':
        model = GNNPosAndHeight_Model(**nn_params).to(model_device)
        train_model, eval_model, test_model = train_GNNPosAndHeight, eval_GNNPosAndHeight, test_GNNPosAndHeight
    elif args.model_name == 'gnn_allthree':
        model = GNNAllThree_Model(**nn_params).to(model_device)
        train_model, eval_model, test_model = train_GNNAllThree, eval_GNNAllThree, test_GNNAllThree
        pred_model = pred_real_GNNAllThree
    else: assert("model name not find")
    
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=200, gamma=0.25)
    
    if args.MODE == 'Train':
        print('=========Start training!=================\n')
        if not args.NotPrintSave: # keep the middle results
            now = datetime.now()
            file_name = f'saves/ecd/model_{args.model_name}/{now.month}-{now.day}-{now.hour}-{now.minute}/'
            try: os.makedirs(file_name)
            except OSError: pass
            train_log_txt, test_log_txt = file_name+'train_log.txt', file_name+'test_log.txt'
            config_log_txt = file_name+'config_log.txt'

            with open(config_log_txt, 'w') as f: # save the model args
                json.dump(args.__dict__, f, indent=2)

        for epoch in tqdm(range(args.epochs)):
            train_mae = train_model(args, model, model_device, train_loader_atom_bond, train_loader_bond_angle, optimizer)
            lr_scheduler.step()
            if (epoch + 1) % 20 == 0:
                print("the %d epoch's learning rate = %f" % (epoch, optimizer.param_groups[0]['lr']))
                valid_mae = eval_model(args, model, model_device, valid_loader_atom_bond, valid_loader_bond_angle)
                print("epoch: {}, train = {}, valid = {}\n".format((epoch + 1), train_mae, valid_mae))
                if not args.NotPrintSave:
                    with open(train_log_txt, "a") as f:
                        f.write("epoch: {}, train = {}, valid = {}\n".format((epoch + 1), train_mae, valid_mae))
            if (epoch + 1) % 100 == 0:
                if not args.NotPrintSave:
                    torch.save(model.state_dict(), file_name+f'/model_save_{epoch + 1}.pth')
                gt, pred, result_info = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, epoch)
        
        print('==================Start testing==============\n')
        for epoch in range(int(args.epochs/100)-1):
            model.load_state_dict(
                torch.load(
                    file_name+'/model_save_{}.pth'.format(str(epoch+1)+'00'),
                    map_location=model_device,)
            )
            gt, pred, result_info = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, epoch)

    if args.MODE == 'Test':
        print('==================Only Start testing==============\n')
        # now = datetime.now()
        # save_path = f'saves/ecd/model_{args.model_name}/10-16-16-14/'
        save_path = f'saves/ecd/model_{args.model_name}/best-2rang/'
        # log_txt = save_path + 'test_log.txt'
        # json_path = save_path + 'test_log.json'
        # pred_dict = {}
        for epoch in range(int(args.epochs/100) - 1):
            model.load_state_dict(
                torch.load(
                    save_path+'model_save_{}.pth'.format(str(epoch+1)+'00'),
                    map_location=model_device,)
            )
            gt, pred, results = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, epoch)
    
    if args.MODE == 'Real':
        ## for Real Drug ECD Prediction
        print('================== Start real drug prediction ==============\n')
        print("Loading Real Drug Dataset")
        
        start_time = time.time()
        drug_path = args.dataset_root + "dataset/ECD/test_multi_chiral_carbon.csv"
        drug_file_name = "drug_atom_info_multi.json"
        
        # drug_path = args.dataset_root + "dataset/ECD/test_single_chiral_carbon.csv"
        # drug_file_name = "drug_atom_info_single.json"
        
        test_loader_atom_bond, test_loader_bond_angle, test_smiles = GetDrugDataloader(args, drug_path)
        print(len(test_smiles))
        save_path = f'ckpts/model_{args.model_name}/best-2rang/'
        visual_path = f'figs/ecd/{args.model_name}/'
        model.load_state_dict(torch.load(save_path+'model_save_{}.pth'.format(str(args.visual_epoch))))

        pred_pos, pred_height = pred_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, args.visual_epoch)
        assert len(test_smiles)==len(pred_pos)
        end_time = time.time()
        
        drug_atom_info = []
        for id in range(len(test_smiles)):
            drug_atom_info.append({
                  'id':id+1, 'smiles':test_smiles[id], 
                  'pred':{'pos':pred_pos[id], 'height':pred_height[id]},
                })
        json.dump(drug_atom_info, open(visual_path+drug_file_name, "w"))
        print("successfully save the drug prediction in: ", visual_path+drug_file_name)
        print("time cost: {}.s, CPU time: {}".format(end_time-start_time, 128*(end_time-start_time)))
    
    if args.MODE == "multi_carbon":
        ## for evaluating the performance on multi-carbon molecules
        print('================== Start real drug prediction ==============\n')
        print("Loading Multi-Carbon Chiral Molecule Dataset")
        
        multi_carbon_path = args.dataset_root + "dataset/ECD/multiple_chiral_molecule_200/multi_chiral_molecule_200.csv"
        
        test_loader_atom_bond, test_loader_bond_angle, test_smiles = GetMultiCarbonDataloader(args, multi_carbon_path)
        save_path = f'ckpts/model_{args.model_name}/best-2rang/'
        model.load_state_dict(torch.load(save_path+'model_save_{}.pth'.format(str(args.visual_epoch))))

        gt, pred, results = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, args.visual_epoch)

    if args.MODE == 'Visual':
        print('================== Start visualizing ==============\n')
        save_path = f'ckpts/model_{args.model_name}/best-2rang/'
        visual_path = f'figs/ecd/{args.model_name}'
        analyse_dir_path = visual_path + f'/analyse_visualization/' # 存放对结果的统计图片
        case_dir_path = visual_path + f'/case_visualization/'       # 存放prediction和gt的对比图
        try: 
            os.makedirs(analyse_dir_path)
            os.makedirs(case_dir_path)
        except OSError: pass

        model.load_state_dict(torch.load(save_path+'/model_save_{}.pth'.format(str(args.visual_epoch))))
        gt, pred, results = test_model(args, model, model_device, test_loader_atom_bond, test_loader_bond_angle, args.visual_epoch)

        # ans_dict = {'gt': gt, 'pred': pred}
        # json.dump(ans_dict, open(visual_path+f"/ans_dict.json", "w"))

        if args.model_name == 'gnn_allthree':

            ## 从gt, pred中提取number_peak, peak_position, peak_height信息并转换结构
            ## 同时在算height性能的时候, 过滤出来性能最好的例子的 smiles以及attn_weights
            num_gt, num_pred = [], []
            position_gt, position_pred =[],[]
            atom_good_cases, atom_excellent_cases, atom_bad_cases, atom_worst_cases,atom_total_cases = [],[],[],[],[]
            height_acc_dict_1_2, height_acc_dict_3_4, height_acc_dict_5_8 = {},{},{}
            position_match_dict = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[],'7':[],'8':[],}

            for batch_id in range(len(gt)):
                assert len(gt[batch_id]['pos']) == len(pred[batch_id]['pos'])
                for sample_id in range(len(gt[batch_id]['pos'])):
                    gt_peak_number, pred_peak_number = len(gt[batch_id]['pos'][sample_id]), len(pred[batch_id]['pos'][sample_id]), 
                    min_len = min(gt_peak_number, pred_peak_number)
                    if min_len == 0: continue
                    num_gt.append(gt_peak_number); num_pred.append(pred_peak_number)

                    position_gt.extend([itm for itm in gt[batch_id]['pos'][sample_id][:min_len]])
                    position_pred.extend([itm for itm in pred[batch_id]['pos'][sample_id][:min_len]])
                    tmp_gt_pos = gt[batch_id]['pos'][sample_id][:min_len]
                    tmp_pred_pos = pred[batch_id]['pos'][sample_id][:min_len]
                    for itm_id in range(min_len):
                        position_match_dict[str(itm_id+1)].append(tmp_pred_pos[itm_id]-tmp_gt_pos[itm_id])
                    
                    match_num, continue_match_num = 0, 0
                    for itm_id in range(1, min_len+1):
                        if gt[batch_id]['height'][sample_id][gt_peak_number-itm_id] == pred[batch_id]['height'][sample_id][pred_peak_number-itm_id]:
                            match_num += 1
                    for itm_id in range(1, min_len+1):
                        if gt[batch_id]['height'][sample_id][gt_peak_number-itm_id] == pred[batch_id]['height'][sample_id][pred_peak_number-itm_id]:
                            continue_match_num += 1
                        else: break
                    
                    if min_len <= 2:
                        if str(min_len-match_num) not in height_acc_dict_1_2.keys():
                            height_acc_dict_1_2[str(min_len-match_num)] = 1
                        else: height_acc_dict_1_2[str(min_len-match_num)] += 1
                    elif min_len >2 and min_len <=4:
                        if str(min_len-match_num) not in height_acc_dict_3_4.keys():
                            height_acc_dict_3_4[str(min_len-match_num)] = 1
                        else: height_acc_dict_3_4[str(min_len-match_num)] += 1
                    elif min_len >= 5:
                        if str(min_len-match_num) not in height_acc_dict_5_8.keys():
                            height_acc_dict_5_8[str(min_len-match_num)] = 1
                        else: height_acc_dict_5_8[str(min_len-match_num)] += 1
                    else: assert 1 == 0

                    total_id = batch_id*len(gt[0]['pos']) + sample_id # 计算case对应的 smiles_id

                    ## atom_excellent_cases(最好的case), 从右向左, 所有的峰都和GT的峰对上了
                    if continue_match_num == gt_peak_number:
                        atom_excellent_cases.append({'id':total_id, 'smiles':test_smiles[total_id], 
                                'attn': results['attn'][batch_id]['attn'][sample_id],'attn_mask': results['attn'][batch_id]['attn_mask'][sample_id],
                                'gt'  :{'pos':gt[batch_id]['pos'][sample_id], 'height':gt[batch_id]['height'][sample_id]},
                                'pred':{'pos':pred[batch_id]['pos'][sample_id], 'height':pred[batch_id]['height'][sample_id]},
                            })
                    elif continue_match_num / gt_peak_number >= 0.6:
                        atom_good_cases.append({'id':total_id, 'smiles':test_smiles[total_id], 
                                'attn': results['attn'][batch_id]['attn'][sample_id],'attn_mask': results['attn'][batch_id]['attn_mask'][sample_id],
                                'gt'  :{'pos':gt[batch_id]['pos'][sample_id], 'height':gt[batch_id]['height'][sample_id]},
                                'pred':{'pos':pred[batch_id]['pos'][sample_id], 'height':pred[batch_id]['height'][sample_id]},
                            })
                    
                    ## atom_worst_cases(最差的case), 从右向左, 所有的峰都和GT的峰都没匹配
                    if continue_match_num==0 and match_num==0:
                        atom_worst_cases.append({'id':total_id, 'smiles':test_smiles[total_id], 
                                'attn': results['attn'][batch_id]['attn'][sample_id],'attn_mask': results['attn'][batch_id]['attn_mask'][sample_id],
                                'gt'  :{'pos':gt[batch_id]['pos'][sample_id], 'height':gt[batch_id]['height'][sample_id]},
                                'pred':{'pos':pred[batch_id]['pos'][sample_id], 'height':pred[batch_id]['height'][sample_id]},
                            })

                    ## 记录所有的cases 在atom_total_cases中
                    atom_total_cases.append({'id':total_id, 'smiles':test_smiles[total_id], 
                        'attn': results['attn'][batch_id]['attn'][sample_id],'attn_mask': results['attn'][batch_id]['attn_mask'][sample_id],
                        'prob': results['attn'][batch_id]['peak_prob'][sample_id],
                        'gt'  :{'pos':gt[batch_id]['pos'][sample_id], 'height':gt[batch_id]['height'][sample_id]},
                        'pred':{'pos':pred[batch_id]['pos'][sample_id], 'height':pred[batch_id]['height'][sample_id]},})  
                        
            ## 绘制 Atom-Attn-Weight 图
            print("Total={}, excellent={}, good={}, worst={}".format(len(atom_total_cases),len(atom_excellent_cases),len(atom_good_cases),len(atom_worst_cases)))
            json.dump(atom_excellent_cases, open(visual_path+f"/excellent_atom_info.json", "w"))
            json.dump(atom_good_cases, open(visual_path+f"/good_atom_info.json", "w"))
            json.dump(atom_worst_cases, open(visual_path+f"/worst_atom_info.json", "w"))
            json.dump(atom_total_cases, open(visual_path+f"/total_atom_info.json", "w"))
