# 序列预测分解任务对应的数据集

import os
import json
import random
import numpy as np
import pandas as pd

import torch

from scipy.signal import find_peaks
from torch_geometric.data import DataLoader
# from torch_geometric.loader import DataListLoader
from torch_geometric.data import Data
from tqdm import tqdm

from utils.util_func import normalize_func
from utils.eval_func import get_sequence_peak

def x_bin_position(real_x, distance):
    # 输入IR光谱中真实的x轴坐标, 返回箱切割之后的箱ID
    # real_x: 500-4000范围内的x坐标, distance: 将500-4000区间按照distance切割
    return int((real_x-500) / distance)

def get_key_padding_mask(tokens):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == -1] = -torch.inf
    return key_padding_mask

def Construct_dataset(args, dataset, data_index):
    graph_info = []

    # column_descriptor = np.load('utils/column_descriptor.npy', allow_pickle=True) # (25, 1826)
    all_descriptor = np.load(args.dataset_root + 'utils/descriptor_all_column.npy')                   # (25847, 1826)

    for i in tqdm(range(len(dataset)), desc="mole_graph_gen"):
        data = dataset[i]
        # col_info = column_name[i]                # 对应csv中的Column列
        # col_specify = args.column_specify[col_info]
        # col_des = np.array(column_descriptor[col_specify[3]])
        atom_feature, bond_feature = [], []
        for name in args.atom_id_names:
            atom_feature.append(data[name]) # len(data[name])=23
        for name in args.bond_id_names[0:3]:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(int(data_index[i]))).to(torch.int64)

        TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[0, 820]/100
        RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[0, 821]
        RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[0, 822]
        MDEC=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[0, 1568]
        MATS=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[0, 457]

        bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)

        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)

        data_atom_bond = Data(atom_feature, edge_index, bond_feature, data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)

        tmp_dict = dict(
            id = data_index[i], graph_atom_bond = data_atom_bond, graph_bond_angle=data_bond_angle
        )
        graph_info.append(tmp_dict)
    
    graph_info.sort(key=lambda x:int(x['id']))

    return graph_info

def read_total_ir(args, index_all, max_peak=15): 
    ir_dict, ir_origianl_dict = dict(), dict()

    ir_final_list = []
    ## read ir spectrum from the json files
    for fileid in tqdm(index_all, desc="ir_peak_gen"):
        filepath = os.path.join(args.dataset_root + "dataset/IR/qm9_ir_spec", f"{fileid}.json")
        raw_ir_info = json.load(open(filepath, "r"))

        ir_x = raw_ir_info['x']
        ir_width_10, ir_width_40 = raw_ir_info['y_10'], raw_ir_info['y_40']
        assert min(ir_x) == 500. and max(ir_x) == 4000.
        assert len(ir_width_10) == len(ir_width_40) == len(ir_x)

        # get peak information
        peaks_raw, peak_infos = find_peaks(x=ir_width_40, height=0.1, distance=100)
        peaks_raw = peaks_raw.tolist()

        peak_num, peak_position_list, peak_height_list = 0, [], []
        if len(peaks_raw) > max_peak:
            peak_num = max_peak
            peaks = peaks_raw[len(peaks_raw)-max_peak: ]  # remove the peaks from left
            peak_position_list = [x_bin_position(ir_x[i], 100) for i in peaks]
            peak_height_list = [ir_width_40[i] for i in peaks]
        else:
            peak_num = len(peaks_raw)
            peak_position_list = [x_bin_position(ir_x[i], 100) for i in peaks_raw] + [-1]*(max_peak - len(peaks_raw))
            peak_height_list = [ir_width_40[i] for i in peaks_raw] + [-1]*(max_peak - len(peaks_raw))
            peaks = peaks_raw + [-1]*(max_peak - len(peaks_raw))

        query_padding_mask = get_key_padding_mask(torch.tensor(peak_position_list))
        
        tmp_ir_dict = dict(
            id = fileid,
            seq_10 = ir_width_10, 
            seq_40 = ir_width_40, 
            peak_num = peak_num,
            peak_position = peak_position_list,
            peak_height = peak_height_list,
            query_mask = query_padding_mask.unsqueeze(0),
        )
        ir_final_list.append(tmp_ir_dict)
    
    ir_final_list.sort(key=lambda x:int(x['id']))

    return ir_final_list 

def GetAtomBondAngleDataloader(args, dataset_all, smiles_all, index_all):
    # given random seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.random.manual_seed(args.rand_seed)

    ir_sequences = read_total_ir(args, index_all)
    total_graph_info = Construct_dataset(args, dataset_all, index_all)
    assert len(ir_sequences) == len(total_graph_info)
    print("Case Before Process = ", len(total_graph_info), len(smiles_all))

    # split graphs containing ECD sequences, put ECD sequence into dataset_graph_atom_bond
    dataset_graph_atom_bond, dataset_graph_bond_angle, dataset_smiles = [], [], []
    for i in range(len(ir_sequences)):
        assert total_graph_info[i]['id'] == ir_sequences[i]['id']
        itm = ir_sequences[i]

        atom_bond = total_graph_info[i]['graph_atom_bond']
        atom_bond['sequence'] = torch.tensor([itm['seq_40']])
        atom_bond['ecd_id'] = torch.tensor(int(itm['id']))
        atom_bond['peak_num'] = torch.tensor([itm['peak_num']])
        atom_bond['peak_position'] = torch.tensor([itm['peak_position']])
        atom_bond['peak_height'] = torch.tensor([itm['peak_height']])
        atom_bond['query_mask'] = itm['query_mask']
        dataset_graph_atom_bond.append(atom_bond)
        dataset_graph_bond_angle.append(total_graph_info[i]['graph_bond_angle'])
        dataset_smiles.append(smiles_all[i])

    total_num = len(dataset_graph_atom_bond)
    print("Case After Process = ",len(dataset_graph_atom_bond), len(dataset_graph_bond_angle), len(dataset_smiles))

    # automatic dataloading and splitting
    data_array = np.arange(0, total_num, 1)

    np.random.shuffle(data_array)

    train_data_atom_bond, valid_data_atom_bond, test_data_atom_bond = [], [], []
    train_data_bond_angle, valid_data_bond_angle, test_data_bond_angle = [], [], []
    train_smiles, valid_smiles, test_smiles = [], [], []

    train_num = int(len(data_array) * args.train_ratio)
    val_num = int(len(data_array) * args.valid_ratio)
    test_num = int(len(data_array) * args.test_ratio)

    train_index, valid_index = data_array[0:train_num], data_array[train_num:train_num + val_num]
    if args.test_mode == 'fixed':
        test_index = data_array[total_num-test_num:]
    if args.test_mode=='random':
        test_index = data_array[train_num + val_num:train_num + val_num + test_num]


    for i in test_index:
        test_data_atom_bond.append(dataset_graph_atom_bond[i])
        test_data_bond_angle.append(dataset_graph_bond_angle[i])
        test_smiles.append(dataset_smiles[i])

    for i in valid_index:
        valid_data_atom_bond.append(dataset_graph_atom_bond[i])
        valid_data_bond_angle.append(dataset_graph_bond_angle[i])
        valid_smiles.append(dataset_smiles[i])

    for i in train_index:
        train_data_atom_bond.append(dataset_graph_atom_bond[i])
        train_data_bond_angle.append(dataset_graph_bond_angle[i])
        train_smiles.append(dataset_smiles[i])
    
    print('=================== Data preprared ================\n')

    train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return (train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond), \
           (train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle), \
           (train_smiles, valid_smiles, test_smiles)