import random
import numpy as np
import pandas as pd

import torch

from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

def Construct_dataset(args, dataset, data_index, T1, speed, eluent, column, column_name):
    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []

    column_descriptor = np.load('utils/column_descriptor.npy', allow_pickle=True) # (25, 1826)
    all_descriptor = np.load('utils/descriptor_all_column.npy')                   # (25847, 1826)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        col_info = column_name[i]                # 对应csv中的Column列
        col_specify = args.column_specify[col_info]
        col_des = np.array(column_descriptor[col_specify[3]])
        atom_feature = []
        bond_feature = []
        for name in args.atom_id_names:
            atom_feature.append(data[name]) # len(data[name])=23
        for name in args.bond_id_names[0:3]:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([float(T1[i]) * float(speed[i])])
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)

        prop=torch.ones([bond_feature.shape[0]])*eluent[i]
        coated = torch.ones([bond_feature.shape[0]]) * col_specify[0]
        diameter = torch.ones([bond_feature.shape[0]]) * col_specify[1]
        immobilized = torch.ones([bond_feature.shape[0]]) * col_specify[2]

        TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820]/100
        RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
        RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
        MDEC=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
        MATS=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]

        if args.Use_geometry_enhanced==True:
            col_TPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[820] / 100
            col_RASA = torch.ones([bond_angle_feature.shape[0]]) * col_des[821]
            col_RPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[822]
            col_MDEC = torch.ones([bond_angle_feature.shape[0]]) * col_des[1568]
            col_MATS = torch.ones([bond_angle_feature.shape[0]]) * col_des[457]
        else:
            col_TPSA = torch.ones([bond_feature.shape[0]]) * col_des[820] / 100
            col_RASA = torch.ones([bond_feature.shape[0]]) * col_des[821]
            col_RPSA = torch.ones([bond_feature.shape[0]]) * col_des[822]
            col_MDEC = torch.ones([bond_feature.shape[0]]) * col_des[1568]
            col_MATS = torch.ones([bond_feature.shape[0]]) * col_des[457]
        if args.Use_column_info == True:
            bond_feature = torch.cat([bond_feature, coated.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, immobilized.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, prop.reshape(-1, 1)], dim=1)

        if args.Use_column_info==True:
            bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)

        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)

        if args.Use_column_info==True:
            if args.Use_geometry_enhanced==True:
                bond_angle_feature = torch.cat([bond_angle_feature, col_TPSA.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_RASA.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_RPSA.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_MDEC.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_MATS.reshape(-1, 1)], dim=1)
            else:
                bond_feature = torch.cat([bond_feature, col_TPSA.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_RASA.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_RPSA.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_MDEC.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_MATS.reshape(-1, 1)], dim=1)

        if y[0] > 60:
            big_index.append(i)
            continue

        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y, data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)

    return graph_atom_bond, graph_bond_angle, big_index

def GetAtomBondAngleDataloader(
        args, dataset_all, index_all, T1_all, 
        Speed_all, Prop_all, transfer_target, Column_info, 
    ):
    assert args.MODEL in ['Train','Test']

    # given random seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.random.manual_seed(args.rand_seed)
    
    dataset_graph_atom_bond, dataset_graph_bond_angle, big_index = Construct_dataset(
        args, dataset_all, index_all, T1_all,
        Speed_all, Prop_all, transfer_target, Column_info
    )
    total_num = len(dataset_graph_atom_bond)
    # automatic dataloading and splitting

    if args.test_mode=='enantiomer':
        '''
        Randomly select enantiomers
        '''
        index_all, fix_index = [], []
        for i in range(len(dataset_graph_atom_bond)):
            index_all.append(int(dataset_graph_atom_bond[i].data_index.data.numpy()))

        charity_all_index = np.unique(np.array(Column_charity_index)).tolist()
        HPLC_all_save = pd.read_csv('dataset/All_column_charity.csv')
        Column_charity_index_save = HPLC_all_save['index'].values
        
        select_num = random.sample(charity_all_index, 500)
        #print(select_num[0:10])

        index_loc = []
        for i in select_num:
            loc = np.where(np.array(Column_charity_index_save) == i)[0]
            index_loc.extend(loc)
        #print(index_loc[0:10])

        #(HPLC_all_save.iloc[index_loc]).to_excel('dataset/test_compound_charity.xlsx')

        for i in index_loc:
            if len(np.where(np.array(index_all) == i)[0]) > 0:
                fix_index.append(np.where(np.array(index_all) == i)[0][0])
        #print(fix_index[0:10])

        print(len(fix_index))
        data_array = np.arange(0, total_num, 1)
        data_array = np.delete(data_array, fix_index)
    else:
        data_array = np.arange(0, total_num, 1)

    np.random.shuffle(data_array)

    train_data_atom_bond, valid_data_atom_bond, test_data_atom_bond = [], [], []
    train_data_bond_angle, valid_data_bond_angle, test_data_bond_angle = [], [], []
    train_column_atom_bond, valid_column_atom_bond, test_column_atom_bond = [], [], []
    train_column_bond_angle, valid_column_bond_angle, test_column_bond_angle = [], [], []

    train_num = int(len(data_array) * args.train_ratio)
    val_num = int(len(data_array) * args.valid_ratio)
    test_num = int(len(data_array) * args.test_ratio)

    if args.test_mode == 'enantiomer':
        train_num, val_num = int(total_num * args.train_ratio), int(total_num * args.valid_ratio)
        train_index, valid_index = data_array[0:train_num], data_array[train_num:train_num+val_num]
        test_index = fix_index
    else:
        train_index, valid_index = data_array[0:train_num], data_array[train_num:train_num + val_num]
        if args.test_mode == 'fixed':
            test_index = data_array[total_num-test_num:]
        if args.test_mode=='random':
            test_index = data_array[train_num + val_num:train_num + val_num + test_num]


    for i in test_index:
        test_data_atom_bond.append(dataset_graph_atom_bond[i])
        test_data_bond_angle.append(dataset_graph_bond_angle[i])

    for i in valid_index:
        valid_data_atom_bond.append(dataset_graph_atom_bond[i])
        valid_data_bond_angle.append(dataset_graph_bond_angle[i])

    for i in train_index:
        train_data_atom_bond.append(dataset_graph_atom_bond[i])
        train_data_bond_angle.append(dataset_graph_bond_angle[i])

    print('========Data  preprared!=============')
    print(test_data_atom_bond[0].y,test_data_atom_bond[477].data_index)

    train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return (train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond), \
           (train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle)
