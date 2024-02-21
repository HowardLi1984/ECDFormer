## only for loading the real drug as test dataset

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from utils.util_func import normalize_func
from utils.eval_func import get_sequence_peak
from utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from rdkit.Chem import AllChem

def Construct_dataset(args, dataset, data_index):
    graph_atom_bond = []
    graph_bond_angle = []

    # column_descriptor = np.load('utils/column_descriptor.npy', allow_pickle=True) # (25, 1826)
    all_descriptor = np.load('utils/descriptor_all_column.npy')                   # (25847, 1826)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
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
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)

        TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820]/100
        RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
        RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
        MDEC=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
        MATS=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]

        bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)

        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)

        data_atom_bond = Data(atom_feature, edge_index, bond_feature, data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)

    return graph_atom_bond, graph_bond_angle

def GetDrugDataloader(args, drug_path):
    # dataset_all, smiles_all, index_all
    raw_index_all, raw_smiles_all = [], []
    if ".csv" in drug_path:
        drug_all = pd.read_csv(drug_path, encoding='gbk')
        print(drug_all.keys())
        raw_smiles_all = drug_all['Smiles'].values.tolist()
        # raw_index_all = drug_all['Unnamed: 0'].values.tolist()   ## for single-dataset
        raw_index_all = drug_all['锘縦'].values.tolist()           ## for multi-datasets
    elif ".json" in drug_path:
        drug_all = json.load(open(drug_path, "r"))
        raw_index_all = [itm['id'] for itm in drug_all]
        raw_smiles_all = [itm['smiles'] for itm in drug_all]
    else: assert "drug_path type error"

    dataset_all, smiles_all, index_all = [], [], []
    
    for i in range(len(raw_smiles_all)): 
        mol = AllChem.MolFromSmiles(raw_smiles_all[i])
        AllChem.EmbedMolecule(mol)
        try: 
            data = mol_to_geognn_graph_data_MMFF3d(mol)
            dataset_all.append(data); smiles_all.append(raw_smiles_all[i])
            index_all.append(raw_index_all[i])
        except ValueError:
            print("error in {}".format(i))

    # given random seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.random.manual_seed(args.rand_seed)

    total_graph_atom_bond, total_graph_bond_angle = Construct_dataset(args, dataset_all, index_all)
    print("Case Before Process = ", len(total_graph_atom_bond), len(total_graph_bond_angle), len(smiles_all))

    # split graphs containing ECD sequences, put ECD sequence into dataset_graph_atom_bond
    # dataset_graph_atom_bond, dataset_graph_bond_angle, dataset_smiles = [], [], smiles_all
    # total_num = len(total_graph_atom_bond)

    # # automatic dataloading and splitting
    # train_data_atom_bond, valid_data_atom_bond, test_data_atom_bond = [], [], []
    # train_data_bond_angle, valid_data_bond_angle, test_data_bond_angle = [], [], []
    # train_column_atom_bond, valid_column_atom_bond, test_column_atom_bond = [], [], []
    # train_column_bond_angle, valid_column_bond_angle, test_column_bond_angle = [], [], []
    # train_smiles, valid_smiles, test_smiles = [], [], []

    # for i in range(total_num):
    #     test_data_atom_bond.append(dataset_graph_atom_bond[i])
    #     test_data_bond_angle.append(dataset_graph_bond_angle[i])
    #     test_smiles.append(dataset_smiles[i])
    
    print('=================== Data preprared ================\n')
    test_loader_atom_bond = DataLoader(total_graph_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(total_graph_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader_atom_bond, test_loader_bond_angle, smiles_all
