## only for loading the multi-carbon chiral molecules as test dataset

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

def get_key_padding_mask(tokens):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == -1] = -torch.inf
    return key_padding_mask

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

def read_total_ecd(sample_path, fix_length=20): 
    # load ecd spectra for 200 multi-carbon chiral molecules
    filepaths = [os.path.join(sample_path, "multiple_chiral_molecule_200/multiple_chiral_molecule_200/csv/"), ]
    ecd_dict = {}
    ecd_original_dict = {}
    # read from csv dictionary
    for filepath in filepaths:
        files = os.listdir(filepath)
        for file in files:
            if file.find(".csv") == -1: continue
            fileid = int(file[:-4])
            single_file_path = os.path.join(filepath, file)
            ECD_info = pd.read_csv(single_file_path).to_dict(orient='list')
            wavelengths_o, mdegs_o = ECD_info['Wavelength (nm)'], ECD_info['ECD (Mdeg)'] # _o is original
            wavelengths = [int(i) for i in wavelengths_o]
            # change small numbers to zero
            mdegs = [int(i) if i > 1 or i < -1 else 0 for i in mdegs_o]
            # remove the zero position
            begin, end = 0, 0
            for i in range(len(mdegs)): 
                if mdegs[i] != 0: begin = i; break
            for i in range(len(mdegs)-1, 0, -1): 
                if mdegs[i] != 0: end = i; break
            ecd_dict[fileid] = dict(wavelengths = wavelengths[begin: end+1],
                                    ecd = mdegs[begin: end+1],)
            ecd_original_dict[fileid] = dict(
                wavelengths = wavelengths, ecd = mdegs,)

    # select fix_length itms from ecd original sequence
    ecd_final_list = []
    for key, itm in ecd_dict.items():
        distance = int(len(itm['ecd'])/(fix_length-1))
        sequence_org = [itm['ecd'][i] for i in range(0, len(itm['ecd']), distance)][:fix_length]
        ## normalization for sequences height
        sequence = normalize_func(sequence_org, norm_range=[-100, 100])
        
        if len(sequence) < fix_length:
            sequence.extend([0]*(fix_length - len(sequence)))
            sequence_org.extend([0]*(fix_length - len(sequence_org))) # 将未经norm的sequence也padding
        assert len(sequence) == fix_length
        
        ## generate the mask of peak values
        peak_mask = [0]*len(sequence)
        for i in range(1, len(sequence)-1): # window_size=3, [0,1(peak),0]
            if sequence[i-1]<sequence[i] and sequence[i]>sequence[i+1]:
                if peak_mask[i-1] != 2: peak_mask[i-1]=1
                peak_mask[i] = 2
                if peak_mask[i+1] != 2: peak_mask[i+1]=1
            if sequence[i-1]>sequence[i] and sequence[i]<sequence[i+1]:
                if peak_mask[i-1] != 2: peak_mask[i-1]=1
                peak_mask[i] = 2
                if peak_mask[i+1] != 2: peak_mask[i+1]=1

        ## generate peak_num, peak_position, peak_position_padding_mask, peak_height
        peak_position_list = get_sequence_peak(sequence)
        peak_number = len(peak_position_list)
        assert peak_number < 9

        peak_height_list = []
        for i in peak_position_list:
            if sequence[i] >= 0: peak_height_list.append(1)
            else: peak_height_list.append(0)
        peak_position_list = peak_position_list + [-1]*(9-peak_number) # position_list, pad成一个size
        peak_height_list = peak_height_list + [-1]*(9-peak_number)      # height_list, 也pad成一个size
        query_padding_mask = get_key_padding_mask(torch.tensor(peak_position_list))

        tmp_dict = dict(
            id = key,
            seq = [0]+sequence, 
            seq_original = sequence_org, 
            seq_mask=peak_mask,
            peak_num = peak_number,
            peak_position = peak_position_list,
            peak_height = peak_height_list,
            query_mask = query_padding_mask.unsqueeze(0),
        )
        ecd_final_list.append(tmp_dict)
    
    ecd_final_list.sort(key=lambda x:x['id'])
    return ecd_final_list, ecd_original_dict  

def GetMultiCarbonDataloader(args, molecule_path):
    # dataset_all, smiles_all, index_all
    molecule_all = pd.read_csv(molecule_path, encoding='gbk')
    raw_smiles_all = molecule_all['SMILES'].values.tolist()
    dataset_all, smiles_all, index_all = [], [], []
    
    for i in range(len(raw_smiles_all)): 
        mol = AllChem.MolFromSmiles(raw_smiles_all[i])
        AllChem.EmbedMolecule(mol)
        try: 
            data = mol_to_geognn_graph_data_MMFF3d(mol)
            dataset_all.append(data); smiles_all.append(raw_smiles_all[i])
            index_all.append(i+1)
        except ValueError:
            print("error in {}".format(i))
            mol = AllChem.MolFromSmiles(raw_smiles_all[i-1])
            AllChem.EmbedMolecule(mol)
            data = mol_to_geognn_graph_data_MMFF3d(mol)
            dataset_all.append(data); smiles_all.append(raw_smiles_all[i-1])
            index_all.append(i+1)

    # given random seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.random.manual_seed(args.rand_seed)

    ecd_sequences, ecd_original_sequences = read_total_ecd(args.sample_path)
    total_graph_atom_bond, total_graph_bond_angle = Construct_dataset(args, dataset_all, index_all)
    dataset_graph_atom_bond, dataset_graph_bond_angle, dataset_smiles = [], [], []
    for itm in ecd_sequences:
        line_num = itm['id'] - 1
        atom_bond = total_graph_atom_bond[line_num]
        atom_bond['sequence'] = torch.tensor([itm['seq']])
        atom_bond['ecd_id'] = torch.tensor(itm['id'])
        atom_bond['seq_mask'] = torch.tensor([itm['seq_mask']])
        atom_bond['seq_original'] = torch.tensor([itm['seq_original']])
        atom_bond['peak_num'] = torch.tensor([itm['peak_num']])
        atom_bond['peak_position'] = torch.tensor([itm['peak_position']])
        atom_bond['peak_height'] = torch.tensor([itm['peak_height']])
        atom_bond['query_mask'] = itm['query_mask']
        dataset_graph_atom_bond.append(atom_bond)
        dataset_graph_bond_angle.append(total_graph_bond_angle[line_num])
        dataset_smiles.append(smiles_all[line_num])
    
    print("Case Before Process = ", len(dataset_graph_atom_bond), len(dataset_graph_bond_angle), len(dataset_smiles))
    
    print('=================== Data preprared ================\n')
    test_loader_atom_bond = DataLoader(dataset_graph_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(dataset_graph_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader_atom_bond, test_loader_bond_angle, dataset_smiles
