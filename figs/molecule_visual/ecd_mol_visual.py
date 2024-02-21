## by Lihao
## 用来做ECD光谱任务的分子中每个原子对结果的贡献程度
## 以及生成整个可视化, 包括带权重的分子结构, 原始的分子结构, 预测/GT光谱

import os
import warnings
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDPaths
 
import dgl
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dgl import model_zoo 
from dgl.data.chem.utils import mol_to_complete_graph, mol_to_bigraph
from dgl.data.chem.utils import atom_type_one_hot
from dgl.data.chem.utils import atom_degree_one_hot
from dgl.data.chem.utils import atom_formal_charge
from dgl.data.chem.utils import atom_num_radical_electrons
from dgl.data.chem.utils import atom_hybridization_one_hot
from dgl.data.chem.utils import atom_total_num_H_one_hot
from dgl.data.chem.utils import one_hot_encoding
from dgl.data.chem import CanonicalAtomFeaturizer
from dgl.data.chem import CanonicalBondFeaturizer
from dgl.data.chem import ConcatFeaturizer
from dgl.data.chem import BaseAtomFeaturizer
from dgl.data.chem import BaseBondFeaturizer
 
from dgl.data.chem import one_hot_encoding
from dgl.data.utils import split_dataset
 
from functools import partial
from sklearn.metrics import roc_auc_score

# 建立可视化模型
import copy
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from IPython.display import display
import matplotlib
import matplotlib.cm as cm
# import cairosvg
import tempfile

# import sys
# sys.path.append('../../utils/')
from utils.draw_ecd import rendering_ecd_pred_gt, rendering_drug_prediction


def origin_molecule_visualization(smiles, save_path):
    ## 对molecule最原始的样貌进行可视化
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)
    op = drawer.drawOptions()

    ## 保存不加weight的原始分子SVG图片
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    original_molecule_svg = drawer.GetDrawingText()
    with open(save_path+'/original.svg', 'w') as f:  f.write(original_molecule_svg)
    print("success in generating {}".format(save_path+'/original.svg'))

# def weight_molecule_visualization(smiles, atom_weights, save_path):
#     # 对molecule中原子重要性 进行可视化
#     number_of_nodes = atom_weights.size(0)
#     min_value = torch.min(atom_weights)
#     max_value = torch.max(atom_weights)
#     atom_weights = (atom_weights - min_value) / (max_value - min_value)

#     norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
#     cmap = cm.get_cmap('bwr')
#     plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
#     atom_colors = {i: plt_colors.to_rgba(atom_weights[i].data.item()) for i in range(number_of_nodes)}

#     mol = Chem.MolFromSmiles(smiles)
#     rdDepictor.Compute2DCoords(mol)
#     drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
#     drawer.SetFontSize(1)
#     op = drawer.drawOptions()
    
#     mol = rdMolDraw2D.PrepareMolForDrawing(mol)
#     drawer.DrawMolecule(mol, highlightAtoms=range(number_of_nodes), highlightBonds=[],
#                             highlightAtomColors=atom_colors)
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText()

#     mol, aw = Chem.MolFromSmiles(smiles), atom_weights.data.numpy()
#     with tempfile.NamedTemporaryFile(delete=True) as tmp:
#         tmp.write(svg.encode())
#         tmp.flush()
#         cairosvg.svg2png(url=tmp.name, write_to=f"{save_path}/weighted.png")
#         print("success in generating {}".format(save_path+'/weighted.png'))

def generate_ecd_atoms_visual(atom_info, img_save_path):
    # 在化学分子上绘制atom贡献程度, 输入atom_info的dict, 以及存放image的路径
    try: os.makedirs(img_save_path)
    except OSError: pass

    org_seq_dict = json.load(open("/root/workspace/aigc/ChemGNN/dataset/ECD/smiles_org.json", "r"))
    smiles_input, atom_weights_input = [], []
    for itm in atom_info:
        smiles_input.append(itm['smiles'])
        attn_weights_origin, attn_mask = itm['attn'], itm['attn_mask']
        attn_weights = []
        for i in range(len(attn_weights_origin)):
            if attn_mask[i]==0.0: attn_weights.append(attn_weights_origin[i])
        attn_weights = torch.tensor(attn_weights)
        atom_weights_input.append(attn_weights)
    
    assert len(smiles_input) == len(atom_weights_input)
    for id in range(len(smiles_input)):
        case_path = os.path.join(img_save_path, str(id))
        try: os.makedirs(case_path)
        except OSError: pass

        origin_molecule_visualization(
            smiles=smiles_input[id],
            save_path=case_path,
        )
        # weight_molecule_visualization(
        #     smiles=smiles_input[id], atom_weights=atom_weights_input[id],
        #     save_path=case_path,
        # )
        rendering_ecd_pred_gt(
            ecd_infos = atom_info[id], org_seq_dict=org_seq_dict,
            save_path=case_path,
        )    

def generate_molecule_id(atom_info, file_save_name):
    # 生成 good, excellent, bad 分子们在excel中的index
    ECD_all = pd.read_csv('/root/workspace/aigc/ChemGNN/dataset/ECD/ecd_info.csv', encoding='gbk')
    ECD_smile_all = ECD_all['SMILES'].values
    firstline_all = ECD_all['Unnamed: 0'].values.tolist()
    index_all = ECD_all['index'].values.tolist()
    
    smiles_dict = {}
    for i in range(len(ECD_smile_all)):
        smile = ECD_smile_all[i]
        if smile not in smiles_dict.keys():
            smiles_dict[smile] = {'firstline_id': firstline_all[i], 'index_id': index_all[i]}
        else: assert "error occurs due to the smile mismatch"
    
    result_dict = {}
    for i in range(len(atom_info)):
        itm = atom_info[i]
        result_dict[i] = dict(
            smiles = itm['smiles'], 
            firstline_id = smiles_dict[itm['smiles']]['firstline_id'],
            index_id = smiles_dict[itm['smiles']]['index_id'],
        )
    print("final id in {} is {}".format(file_save_name, len(atom_info)-1))
    json.dump(result_dict, open(file_save_name, "w"))


def visual_cases_in_test_dataset():
    root_path = "/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/"
    excellent_atom_info = json.load(open(root_path+"excellent_atom_info_new.json", "r"))
    # good_atom_info = json.load(open(root_path+"good_atom_info_new.json", "r"))
    # bad_atom_info = json.load(open(root_path+"bad_atom_info_new.json", "r"))
    # worst_atom_info = json.load(open(root_path+"worst_atom_info_new.json", "r"))

    ## 根据 atom_info.json 来生成case_id
    # generate_molecule_id(atom_info=excellent_atom_info, file_save_name="./excellent_case_id.json")
    # generate_molecule_id(atom_info=good_atom_info, file_save_name="./good_case_id.json")
    # generate_molecule_id(atom_info=bad_atom_info, file_save_name="./bad_case_id.json")
    # generate_molecule_id(atom_info=worst_atom_info, file_save_name="./worst_case_id.json")

    generate_ecd_atoms_visual(atom_info=excellent_atom_info, img_save_path="./excellent_cases/")
    # generate_ecd_atoms_visual(atom_info=good_atom_info, img_save_path="./good_cases/")
    # generate_ecd_atoms_visual(atom_info=bad_atom_info, img_save_path="./bad_cases/")
    # generate_ecd_atoms_visual(atom_info=worst_atom_info, img_save_path="./worst_cases/")

def visual_real_drugs(root_path, json_path, img_save_path):
    # drug_atom_info = json.load(open(root_path+"drug_atom_info_single.json", "r"))
    # img_save_path = "./drug_cases_single/"
    
    # drug_atom_info = json.load(open(root_path+"drug_atom_info_multi.json", "r"))
    # img_save_path = "./drug_cases_multi/"
    
    drug_atom_info = json.load(open(root_path+json_path, "r"))
    try: os.makedirs(root_path+img_save_path)
    except OSError: pass
    
    for itm in drug_atom_info:
        case_path = root_path + img_save_path + str(itm['id'])
        try: os.makedirs(case_path)
        except OSError: pass
        my_sequence_args = {
            'front_pad': 150, 
            'distance': 6, 
            'pos_height': 30.5, 
            'neg_height': -35.0
        }
        rendering_drug_prediction(
            ecd_infos = itm, sequence_args=my_sequence_args,
            save_path=case_path,
        )  
        origin_molecule_visualization(
            smiles=itm['smiles'],
            save_path=case_path,
        )  


if __name__ == "__main__": 
    ## 单独评测可视化部分
    # test_smiles = ['CC(C)CC(C)C', 'C=CCCC']
    # test_atom_weights = [
    #     torch.tensor([[0.1529],[0.1429],[0.1429],[0.0429],[0.1429],[0.1429],[0.1429]]),
    #     torch.tensor([[0.3000],[0.2000],[0.1000],[0.2000],[0.2000]]),
    # ]
    # molecule_visualization(
    #     mol_smiles=test_smiles, mol_atom_weights=test_atom_weights, save_path="./test_imgs2/",
    # )

    visual_cases_in_test_dataset()
    visual_real_drugs(
        root_path="/remote-home/lihao/ai4s/ECDFormer/open_source/figs/gnn_allthree/",
        json_path="drug_atom_info_multi.json", img_save_path="drug_cases_multi/",
    )
    
