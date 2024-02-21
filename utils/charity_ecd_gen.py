## 生成各种数据集中间文件


import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from compound_tools import predict_SMILES_info


def generate_ECD_charity():
    # ECD 手性数据使用的是更新版的csv文件，因此要重新设计新CSV文件的加载方式
    # 生成 dataset_charity_ECD.npy
    # [
    #      {'id': 1, 'SMILES': 'O=C(OC)N', 'info': {redik information}}
    # ]

    # read the total all_column_charity from HPLC task
    HPLC_all = pd.read_csv('../dataset/HPLC/All_column_charity.csv')  # shape: (25867,)
    bad_all_index = np.load('../dataset/HPLC/bad_all_column.npy') #Some compounds that cannot get 3D conformer by RDKit
    HPLC_all = HPLC_all.drop(bad_all_index)
    all_smile_all = HPLC_all['SMILES'].values.tolist()  # shape: (25847, )
    T1_all = HPLC_all['RT'].values
    Speed_all = HPLC_all['Speed'].values
    Prop_all = HPLC_all['i-PrOH_proportion'].values
    dataset_all = np.load('../dataset/HPLC/dataset_charity_all_column.npy',allow_pickle=True).tolist()
    index_all = HPLC_all['Unnamed: 0'].values.tolist() # the first unnamed column
    Column_info = HPLC_all['Column'].values
    Column_charity_index = HPLC_all['index'].values
    assert len(index_all) == len(dataset_all)

    HPLC_all_info = {}
    for i in range(len(dataset_all)):
        if index_all[i] not in HPLC_all_info.keys():
            HPLC_all_info[index_all[i]] = dict(
                smiles = all_smile_all[i],
                info = dataset_all[i],
            )
        else:
            print(i, index_all[i])
            assert "key error when forming HPLC_all_info"

    # read the ECD column charity from ECD task
    ECD_all = pd.read_csv("../dataset/ECD/ecd_info.csv", encoding='gbk')
    index_ECD = ECD_all['Unnamed: 0'].values.tolist()
    hand_index_ECD = ECD_all['index'].values
    smiles_ECD = ECD_all['SMILES'].values.tolist()
    assert len(index_ECD) == len(smiles_ECD)

    ECD_all_info = []
    for i in range(len(index_ECD)):
        assert HPLC_all_info[index_ECD[i]]['smiles'] == smiles_ECD[i]
        ECD_all_info.append(
            dict(
                id = index_ECD[i],
                hand_id = hand_index_ECD[i],
                smiles = smiles_ECD[i],
                info = HPLC_all_info[index_ECD[i]]['info'],
            )
        )
    print("successfully generate ECD_column_charity.npy")
    ECD_all_info = np.array(ECD_all_info)
    np.save("../dataset/ECD/ecd_column_charity.npy", ECD_all_info)

def generate_ECD_with_new_SMILES():
    # by龙达, HPLC给的SMILES存在结构上的歧义, 我们通过SMILES-> Mols->SMILES生成了新的smiles
    # output: ecd_new_smile_charity.npy

    new_smiles_csv = pd.read_csv("../dataset/ECD/ecd_info_with_new_smiles.csv", encoding='gbk')
    smiles_all = new_smiles_csv['SMILES'].values.tolist()  # shape: (25847, )
    new_smiles_all = new_smiles_csv['New_SMILES'].values.tolist()
    hand_index_ECD = new_smiles_csv['index'].values
    index_ECD = new_smiles_csv['Unnamed: 0'].values.tolist()

    ECD_all_info = np.load("../dataset/ECD/ecd_column_charity.npy", allow_pickle=True).tolist()
    assert len(ECD_all_info) == len(index_ECD)
    new_smiles_info = []

    for i in tqdm(range(len(ECD_all_info))):
        itm = ECD_all_info[i]
        assert itm['id'] == index_ECD[i]
        assert itm['hand_id'] == hand_index_ECD[i]
        assert itm['smiles'] == smiles_all[i]

        itm['info'] = predict_SMILES_info(new_smiles_all[i])
        new_smiles_info.append(itm)

    new_smiles_info = np.array(new_smiles_info)
    np.save("../dataset/ECD/ecd_column_charity_new_smiles.npy", new_smiles_info)



if __name__ == "__main__":
    # generate_ECD_charity()

    generate_ECD_with_new_SMILES()