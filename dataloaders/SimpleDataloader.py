
import os
import math

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.util_func import normalize_func

def read_total_ecd(sample_path, fix_length=20): 
    # For full-size datasamples (500 + 1500 cases)
    filepaths = [
        os.path.join(sample_path, "500ECD/data/"), 
        # os.path.join(sample_path, "501-2000ECD/data/"),
        # os.path.join(sample_path, "2k-6kECD/data/"), 
        # os.path.join(sample_path, "6k-8kECD/data/"),
        # os.path.join(sample_path, "8k-11kECD/data/"),
    ]
    ecd_dict = {}
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

    # select fix_length itms from ecd original sequence
    ecd_final_list = []
    for key, itm in ecd_dict.items():
        distance = int(len(itm['ecd'])/(fix_length-1))
        sequence_org = [itm['ecd'][i] for i in range(0, len(itm['ecd']), distance)][:fix_length]
        ## normalization for sequences
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

        tmp_dict = dict(
            id = key,
            seq = [0]+sequence, 
            seq_original = sequence_org, 
            seq_mask=peak_mask,
        )
        ecd_final_list.append(tmp_dict)
    
    ecd_final_list.sort(key=lambda x:x['id'])
    return ecd_final_list  

class PTRDataset(Dataset):
    '''
    input feature of Pretrained Molecule Models    
    '''
    def __init__(self, data_path, ecd_path):
        total_info = np.load(data_path, allow_pickle=True).tolist()
        # total_info[0].keys  ['smiles', 'graph_info', 'feature_info']
        ecd_sequences = read_total_ecd(ecd_path)
        self.graph_feature, self.ecd_sequences = [], [] 

        for i in range(len(ecd_sequences)):
            self.ecd_sequences.append(np.array(ecd_sequences[i]['seq']))
        for i in range(len(total_info)):
            self.graph_feature.append(total_info[i]['feature_info']['graph'])
        
        self.graph_feature = self.graph_feature[:len(self.ecd_sequences)]
        assert len(self.graph_feature) == len(self.ecd_sequences)
        self.total_length = len(self.ecd_sequences)
    
    def __getitem__(self, item):
        return self.graph_feature[item], self.ecd_sequences[item]

    def __len__(self):
        return self.total_length