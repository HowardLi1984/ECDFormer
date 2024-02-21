# 生成SPMS数据

import os
import numpy as np
import torch
from tqdm import tqdm

from SPMS import SPMS

def check_molecule_id():
    # 查看SPMS的ID是否和原先数据集的ID匹配
    # result: SPMS的ID完全能覆盖原先dataset的ID 
    root_path = "/root/workspace/aigc/ChemGNN/dataset/ECD/"
    spms_path = os.path.join(root_path, "SPMS/sdf")
    ecd_paths = [
        os.path.join(root_path, "500ECD/data/"), 
        os.path.join(root_path, "501-2000ECD/data/"),
        os.path.join(root_path, "2k-6kECD/data/"), 
        os.path.join(root_path, "6k-8kECD/data/"),
        os.path.join(root_path, "8k-11kECD/data/"),
    ]
    
    # get all spms ids
    spms_ids = []
    spms_files = os.listdir(spms_path)
    for file in spms_files:
        if file.find(".sdf") == -1: continue
        spms_id = int(file[:-4].split("_")[-1])
        spms_ids.append(spms_id)
    
    for filepath in ecd_paths:
        files = os.listdir(filepath)
        for file in files:
            if file.find(".csv") == -1: continue
            ecd_id = int(file[:-4])
            if ecd_id not in spms_ids:
                print(ecd_id)

def generate_spms():
    ## Initiaze the SPMS
    spms = SPMS('L-proline.sdf',key_atom_num=[],desc_n=40,desc_m=40,sphere_radius=8)
    ## Calculate the SPMS
    spms.GetSphereDescriptors()
    desc = spms.sphere_descriptors
    # desc.type = numpy.ndarray, desc.shape = 40, 80
    
    root_path = "/root/workspace/aigc/ChemGNN/dataset/ECD/"
    spms_path = os.path.join(root_path, "SPMS/sdf")
    ecd_paths = [
        os.path.join(root_path, "500ECD/data/"), 
        os.path.join(root_path, "501-2000ECD/data/"),
        os.path.join(root_path, "2k-6kECD/data/"), 
        os.path.join(root_path, "6k-8kECD/data/"),
        os.path.join(root_path, "8k-11kECD/data/"),
    ]
    result_path = '/root/workspace/aigc/ChemGNN/dataset/ECD/SPMS/spms_gen/'
    
    for filepath in ecd_paths:
        files = os.listdir(filepath)
        for file in tqdm(files):
            if file.find(".csv") == -1: continue
            ecd_id = int(file[:-4])
            spms_name = "/molecule_{}.sdf".format(str(ecd_id))
            spms_calculater = SPMS(spms_path + spms_name, 
                                   key_atom_num=[], # 空，默认取质心作为球面中心
                                   desc_n=40,
                                   desc_m=40,
                                   sphere_radius=8)
            ## Calculate the SPMS
            spms_calculater.GetSphereDescriptors()
            desc = spms_calculater.sphere_descriptors
            np.save(result_path+"{}.npy".format(str(ecd_id)), desc) # desc.shape = 40, 80

if __name__ == "__main__":
    generate_spms()