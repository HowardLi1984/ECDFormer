# 专门绘图 ECD光谱的还原函数
import math
import random
import json
import collections
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d
# from scipy.interpolate import make_interp_spline

# def gaussian(peak_pos, peak_height, magnification=2.1): # 用高斯分布来拟合预测峰
#     ave = peak_pos      # 高斯分布的均值
#     sigma = 0.6        # 高斯分布的标准差
#     peak_height = peak_height * magnification # 通过放大后的peak_height
#     x = np.arange(peak_pos-6*sigma, peak_pos+6*sigma, 0.02)
#     y = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma, -1), np.exp(-np.power(x-ave, 2) / 2 * sigma ** 2))
#     if peak_height < 0:
#         y = np.multiply(abs(peak_height), np.negative(y))
#     else:
#         y = np.multiply(peak_height, y)
#     ## 低通滤波
#     y_new = []
#     for i in range(len(y)):
#         if i == 0 or i == len(y)-1: y_new.append(y[i])
#         else: y_new.append((y[i-1]+y[i]+y[i+1])/3) 
#     return x, np.array(y_new)

def gaussian(peak_pos, peak_height, magnification=3.5):
    peak_height = peak_height*magnification
    ave, sigma = peak_pos, 2.0
    x = np.arange(peak_pos-6, peak_pos+6, 0.02)
    zhishu = -(np.square(x-ave) / (sigma*sigma*2))
    exp = np.exp(zhishu)
    xishu = 1 / (np.sqrt(2*np.pi)*sigma)
    y = xishu*exp
    if peak_height < 0: y = np.multiply(abs(peak_height), np.negative(y))
    else: y = np.multiply(peak_height, y)
    
    ## 低通滤波
    y_new = []
    for i in range(len(y)):
        if i == 0 or i == len(y)-1: y_new.append(y[i])
        else: y_new.append((y[i-1]+y[i]+y[i+1])/3) 
    return x, np.array(y_new)

def get_true_peak_property(gt_info, org_info):
    ## 通过比对原始的ecd序列:org_info 与经过处理之后的ecd标准答案序列: gt_info
    ## 得到峰的front_pad, 峰的位置放缩指标, 正峰/负峰的平均高度, 以及峰的平均宽度
    wavelengths = [int(i) for i in org_info['wavelengths']]
    mdegs = [int(i) if i > 1 or i < -1 else 0 for i in org_info['ecd']]
    begin, end = 0, 0
    for i in range(len(mdegs)): 
        if mdegs[i] != 0: begin = i; break
    for i in range(len(mdegs)-1, 0, -1): 
        if mdegs[i] != 0: end = i; break
    
    fix_length = 20
    front_pad = wavelengths[begin]  # 前面pad的长度
    distance = int(len(mdegs[begin: end+1])/(fix_length-1)) # 峰位置缩放指标
    pos_height_list, neg_height_list = [], []
    for i in range(len(gt_info['height'])):
        if gt_info['height'][i] == 1: # positive peak
            pos_height_list.append(mdegs[begin+distance*gt_info['pos'][i]])
        else:                         # negative peak
            neg_height_list.append(mdegs[begin+distance*gt_info['pos'][i]])
    if len(pos_height_list) > 0: pos_height = sum(pos_height_list) / len(pos_height_list)
    else: pos_height = 50
    if len(neg_height_list) > 0: neg_height = sum(neg_height_list) / len(neg_height_list)
    else: neg_height = -50

    return dict(
        front_pad = front_pad, distance = distance,
        pos_height = pos_height, neg_height = neg_height,
    )

def reconstruct_ecd2(ecd_info, sequence_args):
    ## 将只有峰信息的序列还原成ECD光谱结构, width是峰周边的
    ## ecd_info = {'pos':[], 'height':[]}
    # x_info, y_info = [i for i in range(0, 450)], [0 for i in range(0, 450)]
    x_info, y_info = [], []

    # 过滤可能一致的peak_position
    new_pos, new_height = [], []
    for i in range(len(ecd_info['pos'])):
        if ecd_info['pos'][i] not in new_pos:
            new_pos.append(ecd_info['pos'][i])
            if ecd_info['height'][i] == 1: new_height.append(ecd_info['height'][i])
            else: new_height.append(-1)
    for i in range(len(new_pos)):
        pos_token = sequence_args["front_pad"]+sequence_args["distance"]*new_pos[i] # 有front_pad,就平移new_pos
        if new_height[i] == 1: height_token = sequence_args["pos_height"]
        else: height_token = sequence_args["neg_height"]
        tmp_x, tmp_y = gaussian(pos_token, height_token)
        x_info.extend(tmp_x)
        y_info.extend(tmp_y)
        # for j in range(3): y_info[pos_token-j], y_info[pos_token+j] = height_token/(j+1), height_token/(j+1)
    # x_info, y_info = x_info[80:], y_info[80:]
    x_info = [80] + x_info + [450]
    y_info = [0]  + y_info + [0]
    return {'x':x_info, 'y':y_info}

def rendering_ecd_pred_gt(ecd_infos, org_seq_dict, save_path='.'):
    ## 函数被draw_ecd.py以及fig_lh/molecule_visual/ecd_mol_visual.py共用
    ## 单张图片, 给定峰属性, 渲染出ecd光谱图片
    ## sequence_args: dict, the hyper-parameter for ecd_sequence
    ## ecd_infos: {"gt": {"pos": [5, 10, 17], "height": [0, 1, 0]}, "pred": {"pos": [5, 9, 16], "height": [0, 1, 0]}}
    
    pred_info, gt_info = ecd_infos['pred'], ecd_infos['gt']
    seq_gt_org = org_seq_dict[ecd_infos['smiles']]  # 80-450范围的gt ecd sequence
    sequence_args = get_true_peak_property(gt_info=gt_info, org_info=seq_gt_org)

    seq_pred = reconstruct_ecd2(
        ecd_info = pred_info, 
        sequence_args = sequence_args,)
    
    plt.style.use('classic')
    plt.plot( # 绘制prediction
        seq_pred['x'], seq_pred['y'],
        linewidth=4.0, color='#C6A969', linestyle='-', label='pred')
    plt.plot( # 绘制groundtruth
        seq_gt_org['wavelengths'], seq_gt_org['ecd'],
        linewidth=4.0, color='#597E52', linestyle='-', label='gt')
    
    ## 设置坐标轴范围, 刻度
    plt.xlim((80, 450))
    # plt.ylim((-2, 2))
    my_x_ticks = np.arange(80, 450, 50)
    # my_y_ticks = np.arange(-2, 2, 0.2)
    plt.xticks(my_x_ticks)
    # plt.yticks([])

    plt.tick_params(axis='y',
        labelsize=14, # y轴字体大小设置
        color='black',    # y轴标签颜色设置  
        labelcolor='black', # y轴字体颜色设置
        direction='in' # y轴标签方向设置
    ) 
    plt.tick_params(axis='x',
        labelsize=14, # x轴字体大小设置
        color='black',    # x轴标签颜色设置  
        labelcolor='black', # x轴字体颜色设置
        direction='in' # x轴标签方向设置
    ) 

    plt.legend() # 显示注释
    plt.savefig(save_path+'/pred_gt.svg', bbox_inches='tight', dpi=300)
    plt.clf()

def rendering_drug_prediction(ecd_infos, sequence_args, save_path='.'):
    ## 仅用于 real drug 的序列预测
    ## sequence_args: dict, the hyper-parameter for ecd_sequence
    ## ecd_infos: {"pred": {"pos": [5, 9, 16], "height": [0, 1, 0]}}
    
    pred_info = ecd_infos['pred']

    seq_pred = reconstruct_ecd2(
        ecd_info = pred_info, 
        sequence_args = sequence_args,)
    
    plt.style.use('classic')
    plt.plot( # 绘制prediction
        seq_pred['x'], seq_pred['y'],
        linewidth=2.0, color='red', linestyle='-', label='pred')
    
    ## 设置坐标轴范围, 刻度
    plt.xlim((80, 450))
    my_x_ticks = np.arange(80, 450, 50)
    plt.xticks(my_x_ticks)

    plt.ylim((-80, 80))
    my_y_ticks = np.arange(-80, 80, 10)
    plt.yticks(my_y_ticks)

    plt.tick_params(axis='y',
        labelsize=7, # y轴字体大小设置
        color='black',    # y轴标签颜色设置  
        labelcolor='black', # y轴字体颜色设置
        direction='in' # y轴标签方向设置
    ) 
    plt.tick_params(axis='x',
        labelsize=9, # x轴字体大小设置
        color='black',    # x轴标签颜色设置  
        labelcolor='black', # x轴字体颜色设置
        direction='in' # x轴标签方向设置
    ) 

    plt.legend() # 显示注释
    plt.savefig(save_path+'/pred_gt.svg', bbox_inches='tight', dpi=300)
    plt.savefig(save_path+'/pred_gt.jpg', bbox_inches='tight', dpi=300)
    plt.clf()

def rendering_seq(ecd_infos, org_seq_dict):
    ## 单张图片, 给定峰属性, 渲染出ecd光谱图片
    ## sequence_args: dict, the hyper-parameter for ecd_sequence
    ## ecd_infos: {"gt": {"pos": [5, 10, 17], "height": [0, 1, 0]}, "pred": {"pos": [5, 9, 16], "height": [0, 1, 0]}}
    
    pred_info, gt_info = ecd_infos['pred'], ecd_infos['gt']
    seq_gt_org = org_seq_dict[ecd_infos['smiles']]  # 80-450范围的gt ecd sequence
    sequence_args = get_true_peak_property(gt_info=gt_info, org_info=seq_gt_org)

    seq_pred = reconstruct_ecd2(
        ecd_info = pred_info, 
        sequence_args = sequence_args,)
    return seq_pred

if __name__ == "__main__":
    org_seq_dict = json.load(open("/root/workspace/aigc/ChemGNN/dataset/ECD/smiles_org.json", "r"))

    test1_info = {"gt": {"pos": [1, 3, 4, 6, 8], "height": [0, 0, 0, 0, 0]}, "pred": {"pos": [1, 3, 4, 6, 8], "height": [0, 0, 0, 0, 0]}}
    test2_info = {"smiles": "CC(C)(NC(=O)C[C@@H](CCc1ccccc1)c2ccc(cc2)c3ccccc3)c4ccccn4", "gt": {"pos": [5, 10, 17], "height": [0, 1, 0]}, "pred": {"pos": [5, 9, 17], "height": [0, 1, 0]}}
    
    rendering_ecd_pred_gt(
        ecd_infos = test2_info,
        org_seq_dict = org_seq_dict,
        save_path = '.',
    )
    assert 1 == 0

    root_path = "/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/"
    excellent_info = json.load(open(root_path+"excellent_atom_info.json", "r"))
    good_info = json.load(open(root_path+"good_atom_info.json", "r"))
    # bad_info = json.load(open(root_path+"bad_atom_info.json", "r"))
    worst_info = json.load(open(root_path+"worst_atom_info.json", "r")) 

    for i in range(len(excellent_info)):
        pred_sequence = rendering_seq(
            ecd_infos = excellent_info[i],
            org_seq_dict = org_seq_dict,)
        excellent_info[i]['pred_seq'] = pred_sequence
    for i in range(len(good_info)):
        pred_sequence = rendering_seq(
            ecd_infos = good_info[i],
            org_seq_dict = org_seq_dict,)
        good_info[i]['pred_seq'] = pred_sequence
    # for i in range(len(bad_info)):
    #     pred_sequence = rendering_seq(
    #         ecd_infos = bad_info[i],
    #         org_seq_dict = org_seq_dict,)
    #     bad_info[i]['pred_seq'] = pred_sequence
    for i in range(len(worst_info)):
        pred_sequence = rendering_seq(
            ecd_infos = worst_info[i],
            org_seq_dict = org_seq_dict,)
        worst_info[i]['pred_seq'] = pred_sequence
    
    json.dump(excellent_info, open(root_path+"excellent_atom_info_new.json", "w"))
    json.dump(good_info, open(root_path+"good_atom_info_new.json", "w"))
    # json.dump(bad_info, open(root_path+"bad_atom_info_new.json", "w"))
    json.dump(worst_info, open(root_path+"worst_atom_info_new.json", "w"))