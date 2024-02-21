import math
import json

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import mean_squared_error

def Accuracy(pred, gt):
    # the implimentation of good sample Accuracy. We calculate multi-range accuracy
    # pred: torch.tensor(batch), the tensor of prediction
    # gt:   torch.tensor(batch), the tensor of groundtruth
    pred, gt = pred.tolist(), gt.tolist()
    assert len(pred) == len(gt)

    acc, acc_1, acc_2, acc_3 = 0, 0, 0, 0
    for i in range(len(pred)):
        if pred[i] == gt[i]: 
            acc += 1
        elif abs(pred[i]-gt[i]) == 1:
            acc_1 += 1
        elif abs(pred[i]-gt[i]) == 2:
            acc_2 += 1
        elif abs(pred[i]-gt[i]) == 3:
            acc_3 += 1
        else: continue
    
    return acc, acc_1, acc_2, acc_3

def MAE(pred, gt):
    # the implimentation of Mean Absolute Error
    # MAE == L1 loss
    # pred: torch.tensor(batch, seq_len), the tensor of prediction
    # gt:   torch.tensor(batch, seq_len), the tensor of groundtruth

    pred = torch.tensor(pred, dtype=torch.float32)
    gt = torch.tensor(gt, dtype=torch.float32)
    assert pred.shape == gt.shape
    L1Loss = nn.L1Loss(size_average=True, reduction='mean')
    mae = L1Loss(pred, gt)

    # # L1Loss is equal to below: 
    # R_abs = []
    # batch_size = pred.shape[0]
    # for batch in range(batch_size):
    #     tmp = []
    #     for j in range(len(gt[batch])):
    #         tmp.append(abs(gt[batch][j] - pred[batch][j]))
    #     R_abs.append(sum(tmp) / len(tmp))
    # mae2 = sum(R_abs) / len(R_abs)

    return mae


def MAPE(pred, gt):
    # the implimentation of Mean Absolute Percentage Error
    # pred: torch.tensor(batch, seq_len), the tensor of prediction
    # gt:   torch.tensor(batch, seq_len), the tensor of groundtruth

    pred = torch.tensor(pred, dtype=torch.float32)
    gt = torch.tensor(gt, dtype=torch.float32)
    pred, gt = pred.cpu().numpy(), gt.cpu().numpy()
    mape_loss = np.mean(
        np.abs(np.divide(pred-gt, gt, out=np.zeros_like(gt), where=gt!=0))
    ) * 100
    return mape_loss

def get_sequence_peak(sequence):
    # input- seq: List
    # output- peak_list contains peak position
    peak_list = []
    for i in range(1, len(sequence)-1):
        if sequence[i-1]<sequence[i] and sequence[i]>sequence[i+1]:
            peak_list.append(i)
        if sequence[i-1]>sequence[i] and sequence[i]<sequence[i+1]:
            peak_list.append(i)
    return peak_list

def RMSE_Peak(pred, gt):
    # the implementation of RMSE for Peak Number, Position, also the Peak Symbol acc
    # pred: torch.tensor(batch, seq_len), the tensor of prediction
    # gt:   torch.tensor(batch, seq_len), the tensor of groundtruth
    
    pred = torch.tensor(pred, dtype=torch.float32)
    gt = torch.tensor(gt, dtype=torch.float32)
    pred, gt = pred.cpu().numpy().tolist(), gt.cpu().numpy().tolist()
    batch_size = len(pred)
    
    # calculate RMSE Range
    # range_gt, range_pred = [], []
    # for i in range(batch_size):
    #     range_gt.append(max(gt[i]) - min(gt[i]))
    #     range_pred.append(max(pred[i]) - min(pred[i]))
    # rmse_range = np.sqrt(mean_squared_error(range_gt, range_pred))
    
    # calculate RMSE for Peak number and position
    number_gt, number_pred, position_gt, position_pred = [],[],[],[]
    total_peaks, correct_peak_symbols = 0, 0
    for i in range(batch_size):
        peaks_gt = get_sequence_peak(gt[i])
        peaks_pred = get_sequence_peak(pred[i])
        min_peaks_len,max_peaks_len = min(len(peaks_gt), len(peaks_pred)), max(len(peaks_gt), len(peaks_pred))
        ## peak_height: 
        total_peaks += len(peaks_gt)
        for j in range(min_peaks_len):
            if pred[i][peaks_pred[j]]*gt[i][peaks_gt[j]]>0: correct_peak_symbols += 1
        ## peak number
        number_gt.append(len(peaks_gt))
        number_pred.append(len(peaks_pred))
        ## peak position
        if len(peaks_gt) == min_peaks_len:
            peaks_gt.extend([len(gt[i])] * (max_peaks_len-min_peaks_len) )
        else: peaks_pred.extend([len(pred[i])] * (max_peaks_len-min_peaks_len) )
        position_gt.extend(peaks_gt)
        position_pred.extend(peaks_pred)
        
    rmse_position = np.sqrt(mean_squared_error(position_gt, position_pred))
    rmse_number = np.sqrt(mean_squared_error(number_gt, number_pred))
    symbol_acc = correct_peak_symbols / total_peaks
    return symbol_acc, rmse_position, rmse_number

def Peak_for_draw(pred, gt):
    # the implementation of RMSE for Peak Range, Number, Position
    # return the Peak information
    # pred: torch.tensor(batch, seq_len), the tensor of prediction
    # gt:   torch.tensor(batch, seq_len), the tensor of groundtruth

    pred = torch.tensor(pred, dtype=torch.float32)
    gt = torch.tensor(gt, dtype=torch.float32)
    pred, gt = pred.cpu().numpy().tolist(), gt.cpu().numpy().tolist()
    batch_size = len(pred)
    
    # calculate RMSE Range
    range_gt, range_pred = [], []
    for i in range(batch_size):
        range_gt.append(max(gt[i]) - min(gt[i]))
        range_pred.append(max(pred[i]) - min(pred[i]))
    
    # calculate RMSE for Peak number
    number_gt, number_pred = [], []
    position_gt, position_pred = [], []
    for i in range(batch_size):
        peaks_gt = get_sequence_peak(gt[i])
        peaks_pred = get_sequence_peak(pred[i])
        
        number_gt.append(len(peaks_gt))
        number_pred.append(len(peaks_pred))
        
        min_peaks_len = min(len(peaks_gt), len(peaks_pred))
        max_peaks_len = max(len(peaks_gt), len(peaks_pred))
        
        if len(peaks_gt) == min_peaks_len:
            peaks_gt.extend([len(gt[i])] * (max_peaks_len-min_peaks_len) )
        else:
            peaks_pred.extend([len(pred[i])] * (max_peaks_len-min_peaks_len) )
        position_gt.append(sum(peaks_gt))
        position_pred.append(sum(peaks_pred))
        
    return dict(
        peak_range = (range_gt, range_pred),
        peak_num = (number_gt, number_pred),
        peak_pos = (position_gt, position_pred),
    )