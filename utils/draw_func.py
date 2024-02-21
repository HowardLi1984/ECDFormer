# 绘图相关的函数
import random
import json
import collections
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from collections import Counter

def draw_line_chart(lines_info, save_path):
    ## 绘制峰的高度匹配的曲线, 绘制三条, 分别是简单/中等/困难样本的匹配曲线
    ## lines_info: [{'x':[], 'y':[]}, {'x':[], 'y':[]}, {'x':[], 'y':[]}]
    
    fig = plt.figure(figsize=(6, 4), dpi=300)
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.family'] = 'Times New Roman'
    font_size = 20
    marker_size = 12.0
    marker_thickness = 8.0
    linewidth = 3.0

    ax = fig.add_subplot(111)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], fontsize=8)
    ax.set_yticks([0.0, 100.0, 200.0, 300.0, 400.0])
    ax.set_yticklabels([0.0, 100.0, 200.0, 300.0, 400.0], fontsize=8) 
    # ax.set_xlim(0.05, 0.95)
    # ax.set_ylim(5.9, 7.6)

    color_map, marker_shape_map = ['#FFC000','#C00000','#3FA796'],['s','o','D']
    p_list = []
    for i in range(len(lines_info)):
        x, y = lines_info[i]['x'], lines_info[i]['y']
        x_marker = x

        ax.plot(x, y, color=color_map[i], linestyle='-', linewidth=linewidth)
        tmp_p = ax.plot(x_marker, y, marker=marker_shape_map[i], color=color_map[i], markersize=marker_size * 1.0)[0]
        p_list.append(tmp_p)
        ax.plot(x_marker, y, marker=marker_shape_map[i], 
                color='white', markersize=(marker_size - marker_thickness) * 1.0,
                linewidth=0.0, )
    
    # ax.set_ylabel('mAP$_a$', fontsize=font_size + 2, fontweight='bold')
    # ax.set_xlabel('bound $\Delta$', fontsize=font_size + 2, fontweight='bold')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    # leg = plt.legend( # 成泽森版本, 功能更齐全
    #     handles=p_list,
    #     labels=['$Easy$', '$Medium$', '$Hard$'],
    #     labelspacing=0.2,
    #     columnspacing=0.2,
    #     handletextpad=0.1,
    #     prop={
    #         'weight': 'bold',
    #         'size': font_size + 2
    #     },
    #     loc='upper right',
    #     ncol=3,
    # )

    leg = plt.legend(
        handles=p_list,
        labels=['$Easy$', '$Medium$', '$Hard$'],
        loc='upper right',
        labelspacing=1.2, # label之间的行距离
        handlelength=3.5,
    )

    plt.tight_layout(pad=0)
    plt.savefig(save_path+'peak_height.jpg')
    print("success in generating peak_height.jpg")

def draw_scatter_diagram(pred, gt, save_path, scale=None):
    # 绘制峰的 高度和位置的散点图 散点图
    plt.style.use('ggplot') # 设置风格
    # plt.title('The RMSE for groundtruth and prediction')  # 设置标题
    
    # if scale: # 考虑放缩比例
    #     for i in range(len(pred)):
    #         pred[i] = gt[i] + (pred[i] - gt[i])*scale
    
    plt.plot(   # 绘制 y=x 直线 
        [min(pred), max(pred)], [min(pred), max(pred)], '--', color='black')     
    plt.scatter(    # 设置图形数据
        x=pred, y=gt, 
        s = 15,                # s为点的大小, 默认20
        c = 'firebrick',       # c为点的颜色, 默认'r', firebrick, darkblue
        marker = 'o',  # 散点的形状
        alpha = 0.2,           # 透明度设置, 0-1之间, 默认None, 完全不透明
    )    
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.clf()
    print("success in generating RMSE scatter diagram: {}".format(save_path))

def draw_number_scatter(pred, gt, save_path):
    # 绘制 多个点重合在一起的散点图, 为了更好的表示预测的峰数量
    
    x_info, y_info = gt, pred
    assert len(x_info) == len(y_info)
    scatter_points = [(x_info[i], y_info[i]) for i in range(len(x_info))]
    count_info = Counter(scatter_points)
    # for i in range(2, 5): # 补充 y=x 坐标轴的大小
    #     count_info[(i, i)] += 50
    count_list = [(key, count_info[key]) for key in count_info.keys()]
    count_list.sort(key=lambda x:x[1], reverse=True)
    
    point_x = [itm[0][0] for itm in count_list]
    point_y = [itm[0][1] for itm in count_list]
    point_weight = [itm[1]*10 for itm in count_list]

    plt.plot(   # 绘制 y=x 直线 
        [min(pred), max(pred)], [min(pred), max(pred)], 
        '--', linewidth=0.7,
        color='black',
    ) 
    plt.scatter(
        x = point_x, y = point_y, 
        s = point_weight, # s为点的大小
        alpha = 0.4,
        color = "#F78CA2"
    )

    noise_x = np.random.randn(len(x_info))*0.1
    noise_y = np.random.randn(len(y_info))*0.1
    noise_x, noise_y = noise_x.tolist(), noise_y.tolist()
    new_x = [x_info[i]+noise_x[i] for i in range(len(x_info))]
    new_y = [y_info[i]+noise_y[i] for i in range(len(y_info))]
    plt.scatter(
        x = new_x, y = new_y, 
        s = 5, # s为点的大小
        alpha = 0.1,
        color = "#3D0C11"
    )
    ## 加入网格线
    plt.grid(
        linestyle = '--', linewidth = 0.3,
    )

    # plt.axis([0, 9, 0, 9])
    plt.axis('tight') 
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.clf() # 清除当前 figure 的所有axes
    print("success in peak number visualization")

def reconstruct_ecd(ecd_info, width, true_height=5):
    ## 将只有峰信息的序列还原成ECD光谱结构, width是峰周边的
    ## ecd_info = {'pos':[], 'height':[]}
    x_info, y_info = [], []

    # 过滤可能一致的peak_position
    new_pos, new_height = [], []
    for i in range(len(ecd_info['pos'])):
        if ecd_info['pos'][i] not in new_pos:
            new_pos.append(ecd_info['pos'][i])
            if ecd_info['height'][i] == 1: new_height.append(ecd_info['height'][i])
            else: new_height.append(-1)
    for i in range(len(new_pos)):
        pos_token, height_token = new_pos[i], new_height[i]
        ## 模拟顶点左侧的5个点
        tmp_x_left = [(pos_token-width+j*width/5) for j in range(5)]
        tmp_y_left = [true_height*height_token*(1-2**(-1*j)) for j in range(5)]
        ## 模拟顶点右侧的5个点
        tmp_x_right = [(pos_token+j*width/5) for j in range(5)]
        tmp_y_right = [true_height*height_token*(1-2**(-1*j)) for j in range(4,-1,-1)]
        ## 对X轴坐标掺杂噪声, 帮助可视化
        tmp_x = tmp_x_left + tmp_x_right
        tmp_new_x = [x+random.uniform(-0.1, 0.1) for x in tmp_x] # 噪声在-0.1 - 0.1
        ## 录入x_info, y_info中
        x_info.extend(tmp_new_x)
        y_info.extend(tmp_y_left)
        y_info.extend(tmp_y_right)
    return {'x':x_info, 'y':y_info}

def judge_good_case(list1, list2):
    ## 比较输入的两个列表, 比较同一位点相同元素的数量, 超过0.6就是好的
    list_len = min(len(list1), len(list2))
    equal_num = 0
    for i in range(list_len):
        if list1[i] == list2[i]: equal_num += 1
    if equal_num / list_len >= 0.8: return True
    else: return False

def draw_single_case(sequence_info, tgt_path):
    ## 绘制单张图片
    ## sequence_info -> {'pred': pred_sequence, 'gt': gt_sequence, 'id': figure_id}
    plt.style.use('ggplot')
    plt.plot( # 绘制prediction
        sequence_info['pred']['x'], sequence_info['pred']['y'],
        linewidth=1.2, color='red', linestyle='--', label='pred')
    plt.plot( # 绘制groundtruth
        sequence_info['gt']['x'], sequence_info['gt']['y'],
        linewidth=0.8, color='blue', linestyle='-', label='gt')
    plt.legend() # 显示注释
    plt.savefig(tgt_path+str(sequence_info['id'])+'.jpg', bbox_inches='tight', dpi=300)
    plt.clf()

def draw_prediction_and_gt(pred, gt, save_path):
    ## 对prediction和gt进行可视化作图
    # pred, gt 同样结构
    # pred: 一个list, 每个item是一个batch的信息, itm={'pos':[[], [] ...], 'height':[[], [] ...]}
    figure_id = 0
    peak_width = 1  # 单侧峰的绘制宽度
    assert len(pred) == len(gt)
    good_case_list = []
    num_wrong_good_case_list, num_wrong_bad_case_list = [], []

    for batch_id in range(len(pred)):
        pred_pos, pred_height = pred[batch_id]['pos'], pred[batch_id]['height']
        gt_pos, gt_height = gt[batch_id]['pos'], gt[batch_id]['height']
        tmp_list = [len(pred_pos), len(pred_height), len(gt_pos), len(gt_height)]
        assert max(tmp_list) == min(tmp_list) # 比较pred_pos, pred_height, gt的样例数量是否一致

        for case_id in range(len(pred_pos)):
            figure_id += 1
            pred_info = {'pos':pred_pos[case_id], 'height':pred_height[case_id]}
            gt_info = {'pos':gt_pos[case_id], 'height':gt_height[case_id]}
            pred_sequence = reconstruct_ecd(pred_info, width=peak_width, true_height=5)
            gt_sequence = reconstruct_ecd(gt_info, width=peak_width, true_height=5)

            ## 筛出需要可视化的例子: 
                # 1. 预测较好的例子(height匹配率>0.6, 第一个height对上)  
                # 2. 峰的数量对不上的例子
            if min(len(pred_info['height']), len(gt_info['height'])) == 0: continue
            if pred_info['height'][0]==gt_info['height'][0] and judge_good_case(pred_info['height'], gt_info['height']):
                good_case_list.append({'pred': pred_sequence, 'gt': gt_sequence, 'id': figure_id})
            if len(pred_info['pos']) != len(gt_info['pos']):
                if pred_info['height'][0]==gt_info['height'][0] and judge_good_case(pred_info['height'], gt_info['height']):
                    num_wrong_good_case_list.append({'pred': pred_sequence, 'gt': gt_sequence, 'id': figure_id})
                else:
                    num_wrong_bad_case_list.append({'pred': pred_sequence, 'gt': gt_sequence, 'id': figure_id})
    
    print("total number = {}".format(figure_id))
    print("good cases = {}".format(len(good_case_list) / figure_id))
    print("number wrong cases = {}, in which good case is {}, bad case is {}".format(
        (len(num_wrong_good_case_list)+len(num_wrong_bad_case_list))/figure_id, 
        len(num_wrong_good_case_list)/figure_id, len(num_wrong_bad_case_list)/figure_id,
    ))
    for i in tqdm(range(len(good_case_list))):
        draw_single_case(
            sequence_info = good_case_list[i],
            tgt_path = save_path + 'good/',)
    for i in tqdm(range(len(num_wrong_good_case_list))):
        draw_single_case(
            sequence_info = num_wrong_good_case_list[i],
            tgt_path = save_path + 'peak_wrong/good/',)
    for i in tqdm(range(len(num_wrong_bad_case_list))):
        draw_single_case(
            sequence_info = num_wrong_bad_case_list[i],
            tgt_path = save_path + 'peak_wrong/bad/',)

def draw_bin_graph(peaks_info, save_path):
    ## 绘制箱型图来统计位置的误差信息
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    np.random.seed(5)
    # data = np.random.normal(5, 0.5, (100, 5)) # shape 100, 5
    data = [np.array(itm) for itm in peaks_info]
    fig1 = plt.figure()
    sp1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    sp1.boxplot(
        data, 
        notch=True, widths=0.2,
        patch_artist=True, 
        showmeans=False,   # 是否显示平均值的位点
        # boxprops={'color':'black', 'facecolor': 'darkblue'},  # color边框颜色, facecolor填充颜色
        labels=[f'{x}' for x in range(1, len(data)+1)],
    )
    plt.savefig(save_path+"bin_graph.jpg", bbox_inches='tight', dpi=300)
    plt.clf()

def draw_position_scatter(peaks_info, save_path):
    # 绘制 Position差值 的散点图
    # peaks_info: {'1':[], '2':[], ...}
    x_info, y_info = [], []
    for key in peaks_info.keys():
        for itm in peaks_info[key]:
            x_info.append(int(key))
            y_info.append(itm)
    assert len(x_info) == len(y_info)

    scatter_points = [(x_info[i], y_info[i]) for i in range(len(x_info))]
    count_info = Counter(scatter_points)
    # for i in range(2, 5): # 补充 y=x 坐标轴的大小
    #     count_info[(i, i)] += 50
    count_list = [(key, count_info[key]) for key in count_info.keys()]
    count_list.sort(key=lambda x:x[1], reverse=True)
    
    point_x = [itm[0][0] for itm in count_list]
    point_y = [itm[0][1] for itm in count_list]
    point_weight = [itm[1]*1 for itm in count_list]

    plt.plot(   # 绘制 y=x 直线 
        [0, max(x_info)], [0, 0], 
        '--', linewidth=0.5,
        color='black',
    ) 

    noise_x = np.random.randn(len(x_info))*0.1
    noise_y = np.random.randn(len(y_info))*0.1
    noise_x, noise_y = noise_x.tolist(), noise_y.tolist()
    new_x = [x_info[i]+noise_x[i] for i in range(len(x_info))]
    new_y = [y_info[i]+noise_y[i] for i in range(len(y_info))]
    plt.scatter(
        x = new_x, y = new_y, 
        s = 1, # s为点的大小
        alpha = 0.1,
        color = "#3D0C11"
    )

    plt.scatter(
        x = point_x, y = point_y, 
        s = point_weight, # s为点的大小
        alpha = 0.4,
        color = "#F78CA2"
    )
    # 坐标轴范围
    plt.xlim((1, 9))
    plt.ylim((-9, 9))

    ## 绘制网格
    plt.grid(
        linestyle = '--', linewidth = 0.3,
    )

    # plt.axis([0, 9, 0, 9])
    plt.axis('tight') 
    plt.savefig(save_path+'position_match.jpg', bbox_inches='tight', dpi=300)
    plt.clf() # 清除当前 figure 的所有axes
    print("success in peak position visualization")

if __name__ == "__main__":
    ## 测试 绘制peak_range, peak_position的散点图 的能力
    x = [203, 293, 420, 290, 376, 310, 239, 238, 240]
    y = [289, 293, 381, 328, 376, 298, 239, 238, 240]
    test_path = '../test.jpg'
    draw_scatter_diagram(pred=x, gt=y, save_path=test_path)
    assert 1 == 0

    ## 测试 绘制peak_number的散点图 的能力
    peak_info = json.load(open("/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/peak_num_dict.json"))
    draw_number_scatter(
        gt = peak_info['gt'],
        pred = peak_info['pred'],
        save_path = 'peak_number.jpg',
    )

    ## 测试 可视化预测序列和真实序列的对比图 的能力
    # ans_dict = json.load(open("/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/ans_dict.json"))
    # pred, gt = ans_dict['pred'], ans_dict['gt']
    # my_save_path = "/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/case_visualization/"
    # draw_prediction_and_gt(pred, gt, my_save_path)

    ## 测试 绘制peak_height匹配的三条折线构成的折线图
    lines_info = [
        {'x':[0, 1, 2], 'y':[127, 49, 30]},
        {'x':[0, 1, 2, 3, 4], 'y':[330, 127, 81, 56, 17]},
        {'x':[0, 1, 2, 3, 4, 5, 6], 'y':[105, 44, 19, 11, 9, 3, 1]},
    ]
    my_save_path = "/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/analyse_visualization/"
    draw_line_chart(lines_info, my_save_path)

    ## 测试 绘制peak_position的差值散点图
    position_dict = json.load(open("/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/position_dict.json"))
    my_save_path = "/root/workspace/aigc/ChemGNN/fig_lh/ecd/gnn_allthree/analyse_visualization/"
    draw_position_scatter(peaks_info=position_dict, save_path=my_save_path, rmse=2.07)