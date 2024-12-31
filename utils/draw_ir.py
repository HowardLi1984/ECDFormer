from matplotlib import pyplot as plt
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

# 由于ir的光谱预测, height这个元素的预测不是很明显, 我们
# 通过指数拉伸, 让他们区别变大
# 注意, 指数拉伸的factor一定要为偶数
def exponential_stretch(data, factor=10):
    return [x ** factor for x in data]

def find_min_nonzero(lst):
    # 过滤出非零元素
    non_zero_elements = [x for x in lst if x != 0]
    
    # 如果没有非零元素，则返回 None
    if not non_zero_elements:
        return None
    
    # 返回非零元素的最小值
    return min(non_zero_elements)

def draw_ir_spectra(case_gt, case_pred, save_path, fig_name="test_ir"):
    # x_axis = torch.linspace(500, 4000, 3501)
    # yir = Lorenz_broadening(freq, ir_int, c=x_axis, sigma=40).detach().numpy()
    # x = x_axis.detach().numpy()

    plt.rcParams['axes.unicode_minus'] = False  
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in'

    '''draw ir spectrum'''
    plt.figure(figsize=(15,5))
    # plt.rc('font',family='Times New Roman', size=30)

    x_axis = torch.linspace(500, 4000, 3501).detach().numpy()
    yir_gt, yir_pred = [], []
    peak_gt, peak_pred = [], []

    for i in range(len(case_gt['pos'])):
        peak_gt.append(case_gt['pos'][i]*100+500)
    for i in range(len(case_pred['pos'])):
        peak_pred.append(case_pred['pos'][i]*100+500)

    for i in range(len(x_axis)):
        if i in peak_gt: 
            idx = peak_gt.index(i)
            yir_gt.append(case_gt['height'][idx])
        else: yir_gt.append(0)

        if i in peak_pred:
            idx = peak_pred.index(i)
            yir_pred.append(-1*case_pred['height'][idx])
        else: yir_pred.append(0)

    # 指数拉伸
    pred_small = find_min_nonzero(yir_pred)/2
    yir_pred_new = []
    for y in yir_pred:
        if y == 0: yir_pred_new.append(y)
        else:
            yir_pred_new.append(y-pred_small)
    assert min(yir_pred_new) < 0
    yir_pred = exponential_stretch(yir_pred_new)
    yir_gt, yir_pred = np.array(yir_gt), np.array(yir_pred)
    plt.plot(x_axis, yir_gt/yir_gt.max(), lw=2, label='IR', color='r')
    plt.plot(x_axis, (-1)*yir_pred/yir_pred.max(), lw=2, label='IR', color='b')

    ax = plt.gca()
    # 将横坐标轴移到中间
    ax.spines['bottom'].set_position(('data', 0))  # y=0 处设置 x 轴
    ax.spines['left'].set_position(('data', 500))    # x=0 处设置 y 轴

    # 隐藏顶部和右侧的轴
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # 移动刻度标签
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


    plt.xlim(500,4000)
    plt.tick_params(labelsize=10, width=1, length=1)
    
    file_jpg_path = f"{save_path}{fig_name}.jpg"
    file_svg_path = f"{save_path}{fig_name}.svg"
    
    plt.savefig(file_jpg_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(file_svg_path, bbox_inches='tight', pad_inches=0.1)

def test_data():
    case_gt = dict(
        pos=[0, 2, 4, 5, 6, 8, 11, 17, 24, 29],
        height=[0.28920692205429077, 1.7623722553253174, 1.6905590295791626, 0.25644153356552124, 0.44524678587913513, 1.026818871498108, 0.5243917107582092, 0.11723333597183228, 0.4352776110172272, 0.11473436653614044],
    )
    case_pred = dict(
        pos=[0, 2, 4, 6, 7, 9, 11, 24, 29],
        height=[1.3426212072372437, 1.3246160745620728, 1.3502660989761353, 1.3235712051391602, 1.3707696199417114, 1.282312035560608, 1.3365107774734497, 1.2955917119979858, 1.3092421293258667],
    )
    return case_gt, case_pred

if __name__ == "__main__":
    case_gt, case_pred = test_data()
    draw_ir_spectra(case_gt, case_pred)
