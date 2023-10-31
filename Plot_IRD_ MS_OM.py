"""
Plot the importance distribution of intermediate representation for the original model and NAIR.
"""

import torch
from models.mobilenetv2_IM import MobileNetV2_IM
import matplotlib.pyplot as plt
import numpy as np
from models.mobilenetV2_MS_1Tail import MobileNetV2_Single_Tail
import os


def IRD_MS_OM():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------load original model------------------------
    net_IM = MobileNetV2_IM()
    print('==> loading initial model checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        checkpoint = torch.load('checkpoint/IM_ckpt_lr0.01.pth')
        net_IM.load_state_dict(checkpoint['net'])
    else:
        checkpoint = torch.load('checkpoint/IM_ckpt_lr0.01.pth', map_location=torch.device('cpu'))
        net_IM.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    Raw_IM_kw = net_IM.state_dict()["layers.3.conv1.weight"]
    # calculate the L1 norm
    IM_kw = torch.clone(Raw_IM_kw)
    IM_kw = torch.squeeze(IM_kw)
    IM_L1 = torch.norm(IM_kw, p=1, dim=0)  # get the initial model kernel weight L1 norm
    # combine
    IM_L1_combine = []
    for i in range(int((IM_L1.shape[0]) / 2)):
        IM_L1_combine.append(IM_L1[i * 2] + IM_L1[i * 2 + 1])
    IM_L1_combine_nor = []
    IM_L1_combine_MAX = np.max(IM_L1_combine)
    for j in IM_L1_combine:
        out = float(j / IM_L1_combine_MAX)
        IM_L1_combine_nor.append(out)
    IM_L1_combine_nor = torch.tensor(IM_L1_combine_nor)

    # --------------------------load MS model with weight-------------------------------
    net_MS = MobileNetV2_Single_Tail()
    print('==> loading multi_slices model checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        checkpoint = torch.load('checkpoint/MS_Base_OM_grad_820_1tail.pth')
        net_MS.load_state_dict(checkpoint['net'])
    else:
        checkpoint = torch.load('checkpoint/MS_Base_OM_grad_820_1tail.pth', map_location=torch.device('cpu'))
        net_MS.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    Raw_MS_kw2 = net_MS.state_dict()["layers_head.2.conv3.weight"]
    MS_kw2 = torch.clone(Raw_MS_kw2)
    MS_kw2 = torch.squeeze(MS_kw2)
    MS_L12 = torch.norm(MS_kw2, p=1, dim=1)
    # combine
    MS_L1_combine2 = []
    for i in range(int((MS_L12.shape[0]) / 2)):
        MS_L1_combine2.append(MS_L12[i * 2] + MS_L12[i * 2 + 1])
    MS_L1_combine_nor2 = []
    MS_L1_combine_MAX2 = np.max(MS_L1_combine2)
    for j in MS_L1_combine2:
        out = float(j / MS_L1_combine_MAX2)
        MS_L1_combine_nor2.append(out)
    MS_L1_combine_nor2 = torch.tensor(MS_L1_combine_nor2)

    # -----------------------twin plot---------------------------
    IM_Combine_sort, IM_Combine_index = torch.sort(IM_L1_combine_nor)
    MS_Combine_sort2, MS_Combine_index2 = torch.sort(MS_L1_combine_nor2)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(IM_L1_combine_nor, color='blue', linewidth=3, marker='^', linestyle='--', ms=9, label='OM')
    ax1.plot(MS_L1_combine_nor2, color='blue', linewidth=3, marker='8', ms=8, label='NAIR')
    l1 = ax1.legend(loc=2, bbox_to_anchor=(0, 1.153))

    ax2.plot(IM_Combine_sort, color='red', linewidth=3, marker='^', linestyle='--', ms=9, label='OM Descending')
    ax2.plot(MS_Combine_sort2, color='red', linewidth=3, marker='8', ms=8, label='NAIR Descending')
    ax2.legend(loc=1, bbox_to_anchor=(1, 1.153))

    ax1.tick_params(axis='y', colors='blue', labelsize=13)
    ax2.tick_params(axis='y', colors='red', labelsize=13)
    ax1.tick_params(axis='x', labelsize=13)

    ax1.set_xlabel("Channel", fontsize=14)
    ax1.set_ylabel("Relative Importance", color='blue', fontsize=14)
    ax2.set_ylabel("Descending Order", color='red', fontsize=14)

    plt.grid()
    plt.show()


if __name__ == '__main__':
    IRD_MS_OM()
