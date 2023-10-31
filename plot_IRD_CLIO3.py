"""
CLIO_pi():
The Intermediate Representation Distribution(IRD) of intermediate representation for CLIO in different scenarios.

CLIO_Pi_ACC():
The impact of each channel in intermediate representation on inference accuracy.
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from models.mobilenetv2_MS_Original import MobileNetV2_MS_Original


def CLIO_pi():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load CLIO without pi
    net_Cave_Nopai = MobileNetV2_MS_Original()
    print('==> loading CLIO without pai model checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        checkpoint = torch.load('checkpoint/CLIO_NoPi_Original.pth')
        net_Cave_Nopai.load_state_dict(checkpoint['net'])
    else:
        checkpoint = torch.load('checkpoint/CLIO_NoPi_Original.pth', map_location=torch.device('cpu'))
        net_Cave_Nopai.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    # Raw_Cave_kw_Nopai = net_Cave_Nopai.state_dict()["layers_feat.11.0.conv1.weight"]
    Raw_Cave_kw_Nopai = net_Cave_Nopai.state_dict()["layers_head.2.conv3.weight"]
    Cave_kw_Nopai = torch.clone(Raw_Cave_kw_Nopai)
    Cave_kw_Nopai = torch.squeeze(Cave_kw_Nopai)
    Cave_L1_Nopai = torch.norm(Cave_kw_Nopai, p=1, dim=1)
    Cave_L1_Nopai = Cave_L1_Nopai.cpu().numpy()
    # normalization
    C_sum_L1_Nopai = np.sum(Cave_L1_Nopai)
    C_max_L1_Nopai = np.max(Cave_L1_Nopai)
    C_L1_nor_Nopai = []
    for z in Cave_L1_Nopai:
        z = float(z / C_sum_L1_Nopai)
        C_L1_nor_Nopai.append(z)

    # CLIO without pai kernel weight combine two and normalization
    Cave_L1_combine_Nopai = []
    for i in range(int((Cave_L1_Nopai.shape[0]) / 2)):
        Cave_L1_combine_Nopai.append(Cave_L1_Nopai[i * 2] + Cave_L1_Nopai[i * 2 + 1])
    Cave_L1_combine_nor_Nopai = []
    Cave_L1_combine_MAX_Nopai = np.max(Cave_L1_combine_Nopai)
    Cave_L1_combine_SUM_Nopai = np.sum(Cave_L1_combine_Nopai)
    for j in Cave_L1_combine_Nopai:
        out = float(j / Cave_L1_combine_MAX_Nopai)
        Cave_L1_combine_nor_Nopai.append(out)

    # load CLIO with concave pi
    net_Cave = MobileNetV2_MS_Original()
    print('==> loading CLIO_concave model checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        checkpoint = torch.load('checkpoint/CLIO_Cave_Original.pth')
        net_Cave.load_state_dict(checkpoint['net'])
    else:
        checkpoint = torch.load('checkpoint/CLIO_Cave_Original.pth', map_location=torch.device('cpu'))
        net_Cave.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    # Raw_Cave_kw = net_Cave.state_dict()["layers_feat.11.0.conv1.weight"]
    Raw_Cave_kw = net_Cave.state_dict()["layers_head.2.conv3.weight"]
    Cave_kw = torch.clone(Raw_Cave_kw)
    Cave_kw = torch.squeeze(Cave_kw)
    Cave_L1 = torch.norm(Cave_kw, p=1, dim=1)
    Cave_L1 = Cave_L1.cpu().numpy()
    C_sum_L1 = np.sum(Cave_L1)
    C_max_L1 = np.max(Cave_L1)
    C_L1_nor = []
    for z in Cave_L1:
        z = float(z / C_sum_L1)
        C_L1_nor.append(z)
    Raw_Cave_pro = [0.2271, 0.1663, 0.1309, 0.1008, 0.0698, 0.0630, 0.0581, 0.0520, 0.0508, 0.0456, 0.0235, 0.0122]
    Cave_pro = []
    for n in Raw_Cave_pro:
        Cave_pro.append(n)
        Cave_pro.append(n)
    Cave_pro = np.array(Cave_pro)
    Cave_pro_sum = np.sum(Cave_pro)
    Cave_pro_max = np.max(Cave_pro)
    Cave_pro_nor = []
    for i in Cave_pro:
        i = float(i / Cave_pro_max)
        Cave_pro_nor.append(i)

    # concave kernel weight combine two and normalization
    C_L1_combine = []
    for i in range(int((Cave_L1.shape[0]) / 2)):
        C_L1_combine.append(Cave_L1[i * 2] + Cave_L1[i * 2 + 1])
    C_L1_combine_nor = []
    C_L1_combine_MAX = np.max(C_L1_combine)
    C_L1_combile_SUM = np.sum(C_L1_combine)
    for j in C_L1_combine:
        out = float(j / C_L1_combine_MAX)
        C_L1_combine_nor.append(out)

    # concave probability normalization
    new_cave_pro_nor = []
    for i in Raw_Cave_pro:
        i = float(i / Cave_pro_max)
        new_cave_pro_nor.append(i)

    # load CLIO with random pi
    net_Cave = MobileNetV2_MS_Original()
    print('==> loading CLIO_RandomPi model checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        checkpoint = torch.load('checkpoint/CLIO_RandomPi_ckpt.pth')
        net_Cave.load_state_dict(checkpoint['net'])
    else:
        checkpoint = torch.load('checkpoint/CLIO_RandomPi_ckpt.pth', map_location=torch.device('cpu'))
        net_Cave.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    # Raw_Cave_kw_RPI = net_Cave.state_dict()["layers_feat.11.0.conv1.weight"]
    Raw_Cave_kw_RPI = net_Cave.state_dict()["layers_head.2.conv3.weight"]
    Cave_kw_RPI = torch.clone(Raw_Cave_kw_RPI)
    Cave_kw_RPI = torch.squeeze(Cave_kw_RPI)
    Cave_L1_RPI = torch.norm(Cave_kw_RPI, p=1, dim=1)
    Cave_L1_RPI = Cave_L1_RPI.cpu().numpy()
    C_sum_L1_RPI = np.sum(Cave_L1_RPI)
    C_max_L1_RPI = np.max(Cave_L1_RPI)
    C_L1_nor_RPI = []
    for t in Cave_L1_RPI:
        t = float(t / C_max_L1_RPI)
        C_L1_nor_RPI.append(t)
    Raw_RPI_pro = [0.08309387, 0.0334896, 0.02711933, 0.06159817, 0.10935165, 0.00345391, 0.13172758, 0.16480117,
                   0.01119798, 0.15537634, 0.02092773, 0.19786267]
    RPI_pro = []
    for n in Raw_RPI_pro:
        RPI_pro.append(n)
        RPI_pro.append(n)
    RPI_pro = np.array(RPI_pro)
    RPI_pro_sum = np.sum(RPI_pro)
    RPI_pro_max = np.max(RPI_pro)
    RPI_pro_nor = []
    for i in RPI_pro:
        i = float(i / RPI_pro_max)
        RPI_pro_nor.append(i)

    # random pi kernel weight combine two and normalization
    RPI_L1_combine = []
    for i in range(int((Cave_L1_RPI.shape[0]) / 2)):
        RPI_L1_combine.append(Cave_L1_RPI[i * 2] + Cave_L1_RPI[i * 2 + 1])
    RPI_L1_combine_nor = []
    RPI_L1_combine_MAX = np.max(RPI_L1_combine)
    RPI_L1_combine_SUM = np.sum(RPI_L1_combine)
    for j in RPI_L1_combine:
        out = float(j / RPI_L1_combine_MAX)
        RPI_L1_combine_nor.append(out)

    # concave probability normalization
    RPI_pro_nor = []
    for i in Raw_RPI_pro:
        i = float(i / RPI_pro_max)
        RPI_pro_nor.append(i)

    # plot
    plt.figure(1)
    # x = np.linspace(0, 12, 12)
    plt.plot(Cave_L1_combine_nor_Nopai, color='red', marker='o', linestyle='--', ms=8, linewidth=3,
             label='Without $\pi$')
    plt.plot(C_L1_combine_nor, color='green', marker='*', linewidth=3, ms=10,
             label='With $\pi$')
    plt.plot(RPI_L1_combine_nor, linestyle='--', marker='d', linewidth=3, ms=8, color='blue',
             label='With random $\pi$')

    plt.tick_params(labelsize=13)

    plt.grid()

    plt.xlabel("Channel", fontsize=14)
    plt.ylabel("Relative Importance", fontsize=14)

    plt.legend()
    plt.show()

def CLIO_Pi_ACC():
    CLIO_NoPi = [12.740000000000002, 22.75, 32.769999999999996, 61.480000000000004, 72.54, 77.12, 80.17999999999999, 85.99,
     88.14999999999999, 89.38000000000001, 89.67, 90.2, 90.36999999999999, 90.81, 90.9, 90.97, 91.17, 91.17, 91.17,
     91.33, 91.36, 91.36999999999999, 91.42, 91.53]
    CLIO_Distribution = [10.84, 33.43, 63.190000000000005, 76.38000000000001, 73.97, 77.89, 81.57, 82.78999999999999, 84.87, 86.77, 87.11,
     87.53, 87.86, 88.1, 88.09, 88.07000000000001, 88.03999999999999, 88.11, 88.26, 88.19, 88.2, 88.35,
     88.27000000000001, 88.29]
    CLIO_RandomPi = [9.76, 17.84, 33.62, 47.13, 56.02, 64.52, 71.78999999999999, 74.22999999999999, 79.32000000000001,
                     82.17, 85.06, 85.91, 86.07000000000001, 86.71, 87.01, 87.33, 87.36, 87.51, 87.58, 87.58, 87.55, 87.53, 87.6, 87.63]

    # plot
    plt.figure(1)
    plt.plot(CLIO_NoPi, color='red', marker='o', linestyle='--', ms=7, linewidth=3, label='Without $\pi$')
    plt.plot(CLIO_Distribution, color='green', marker='*', linewidth=3, ms=10, label='With $\pi$')
    plt.plot(CLIO_RandomPi, linestyle='--',marker='d', linewidth=3, ms=7, color='blue', label='With random $\pi$')

    plt.tick_params(labelsize=13)

    plt.xlabel("Channel", fontsize = 14)
    plt.ylabel("Inference Accuracy", fontsize = 14)

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    CLIO_pi()
    CLIO_Pi_ACC()