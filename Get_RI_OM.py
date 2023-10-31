"""
checkpoint：IM_ckpt.pth & IM_ckpt_lr0.01.pth
Get the relative importance from original model and load it in Plot_IRD_OM_hist.m
"""

import torch
from models.mobilenetv2_IM import MobileNetV2_IM
import os

def Get_RI():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------load original model 1------------------------
    net_IM = MobileNetV2_IM()
    print('==> loading initial model_1 checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        net_IM = torch.load('checkpoint/IM_ckpt.pth')['net']
    else:
        checkpoint = torch.load('checkpoint/IM_ckpt_lr0.01.pth', map_location=torch.device('cpu'))
        net_IM.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    Raw_IM_kw = net_IM.state_dict()["layers.3.conv1.weight"]
    # calculate the L1 norm
    IM_kw = torch.clone(Raw_IM_kw)
    IM_kw = torch.squeeze(IM_kw)
    IM_L1 = torch.norm(IM_kw, p=1, dim=0)  # get the initial model kernel weight L1 norm

    IM_L1_Select1 = []
    for i in range(len(IM_L1)):
        if i % 2 == 0:
            IM_L1_Select1.append(float(IM_L1[i]))

    # ---------------load original model 2------------------------
    print('==> loading initial model_2 checkpoint..')
    checkpoint = torch.load('checkpoint/IM_ckpt_lr0.01.pth')
    net_IM.load_state_dict(checkpoint['net'])

    Raw_IM_kw1 = net_IM.state_dict()["layers.3.conv1.weight"]
    # calculate the L1 norm
    IM_kw1 = torch.clone(Raw_IM_kw1)
    IM_kw1 = torch.squeeze(IM_kw1)
    IM_L11 = torch.norm(IM_kw1, p=1, dim=0)  # get the initial model kernel weight L1 norm

    IM_L1_Select2 = []
    for i in range(len(IM_L11)):
        if i % 2 == 0:
            IM_L1_Select2.append(float(IM_L11[i]))

    print('Relative importance of model1: {}'.format(IM_L1_Select1))
    print('Relative importance of model2: {}'.format(IM_L1_Select2))


if __name__ == '__main__':
    Get_RI()