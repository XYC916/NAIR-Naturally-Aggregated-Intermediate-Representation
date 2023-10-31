"""
Training code for NAIR
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from models.mobilenetV2_MS_1Tail import MobileNetV2_Single_Tail

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_8 = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=80, shuffle=True, num_workers=0)  # num_workers=2

testset = torchvision.datasets.CIFAR10(
    root='data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)  # num_workers=2

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

MobileNetV2 = MobileNetV2_Single_Tail()

MobileNetV2 = MobileNetV2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(MobileNetV2.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# print(optimizer.param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def Load_OM_Initial_low_weights():
    # load weights from original model
    original_model_ckpt_path = 'checkpoint/IM_ckpt_lr0.01.pth'
    OM_checkpoint = torch.load(original_model_ckpt_path)['net']
    # OM_checkpoint = OM.state_dict()
    print('Loading the trained original model from {}'.format(original_model_ckpt_path))
    model_dict = MobileNetV2.state_dict()
    update_dict = dict()

    for MS_name, MS_weights in model_dict.items():
        MS_name_temp = MS_name
        if 'layers_head' in MS_name:
            MS_name_split = MS_name.split('.')
            MS_name_split[0] = 'layers'
            MS_name_temp = '.'.join(MS_name_split)
        if 'layers_tail' in MS_name:
            MS_name_split = MS_name.split('.')
            MS_name_split[0] = 'layers'
            MS_name_split[1] = str(int(MS_name_split[1]) + 4)
            MS_name_temp = '.'.join(MS_name_split)
        if MS_name_temp in OM_checkpoint:
            update_dict.update({MS_name: OM_checkpoint[MS_name_temp]})

    model_dict.update(update_dict)
    MobileNetV2.load_state_dict(model_dict)
    for weights in MobileNetV2.parameters():
        weights.requires_grad = True


def ScaleFactor_Base_L1_norm():
    OM_checkpoint = torch.load('checkpoint/IM_ckpt_lr0.01.pth')['net']
    Raw_IM_kw = OM_checkpoint["layers.3.conv1.weight"]
    # calculate the L1 norm
    IM_kw = torch.clone(Raw_IM_kw)
    IM_kw = torch.squeeze(IM_kw)
    L1 = torch.norm(IM_kw, p=1, dim=0)  # get the initial model kernel weight L1 norm
    L1_combine = []
    for i in range(int(L1.shape[0] / 2)):
        L1_combine.append(L1[i * 2] + L1[i * 2 + 1])
    # L1_combine = L1_combine.numpy()
    L1_max = np.max(L1_combine)
    L1_nor = []
    for i in L1_combine:
        temp = float(i / L1_max)
        L1_nor.append(temp ** 3)
        L1_nor.append(temp ** 3)
    L1_nor = torch.tensor(L1_nor)

    return L1_nor

def L1_Norm_Penalty():
    kw = MobileNetV2.state_dict()['layers_head.2.conv3.weight']
    raw_kw = torch.squeeze(kw)
    L1 = torch.norm(raw_kw, p=1, dim=0)
    L1_combine = []
    for i in range(int(L1.shape[0] / 2)):
        L1_combine.append(L1[i * 2] + L1[i * 2 + 1])
    low_weights_index = [1, 4, 9, 5, 7, 6, 11, 0, 3, 2]
    Penalty = 0
    for i in low_weights_index:
        Penalty = Penalty + L1_combine[i]
    Penalty = Penalty*0.01

    return Penalty

def Weight_Base_L1_Norm():
    OM_checkpoint = torch.load('checkpoint/IM_ckpt_lr0.01.pth')['net']
    Raw_IM_kw = OM_checkpoint["layers.3.conv1.weight"]
    # calculate the L1 norm
    IM_kw = torch.clone(Raw_IM_kw)
    IM_kw = torch.squeeze(IM_kw)
    L1 = torch.norm(IM_kw, p=1, dim=0)  # get the initial model kernel weight L1 norm
    L1_combine = []
    for i in range(int(L1.shape[0] / 2)):
        L1_combine.append(L1[i * 2] + L1[i * 2 + 1])
    L1_sort = torch.tensor(L1_combine)
    L1_sort, index = torch.sort(L1_sort, descending=True)
    L1_sum = [L1_sort[0]]
    temp = L1_sort[0]
    for i in range(len(L1_sort) - 1):
        temp = temp + L1_sort[i+1]
        L1_sum.append(temp)

    L1_max = np.max(L1_sum)
    L1_nor = []
    for i in L1_sum:
        temp = float(i / L1_max)
        L1_nor.append(temp ** 3)

    L1_weight = []
    for i in L1_nor:
        L1_weight.append(1-i)
    L1_weight[len(L1_weight)-1] = 1
    L1_weight = torch.tensor(L1_weight)

    return L1_weight

# Training
def train(epoch, l1_weight):
    MobileNetV2.train()
    total_train_step = 0
    train_loss = 0
    flag = 1
    loss = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = MobileNetV2(inputs, flag)

        for i in range(len(out)):
            loss.append(criterion(out[i], targets))

        penalty = L1_Norm_Penalty()

        overall_loss = l1_weight[0]*loss[0] + l1_weight[1]*loss[1] + l1_weight[2]*loss[2] + l1_weight[3]*loss[3] +\
                       l1_weight[4]*loss[4] + l1_weight[5]*loss[5] + l1_weight[6]*loss[6] + l1_weight[7]*loss[7] +\
                       l1_weight[8]*loss[8] + l1_weight[9]*loss[9] + l1_weight[10]*loss[10] + l1_weight[11]*loss[11]+\
                       penalty

        train_loss += overall_loss
        optimizer.zero_grad()  # clear all grad

        overall_loss.backward()  # calculate grad

        optimizer.step()  # update weights

        loss.clear()  # clear the loss list

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("epoch{} Number of trainings：{}, Loss：{}".format((epoch + 1), total_train_step, train_loss.item()))
            train_loss = 0

def test(epoch):
    global best_acc, best_acc_8
    MobileNetV2.eval()
    test_loss = 0
    correct = 0
    total = 0
    flag = 0

    test_loss_8 = 0
    correct_8 = 0
    total_8 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_8, outputs = MobileNetV2(inputs, flag)

            # channel 8
            loss_8 = criterion(outputs_8, targets)
            test_loss_8 += loss_8
            _, predicted_8 = outputs_8.max(1)
            total_8 += targets.size(0)
            correct_8 += predicted_8.eq(targets).sum().item()

            # all channel
            loss = criterion(outputs, targets)
            test_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    acc_8 = 100. * correct_8 / total_8
    print("epoch{} ACC：{}%， Loss：{}".format(epoch + 1, acc, test_loss.item()))
    print("epoch{} channel_8 ACC：{}%， Loss：{}".format(epoch + 1, acc_8, test_loss_8.item()))

    # Save the model with the best overall accuracy
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': MobileNetV2.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/MS_Base_OM_grad_94_1tail.pth')
        best_acc = acc
        print("Best Acc：{}%".format(best_acc))
    # Save the model with the best channel_8 accuracy
    if acc_8 > best_acc_8:
        print('Saving..')
        state = {
            'net': MobileNetV2.state_dict(),
            'acc': acc_8,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/MS_Base_OM_grad_head_94_1tail.pth')
        best_acc_8 = acc_8
        print("Best Acc：{}%".format(best_acc_8))

    return test_loss


start_epoch = 0
end_epoch = 200

Load_OM_Initial_low_weights()
L1_weight = Weight_Base_L1_Norm()

# load the model with the best overall accuracy
if os.path.exists('checkpoint/MS_Base_OM_grad_94_1tail.pth'):
    checkpoint = torch.load('checkpoint/MS_Base_OM_grad_94_1tail.pth')
    MobileNetV2.load_state_dict(checkpoint['net'])
    start_epoch = (checkpoint['epoch'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = (checkpoint['acc'])
    print('Load epoch {} success!'.format(start_epoch))
else:
    epoch = 0
    print('No save model, will be trained from scratch!')

# load the model with the best channel_8 accuracy
if os.path.exists('checkpoint_82/MS_Base_OM_grad_head_94_1tail.pth'):
    checkpoint = torch.load('checkpoint/MS_Base_OM_grad_head_94_1tail.pth')
    best_acc_8 = (checkpoint['acc'])
    print('Load epoch {} success！'.format(start_epoch))
else:
    epoch = 0
    print('No save model, will be trained from scratch!！')

for epoch in range(start_epoch, end_epoch):
    start = time.clock()
    print("--------------The {}th of training have begun.--------------".format(epoch + 1))
    train(epoch, L1_weight)
    loss = test(epoch)
    scheduler.step()
    end = time.clock()
    if (end - start) < 60:
        print("The {}th training round takes {} seconds to run".format((epoch + 1), (end - start)))
    else:
        min = int((end - start) / 60)
        sec = (end - start) % 60
        print("The {} minute {} second required for the {} round of training runs".format((epoch + 1), min, sec))
