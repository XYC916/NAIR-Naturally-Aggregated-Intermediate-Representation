"""
Testing code for NAIR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time

from models.mobilenetV2_MS_1Tail import MobileNetV2_Single_Tail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 testing')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')

    net = MobileNetV2_Single_Tail()

    net = net.to(device)

    print('==> loading from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        checkpoint = torch.load('checkpoint/MS_Base_OM_grad_820_1tail.pth')
        net.load_state_dict(checkpoint['net'])
    else:
        checkpoint = torch.load('checkpoint/MS_Base_OM_grad_820_1tail.pth', map_location=torch.device('cpu'))
        net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    flag = 0
    count = 0
    start = time.clock()

    test_loss_8 = 0
    correct_8 = 0
    total_8 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_8, outputs = net(inputs, flag)

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

            end = time.clock()
            print("batch{} required for operation {}s".format((batch_idx + 1), (end - start)))
            start = end

            count += 1
            if count % 100 == 0:
                break

    # Save checkpoint.
    acc = 100. * correct / total
    acc_8 = 100. * correct_8 / total_8
    print("The accuracy on the overall test set is：{}%, The accuracy of transmitting only channel_head is：{}%".format(acc, acc_8))
