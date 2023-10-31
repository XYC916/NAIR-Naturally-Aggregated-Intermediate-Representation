"""
Plot the inference accuracy of distributed deep learning systems with single tail model over the latency domain.
"""

import matplotlib.pyplot as plt
import numpy as np


def GetLatency_Channel():
    speed = 250000  # CLIO 250kbps
    # intermediate data shape [1, 24, 32, 32], float32
    # calculate the latency
    Data_bit = []
    for i in range(24):
        Data_bit.append((1 * (i + 1) * 32 * 32) * 32)
    Data_bit_len = len(Data_bit)
    Latency = []
    for i in range(Data_bit_len):
        Latency.append((Data_bit[i] / speed) * 1000)

    return Latency


def GetLatency_Slice():  # two channels as slice
    speed = 250000  # CLIO 250kbps
    # intermediate data shape [1, 24, 32, 32], float32
    # calculate the latency
    Data_bit = []
    for i in range(12):
        Data_bit.append((1 * ((i + 1) * 2) * 32 * 32) * 32)
    Data_bit_len = len(Data_bit)
    Latency = []
    for i in range(Data_bit_len):
        Latency.append((Data_bit[i] / speed) * 1000)

    return Latency


def Plot(Latency, OM_ACC, CLIO_ACC, CLIO_ACC_1, MS_ACC):
    y_ticks = np.arange(0, 100, 10)
    plt.figure(1)
    plt.yticks(y_ticks)

    plt.plot(Latency[1], OM_ACC[1], marker="X", color='red', markersize=11)
    plt.plot(Latency[1], CLIO_ACC[1], marker="X", color='blue', markersize=11)
    plt.plot(Latency[1], CLIO_ACC_1[1], marker="X", color='green', markersize=11)
    plt.plot(Latency[1], MS_ACC[1], marker="X", color='black', markersize=11)

    plt.plot(Latency, OM_ACC, color='red', marker='^', markersize=8, linewidth=3,
             label='DeepN JPEG')
    plt.plot(Latency, CLIO_ACC_1, color='green', marker='d', markersize=8, linewidth=3,
             label='CLIO larger $\pi_i$')
    plt.plot(Latency, CLIO_ACC, color='blue', marker='*', markersize=11, linewidth=3,
             label='CLIO normal $\pi_i$')
    plt.plot(Latency, MS_ACC, color='black', marker='o', markersize=8, linewidth=3,
             label='NAIR')

    plt.tick_params(labelsize=13)

    plt.axvline(Latency[1], color='red', linestyle='--',label='262ms')

    plt.xlabel("Latency/ms", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)

    plt.grid()
    plt.legend()
    plt.show()

def Plot_Enlarge(Latency, OM_ACC, CLIO_ACC, CLIO_ACC_1, MS_ACC):
    Latency, OM, CLIO, CLIO_1, MS = Latency[8:], OM_ACC[8:], CLIO_ACC[8:], CLIO_ACC_1[8:], MS_ACC[8:]
    plt.figure(1)

    plt.plot(Latency, OM, color='red', marker='^', markersize=11,  linewidth=3,)
    plt.plot(Latency, CLIO_1, color='green', marker='d', markersize=11, linewidth=3)
    plt.plot(Latency, CLIO, color='blue', marker='*', markersize=14, linewidth=3)
    plt.plot(Latency, MS, color='black', marker='o', markersize=11, linewidth=3)

    plt.grid()
    plt.show()


if __name__ == '__main__':
    Slice_Latency = [0, 262.144, 524.288, 786.432, 1048.576, 1310.72, 1572.864, 1835.008, 2097.152, 2359.2960000000003,
                     2621.44, 2883.584, 3145.728]
    Original_acc = [0.1, 16.0, 20.0, 28.999999999999996, 62.0, 77.0, 77.0, 83.0, 89.0, 93.0, 93.0, 93.0, 93.79]
    CLIO_acc = [0.1, 22.75, 61.480000000000004, 77.12, 85.99, 89.38000000000001, 90.2, 90.81, 90.97, 91.17,
                91.33, 91.36999999999999, 91.53]
    CLIO_acc_1 = [0.1, 43.18, 80.51, 84.98, 88.3, 89.18, 89.66, 89.97, 90.06, 90.13, 90.16999999999999, 90.16, 90.29]
    MS_acc = [0.1, 91.49, 93.46, 93.75, 93.76, 93.75, 93.89, 93.88, 93.9, 93.88, 93.95, 94.02, 93.95]
    Plot(Slice_Latency, Original_acc, CLIO_acc, CLIO_acc_1, MS_acc)
