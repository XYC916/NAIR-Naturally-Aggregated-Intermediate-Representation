"""
Plot the inference accuracy of distributed deep learning systems with multiple tail models over the latency domain.
"""

import matplotlib.pyplot as plt

def Plot_Multi_Tails_ACC(Latency, CLIO_BA, CLIO_IC, MS, JPEG):
    index = len(Latency) - len(JPEG)
    JPEG_Latency = Latency[index:len(Latency)]

    plt.plot(JPEG_Latency[0], JPEG[0], marker="X", color='red', markersize=11)
    plt.plot(Latency[0], CLIO_BA[0], marker="X", color='blue', markersize=11)
    plt.plot(Latency[0], CLIO_IC[0], marker="X", color='green', markersize=11)
    plt.plot(Latency[0], MS[0], marker="X", color='black', markersize=11)

    plt.plot(JPEG_Latency, JPEG, color='red', marker='^', markersize=8, linewidth=3,
             label='DeepN JPEG')
    plt.plot(Latency, CLIO_IC, color='green', marker='d', markersize=8, linewidth=3,
             label='CLIO larger $\pi_i$')
    plt.plot(Latency, CLIO_BA, color='blue', marker='*', markersize=11, linewidth=3,
             label='CLIO normal $\pi_i$')
    plt.plot(Latency, MS, color='black', marker='o', markersize=8, linewidth=3,
             label='NAIR')

    plt.xticks()
    plt.tick_params(labelsize=13)
    plt.axvline(Latency[0], color='red', linestyle='--',label='262ms')

    plt.xlabel("Latency/ms", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Slice_Latency = [262.144, 524.288, 786.432, 1048.576, 1310.72, 1572.864, 1835.008, 2097.152, 2359.2960000000003,
                     2621.44, 2883.584, 3145.728]
    CLIO_Best_ACC = [81.11, 87.729, 89.62, 90.34, 90.74, 90.91, 91.06, 91.14, 91.32, 91.37, 91.52, 91.53]
    CLIO_Information_Centralized = [83.54, 87.93, 89.12, 89.67, 89.51, 89.77, 89.81, 90.1, 89.8, 90.11, 90.48, 90.29]
    MS = [91.49, 93.46, 93.75, 93.76, 93.75, 93.89, 93.88, 93.9, 93.88, 93.95, 94.02, 93.95]
    JPEG = [62.0, 77.0, 77.0, 83.0, 89.0, 93.0, 93.0, 93.0, 93.79]

    Plot_Multi_Tails_ACC(Slice_Latency, CLIO_Best_ACC, CLIO_Information_Centralized, MS, JPEG)