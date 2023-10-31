"""
Plot The inference accuracy of the model with various bit error rates.
"""

import matplotlib.pyplot as plt
import numpy as np

BER1 = [92.0, 92.0, 92.0, 92.0, 93.0, 92.0, 90.0, 92.0, 92.0, 91.0, 92.0, 91.0, 91.0, 92.0, 92.0, 92.0, 92.0]
BER10 = [83.0, 92.0, 91.0, 92.0, 94.0, 93.0, 92.0, 90.0, 90.0, 93.0, 91.0, 91.0, 92.0, 93.0, 92.0, 91.0, 93.0]
BER20 = [74.0, 89.0, 90.0, 91.0, 91.0, 91.0, 92.0, 89.0, 90.0, 88.0, 91.0, 89.0, 88.0, 90.0, 93.0, 91.0, 92.0]
BER30 = [56.0, 79.0, 87.0, 89.0, 88.0, 92.0, 85.0, 87.0, 85.0, 86.0, 88.0, 85.0, 87.0, 84.0, 89.0, 90.0, 92.0]
BER40 = [40.0, 70.0, 82.0, 76.0, 81.0, 86.0, 84.0, 80.0, 86.0, 83.0, 85.0, 84.0, 89.0, 76.0, 83.0, 90.0, 91.0]
BER50 = [28.9, 56.0, 75.0, 57.9, 72.0, 81.0, 79.0, 75.0, 79.0, 81.0, 80.0, 82.0, 79.0, 78.0, 83.0, 81.0, 91.0]

x_ticks = np.arange(1, 18, 1)

plt.figure(1)
plt.xticks(x_ticks)
plt.margins(x=0.02)
plt.margins(y=0.02)

plt.plot(x_ticks, BER1, marker='|', label='1%', ms=15, linewidth=2.5)
plt.plot(x_ticks, BER10, marker='o', label='10%', ms=7, linewidth=2.5)
plt.plot(x_ticks, BER20, marker='x', label='20%', ms=7, linewidth=2.5)
plt.plot(x_ticks, BER30, marker='s', label='30%', ms=7, linewidth=2.5)
plt.plot(x_ticks, BER40, marker='d', label='40%', ms=7, linewidth=2.5)
plt.plot(x_ticks, BER50, marker='v', label='50%', ms=7, linewidth=2.5)

plt.tick_params(labelsize=13)

plt.xlabel("Layer Index", fontsize=14)
plt.ylabel("Inference Accuracy", fontsize=14)

plt.grid()
plt.legend()
plt.show()
