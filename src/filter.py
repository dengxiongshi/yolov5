import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from obspy import read
from scipy.signal import savgol_filter as sgolay
from scipy import signal

file_path = r"C:\Users\dengxs\Desktop\save_gyro.csv"

dataset = np.genfromtxt(file_path, delimiter=',')
dataset = dataset[1:, :]

# data = pd.read_csv(file_path, delimiter=',')
# tt = dataset[1:, :]

tr = dataset[:, 3]  # X轴数据


# 原始信号 s1, 及对应的时间t
s1 = tr
t = np.arange(len(s1)) * 0.001

# 相同权重卷积平均
s2 = np.convolve(s1, np.ones(50) / 50, mode='same')

# Savitsky-Golay 平滑滤波
s3 = sgolay(s1, 91, 3)

# 低通滤波
fs = 4000  # 采样率 (赫兹)
fc = 2  # 截止频率 （赫兹）
b, a = signal.butter(4, 2.0 * fc / fs, 'lowpass')
s4 = signal.filtfilt(b, a, s1)

# 高通滤波
fs = 1 / (t[1] - t[0])  # 采样率 (赫兹)
fc = 1  # 截止频率 （赫兹）
b, a = signal.butter(4, 2.0 * fc / fs, 'highpass')
s5 = signal.filtfilt(b, a, s1)

# 带通滤波
fs = 1 / (t[1] - t[0])  # 采样率 (赫兹)
f1 = .1;
f2 = 2  # 通带的两个截止频率 （赫兹）
b, a = signal.butter(4, [2.0 * f1 / fs, 2.0 * f2 / fs], 'bandpass')
s6 = signal.filtfilt(b, a, s1)

# # 带阻滤波
fs = 1 / (t[1] - t[0])  # 采样率 (赫兹)
f1 = .5;
f2 = 5  # 阻带的两个截止频率
b, a = signal.butter(4, [2.0 * f1 / fs, 2.0 * f2 / fs], 'bandstop')
s7 = signal.filtfilt(b, a, s1)

# 画图对比
l = 6
# plt.title("X")
plt.figure(figsize=(15, 20))
plt.subplot(l, 1, 1)
plt.plot(t, s1, label='Original signal')
plt.plot(t, s2, label='Convolved&filtered')
plt.legend(fontsize=15)

plt.subplot(l, 1, 2)
plt.plot(t, s1, label='Original signal')
plt.plot(t, s3, label='Savitzky-Golay&filtered')
plt.legend(fontsize=15)

plt.subplot(l, 1, 3)
plt.plot(t, s1, label='Original signal')
plt.plot(t, s4, label='Lowpass filtered')
plt.legend(fontsize=15)

plt.subplot(l, 1, 4)
plt.plot(t, s1, label='Original signal')
plt.plot(t, s5, label='Highpass filtered')
plt.legend(fontsize=15)

plt.subplot(l, 1, 5)
plt.plot(t, s1, label='Original signal')
plt.plot(t, s6, label='Bandpass filtered')
plt.legend(fontsize=15)

plt.subplot(l, 1, 6)
plt.plot(t, s1, label='Original signal')
plt.plot(t, s7, label='Bandstop filtered')
plt.legend(fontsize=15)

plt.show()
