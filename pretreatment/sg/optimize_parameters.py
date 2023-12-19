import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
config = {
"font.family": 'serif', # 衬线字体
"font.size": 12, # 相当于小四大小
"font.serif": ['SimSun'], # 宋体
"mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
'axes.unicode_minus': False # 处理负号，即-号
}
plt.rcParams.update(config)
from scipy.signal import savgol_filter, general_gaussian
 
data = pd.read_csv('./1.csv')
X = data.values[0]
wl = np.arange(206.727,1049.154,(1049.154-206.727)/2048)

# 1、
# Calculate the power spectrum 
ps = np.abs(np.fft.fftshift(np.fft.fft(X)))**2

# Define pixel in original signal and Fourier Transform
pix = np.arange(X.shape[0])
fpix = np.arange(ps.shape[0]) - ps.shape[0]//2

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

axes[0].plot(wl[:1410], X[:1410])
axes[0].set_xlabel('波长（nm）')
axes[0].set_ylabel('吸光度')

axes[1].semilogy(fpix, ps, 'b')
axes[1].set_xlabel('频率（Hz）')
axes[1].set_ylabel('功率谱密度（W/Hz）')
plt.savefig('1.png', dpi=500, bbox_inches="tight")
plt.show()

# 3、
# Set some reasonable parameters to start wit
# Calculate three different smoothed spectra
X_smooth_1 = savgol_filter(X, 10, polyorder = 2, deriv=0)
X_smooth_2 = savgol_filter(X, 15, polyorder = 3, deriv=0)
X_smooth_3 = savgol_filter(X, 20, polyorder = 4, deriv=0)
X_smooth_4 = savgol_filter(X, 25, polyorder = 5, deriv=0)
X_smooth_5 = savgol_filter(X, 30, polyorder = 6, deriv=0)
 
# Calculate the power spectra in a featureless region
ps = np.abs(np.fft.fftshift(np.fft.fft(X[600:750])))**2
ps_1 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_1[600:750])))**2
ps_2 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_2[600:750])))**2
ps_3 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_3[600:750])))**2
ps_4 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_4[600:750])))**2
ps_5 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_5[600:750])))**2
 
# Define pixel in Fourier space
fpix = np.arange(ps.shape[0]) - ps.shape[0]//2

plt.figure(figsize=(12,7))
plt.semilogy(fpix, ps, 'b', label = 'No smoothing')
plt.semilogy(fpix, ps_1, 'r', label = 'Smoothing: w=10,p=2')
plt.semilogy(fpix, ps_2, 'g', label = 'Smoothing: w=15,p=3')
plt.semilogy(fpix, ps_3, 'c', label = 'Smoothing: w=20,p=4')
plt.semilogy(fpix, ps_4, 'm', label = 'Smoothing: w=25,p=5')
plt.semilogy(fpix, ps_5, 'y', label = 'Smoothing: w=30,p=6')
plt.legend()
plt.ylabel('功率谱密度（W/Hz）')
plt.xlabel('频率（Hz）')
plt.savefig('2.png', dpi=500, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12,7))
plt.plot(wl[:1410], X[:1410], 'r', label='原始光谱')
# loss
plt.plot(wl[:1410], X_smooth_4[:1410], 'g', label='S-G滤波后的光谱')
plt.xlabel('波长（nm）')
plt.ylabel('吸光度')
plt.legend(loc="upper right")
plt.savefig("3.png", dpi=500, bbox_inches="tight")
plt.show()