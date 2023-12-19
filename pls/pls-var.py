'''
0. import package
0. 引入包
'''
from sys import stdout

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from scipy.signal import savgol_filter # 文档地址：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
import scipy.io as scio
 
from sklearn.cross_decomposition import PLSRegression # 文档地址：https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html?highlight=plsregression
from sklearn.model_selection import cross_val_predict # 文档地址：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html?highlight=cross_val_predict
# 文档地址（mean_squared_error）：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared_error#sklearn.metrics.mean_squared_error
# 文档地址（r2_score）：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html?highlight=r2_score#sklearn.metrics.r2_score
from sklearn.metrics import mean_squared_error, r2_score 
#plt.rcParams['font.sans-serif']=['Microsoft YaHei']
#plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams['font.size'] = 12 
config = {
"font.family": 'serif', # 衬线字体
"font.size": 12, # 相当于小四大小
"font.serif": ['SimSun'], # 宋体
"mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
'axes.unicode_minus': False # 处理负号，即-号
}
plt.rcParams.update(config)

import logging

'''
1. import data
1. 引入数据
'''

# Read data 读取数据
data = pd.read_csv('./co-30条参考光谱.csv')
data_obj = pd.read_csv('./merged_index.csv')
X = data.values[:,1:1411] # 吸光度
y = data['concentration']# 浓度

# Define wavelength range
wl = np.arange(206.727,779.841,(779.841-206.727)/1410) # 波长

# Calculate derivatives
X1 = savgol_filter(X, 25, polyorder = 5, deriv=1) # S-G滤波；deriv求导阶数；polyorder多项式阶数；11 窗口长度；X输入数据
# X2 = savgol_filter(X, 13, polyorder = 2,deriv=2)

'''
pls
'''
# Define the PLS regression object 定义pls对象
pls = PLSRegression(n_components=8)
# Fit data 拟合数据
pls.fit(X1, y)

'''
previous method
# 简单pls方法
'''

# Plot spectra 绘制光谱
plt.figure(figsize=(12,7))
#with plt.style.context(('seaborn-ticks')):
plt.plot(wl, X1.T)
plt.xlabel('波长（nm）')
plt.ylabel('吸光度') # 一阶导数吸收光谱
plt.savefig('plsvar_1.png', dpi=500, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12,7))
plt.plot([i for i in range(1,1410+1)], np.abs(pls.coef_[:,0])) # 每个变量的pls回归系数的绝对值
plt.xlabel('波长变量')
plt.ylabel('回归系数绝对值') # PLS系数绝对值
plt.savefig('plsvar_5.png', dpi=500, bbox_inches="tight")
plt.show()

'''
opt
'''
# Get the list of indices that sorts the PLS coefficients in ascending order 得到对PLS系数绝对值进行升序排序的指数列表
# of the absolute value
sorted_ind = np.argsort(np.abs(pls.coef_[:,0])) # argsort 表示对数据进行从小到大进行排序，返回数据的索引值。
 
# Sort spectra according to ascending absolute value of PLS coefficients 根据PLS回归系数绝对值的上升排序，回归系数越大表示该变量对结果的影响权重越大
Xc = X1[:,sorted_ind]

def pls_variable_selection(X, y, max_comp):
    
    # Define MSE array to be populated 定义要填充的均方误差数组
    mse = np.zeros((max_comp,X.shape[1])) # 生成0矩阵；max_comp行，X的列数（变量数）列；每个格子（i，j）代表 i个组件下，剔除j个变量后的mse值，最后选取最优
 
    # Loop over the number of PLS components 遍历PLS组件的数量
    for i in range(max_comp): # i为组件数
        
        # Regression with specified number of components, using full spectrum 使用全光谱变量进行具有指定数量的组件的pls回归
        pls1 = PLSRegression(n_components=i+1) # i从开始循环所以是i+1
        pls1.fit(X, y) # 原始X光谱拟合模型 pls1
        
        # Indices of sort spectra according to ascending absolute value of PLS coefficients 根据PLS系数绝对值递增排序的光谱指标，系数大小决定了相关度
        sorted_ind = np.argsort(np.abs(pls1.coef_[:,0]))
 
        # Sort spectra accordingly 将索引转为数组值
        Xc = X[:,sorted_ind]
 
        # Discard one wavelength at a time of the sorted spectra, 在已排序的光谱变量中，一次丢弃一个波长，回归并计算MSE交叉验证一个波长在一个时间的排序光谱
        # regress, and calculate the MSE cross-validation
        for j in range(Xc.shape[1]-(i+1)): # j为目前现存的变量数（每次剔除一个系数最小的）
 
            pls2 = PLSRegression(n_components=i+1)
            pls2.fit(Xc[:, j:], y) # 剔除系数最小的一个变量后的光谱拟合 pls2
            
            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=5) # 交叉验证
 
            mse[i,j] = mean_squared_error(y, y_cv) # 记录mse
    
        comp = 100*(i+1)/(max_comp) # 完成进度
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
 
    # Calculate and print the position of minimum in MSE 计算并输出最小均方误差的位置
    mseminx,mseminy = np.where(mse==np.min(mse[np.nonzero(mse)]))
 
    print("Optimised number of PLS components: ", mseminx[0]+1) # 最佳的组件数量
    print("Wavelengths to be discarded ",mseminy[0]) # 多少个变量被抛弃时得到最小mse结果
    print('Optimised MSEP ', mse[mseminx,mseminy][0]) # 最小mse的值
    stdout.write("\n")
    # plt.imshow(mse, interpolation=None)
    # plt.show()
 
 
    # Calculate PLS with optimal components and export values 计算最优pls组件数下的输出值
    pls = PLSRegression(n_components=mseminx[0]+1)
    pls.fit(X, y)
        
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))
 
    Xc = X[:,sorted_ind]
 
    return(Xc[:,mseminy[0]:],mseminx[0]+1,mseminy[0], sorted_ind)


'''

'''
def simple_pls_cv(X, y, n_comp):
 
    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, y)
    y_c = pls.predict(X) # 预测
    print('real value: {}'.format(y))
    print('predict value: {}'.format(y_c))
 
    # Cross-validation 交叉验证
    y_cv = cross_val_predict(pls, X, y, cv=10)    
 
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
 
    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
 
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
 
    # Plot regression 
 
    z = np.polyfit(y, y_cv, 1)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(y_cv, y, c='red', edgecolors='k', label='样本散点')
    ax.plot(z[1]+z[0]*y, y, c='blue', linewidth=1, label='回归曲线')
    ax.plot(y, y, color='green', linewidth=1, label='y=x')
    plt.legend(loc="lower right")
    plt.xlabel('预测钴离子浓度（mg/L）')
    plt.ylabel('真实钴离子浓度（mg/L）') # PLS系数绝对值
    plt.savefig('plsvar_3.png', dpi=500, bbox_inches="tight")
    plt.show()

'''
'''
opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X1, y, 15) # 
simple_pls_cv(opt_Xc, y, ncomp)

'''
'''
ix = np.in1d(wl.ravel(), wl[sorted_ind][:wav])
 
import matplotlib.collections as collections
 
# Plot spectra with superimpose selected bands 绘制选定波段光谱
fig, ax = plt.subplots(figsize=(12,7))
ax.plot(wl, X1.T)
plt.ylabel('吸光度')
plt.xlabel('波长 (nm)')
collection = collections.BrokenBarHCollection.span_where(
    wl, ymin=-1, ymax=1, where=ix == False, facecolor='blue', alpha=0.3)
ax.add_collection(collection)
plt.savefig('plsvar_4.png', dpi=500, bbox_inches="tight")
plt.show()