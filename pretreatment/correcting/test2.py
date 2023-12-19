def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''
 
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
 
    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference

    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
 
    return (data_msc, ref)

def snv(input_data):

    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data

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
 
# import data and define wavelengths
data = pd.read_csv('./merged_smoothed.csv')
all = []
for i in range(880):
    spec = data.values[i][:2048]
    spec.tolist()
    all.append(spec)
all = np.array(all)
X = all
wl = np.arange(206.727,779.841,(779.841-206.727)/1410) # 波长

# Xmsc = msc(X)[0] # Take the first element of the output tuple
Xmsc = msc(X)
Xsnv = snv(X)

plt.figure(figsize=(12,7))
plt.plot(wl, X.T[:1410])
plt.xlabel('波长（nm）')
plt.ylabel('吸光度')
plt.savefig('./correcting_5.png', dpi=500, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(wl,Xmsc[0].T[:1410])
plt.xlabel('波长（nm）')
plt.ylabel('吸光度')
plt.savefig('./correcting_2.png', dpi=500, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(wl, Xsnv.T[:1410])
plt.xlabel('波长（nm）')
plt.ylabel('吸光度')
plt.savefig('./correcting_3.png', dpi=500, bbox_inches="tight")
plt.show()

# import data and define wavelengths
data = pd.read_csv('./ref_merged.csv')
all = []
for i in range(29):
    spec = data.values[i][:2048]
    spec.tolist()
    all.append(spec)
all = np.array(all)
X = all
wl = np.arange(206.727,779.841,(779.841-206.727)/1410) # 波长

plt.figure(figsize=(12,7))
plt.plot(wl, X.T[:1410])
plt.xlabel('波长（nm）')
plt.ylabel('吸光度')
plt.savefig('./correcting_4.png', dpi=500, bbox_inches="tight")
plt.show()