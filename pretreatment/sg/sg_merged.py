"""
@Description :   sg处理merged.csv
@Author      :   ChenyuFang 
@Time        :   2022/04/26 12:22:36
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv

all = np.array([])
data = pd.read_csv('./merged.csv')

with open('./merged_soothed.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    for i in range(881):
        spec = data.values[i][:2048]
        smooth_spec = savgol_filter(spec, 4*5+1, polyorder = 3*2, deriv=0)
        smooth_spec = np.append(smooth_spec, data.values[i][2048])
        writer.writerow(smooth_spec.tolist())
        print(i)


    


    