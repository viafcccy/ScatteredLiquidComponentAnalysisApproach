"""
@Description :   逐一校正，输出到csv
@Author      :   ChenyuFang 
@Time        :   2022/04/26 20:44:57
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from copy import deepcopy

def snv(input_data):
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data

all = []
data = pd.read_csv('./merged_smoothed.csv')

for i in range(880):
    spec = data.values[i][:2048]
    spec.tolist()
    all.append(spec)

all = np.array(all)
print(all)
print(all.shape)
correct_spec = snv(all)
data = pd.DataFrame(correct_spec)
data.to_csv('./merged_smoothed_corrected.csv')