import csv
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt

"""
@Description :   转为gaf保存.npy到特定文件夹
@Author      :   ChenyuFang 
@Time        :   2022/04/25 00:22:05
"""

wave_length = [i+1 for i in range(2048)]

for i in range(30):
    print('正在读取' + str(i+1) + '组...')
    group_index = i + 1
    with open('./processed_data/目标光谱库/' + str(group_index) + '.csv', 'r') as f:
        reader = csv.reader(f)

        X = []

        # 读取行数
        count = 0
        for row in reader:
            data = list(map(float,row))[:,2049]
            X.append(data)
            count += 1

        print('该组有' + str(count) + '个数据')

        print('开始处理...')
        gaf = GramianAngularField(method='difference', overlapping=True, image_size=0.125) # difference summation
        X_gaf = gaf.fit_transform(X)
        print('处理完成...')

        for i in range(count):
            print('正在保存组内第' + str(i+1) + '个数据...')
            gram = X_gaf[i]
            dir = './expanded_data/obj/'
            np.save(dir + str(group_index) + '_' + str(i+1) + '.npy' , gram)
            # print(gram.shape)
            # exit()
            # np.savetxt(dir + str(group_index) + '_' + str(i+1) + '.txt',gram)