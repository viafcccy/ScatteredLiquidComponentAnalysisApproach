import os
import pandas as pd
import csv

"""
@Description :   将processed_data下的数据合并为一个csv，保存至obj_merged_data
@Author      :   ChenyuFang 
@Time        :   2022/04/24 21:55:17
"""

processed_floder = './processed_data/目标光谱库'
csv_list = [os.path.join(processed_floder, file) for file in os.listdir(processed_floder) if file.endswith('.csv')]
print(csv_list)

print(u'共发现%s个CSV文件'% len(csv_list))
print(u'正在处理............')
for i in csv_list: #循环读取同文件夹下的csv文件

    fr = open(i,'rb').read()
    with open('./obj_merged_data/merged.csv','ab') as f: #将结果保存为result.csv
        f.write(fr)

print(u'合并完毕！')


