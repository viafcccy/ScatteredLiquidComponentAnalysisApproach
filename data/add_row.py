from email import header


"""
@Description :   向merge_csv.py处理后的数据，最后一列的位置添加一列（这一列为组别）
@Author      :   ChenyuFang 
@Time        :   2022/04/24 21:54:04
"""

import os
import pandas as pd
import csv

processed_floder = './data/processed_data/目标光谱库'
csv_list = [os.path.join(processed_floder, file) for file in os.listdir(processed_floder) if file.endswith('.csv')]
print(csv_list)

print(u'共发现%s个CSV文件'% len(csv_list))
print(u'正在处理............')
for i in csv_list: #循环读取同文件夹下的csv文件    
    print(i)
    name = i.split("/")[-1]
    name_num = name.split(".")[0]
    print(name_num)

    with open(i) as csvFile:
        rows = csv.reader(csvFile)
        with open(('./data/merge/' + name), 'w') as f:
            writer = csv.writer(f)
            for row in rows:
                row.append(int(name_num))
                writer.writerow(row)
