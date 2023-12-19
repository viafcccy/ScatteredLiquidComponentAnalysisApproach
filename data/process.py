import csv
import os

"""
@Description :   将原始数据，改为csv内紧凑的2D数组
@Author      :   ChenyuFang 
@Time        :   2022/04/24 21:56:40
"""

# 读取一个原始光谱数据改写为便于处理的格式
def conversion_layout(old_file_name, new_file_name, previous_floder, processed_floder, concentration = 0, need_position = False, pattern = 'a'):

    if concentration == -1: # 为-1时不保存第一列，浓度列，作为处理30条谱线合成时使用
        position = []
        value = []
    else:
        position = ['concentration']
        value = [float(concentration)]
    with open(previous_floder + '/' + old_file_name, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        for line_index,line in enumerate(csv_reader):
            if line_index > 8:
                line_str = line[0]
                info = line_str.split(',')
                position.append(float(info[0]))
                value.append(float(info[1]))

    with open(processed_floder + '/' + new_file_name, pattern, newline='') as csvfile: # 覆盖写入用'w'，追加为'a'
        csv_writer = csv.writer(csvfile)
        if need_position:
            csv_writer.writerow(position)
        csv_writer.writerow(value)

    return None

"""
常数
"""
# 锌离子浓度
zn = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# 钴离子浓度
co = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# 铁离子浓度
fe = [1.5, 3, 2, 2.5, 5, 3.5, 1, 5, 1.5, 1, 2, 3, 2.5, 4, 3.5, 1, 4.5, 1.5, 2, 4, 4.5, 4, 0.5, 3, 2.5, 0.5, 5, 4.5, 0.5, 3.5]

"""
参比光谱
"""
# 处理参比光谱
previous_floder = './previous_data'
processed_floder = './processed_data'
conversion_layout('参比光谱.csv', '参比光谱.csv', previous_floder, processed_floder, need_position= True, pattern= 'w')


"""
参考光谱
"""
# 处理zn标准库光谱
previous_floder = './previous_data/参考光谱库'
processed_floder = './processed_data/参考光谱库'
file_list = [os.path.join(previous_floder, file) for file in os.listdir(previous_floder) if file.endswith('.csv')]
index = 0
for file_name in file_list:
    if index == 0:
        conversion_layout(file_name.split("/")[-1], 'zn-30条参考光谱.csv', previous_floder, processed_floder, concentration = zn[index],need_position = True, pattern= 'w')
    else:
        conversion_layout(file_name.split("/")[-1], 'zn-30条参考光谱.csv', previous_floder, processed_floder, concentration = zn[index])
    index = index + 1

# 处理co标准库光谱
previous_floder = './previous_data/参考光谱库'
processed_floder = './processed_data/参考光谱库'
file_list = [os.path.join(previous_floder, file) for file in os.listdir(previous_floder) if file.endswith('.csv')]
index = 0
for file_name in file_list:
    if index == 0:
        conversion_layout(file_name.split("/")[-1], 'co-30条参考光谱.csv', previous_floder, processed_floder, concentration = co[index], need_position = True, pattern= 'w')
    else:
        conversion_layout(file_name.split("/")[-1], 'co-30条参考光谱.csv', previous_floder, processed_floder, concentration = co[index])
    index = index + 1

# 处理fe标准库光谱
previous_floder = './previous_data/参考光谱库'
processed_floder = './processed_data/参考光谱库'
file_list = [os.path.join(previous_floder, file) for file in os.listdir(previous_floder) if file.endswith('.csv')]
index = 0
for file_name in file_list:
    if index == 0:
        conversion_layout(file_name.split("/")[-1], 'fe-30条参考光谱.csv', previous_floder, processed_floder, concentration = fe[index], need_position = True, pattern= 'w')
    else:
        conversion_layout(file_name.split("/")[-1], 'fe-30条参考光谱.csv', previous_floder, processed_floder, concentration = fe[index])
    index = index + 1

"""
目标光谱
"""
for group_num in range(1,31):
    previous_floder = './previous_data/目标光谱库' + '/' + str(group_num)
    processed_floder = './processed_data/目标光谱库'
    file_list = [os.path.join(previous_floder, file) for file in os.listdir(previous_floder) if file.endswith('.csv')]
    print('当前正在处理文件夹：' + str(previous_floder))
    index = 0
    for file_name in file_list:
        if index == 0:
            conversion_layout(file_name.split("/")[-1], str(group_num) + '.csv', previous_floder, processed_floder, concentration = -1, need_position = False, pattern= 'w')
        else:
            conversion_layout(file_name.split("/")[-1], str(group_num) + '.csv', previous_floder, processed_floder, concentration = -1)
        index = index + 1
        print('当前正在处理文件：' + str(file_name))