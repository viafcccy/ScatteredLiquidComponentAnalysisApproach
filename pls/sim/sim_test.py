from sim.sepc_sim import SID, SAM
import pandas as pd
import numpy as np

data_refer = pd.read_csv('./data/processed_data/参考光谱库/fe-30条参考光谱.csv')
var_refer = data_refer.values[:,1:] # 吸光度
conc = data_refer['concentration'] # 浓度

data_obj = pd.read_csv('./data/processed_data/目标光谱库/1.csv')
var_obj = data_obj
sim_value_arr = []

for refer_index in range(30):
    sepc_refer = var_refer[refer_index,:]
    sepc_obj = np.mean(data_obj, 0) # 0 列平均，1 行平均
    sim_value = SID(sepc_refer, sepc_obj)
    sim_value_arr.append(sim_value)

print(sim_value_arr)