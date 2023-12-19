import sim.sim_var
import pandas as pd

a = [1,1,1,1,1,1,1]
b = [2,2,3,3,2,2,2]
sim_value = sim.sim_var.sim_var(3, a, b)

# 获取全目标光谱变量参考值
data = pd.read_csv('./co-30条参考光谱.csv')
data_obj = pd.read_csv('./merged_index.csv')

X_ref = data.values[:,1:] # 吸光度
y_ref = data['concentration'] # 浓度

x_obj = data_obj.values[:,:2048]
y_obj_index = [i[2048] for i in data_obj.values]

sim_list = []
for i in range(len(y_obj_index)):
    sim_value = sim.sim_var.sim_var(i, X_ref, y_ref)
    if sim == False:
        sim = 999
    sim_list.append(sim_value)
    print(sim_value)


