"""
@Description :   输入变量的index，取左右各3个共7个变量，与参考光谱计算相似度。
@Author      :   ChenyuFang 
@Time        :   2022/04/28 20:23:33
"""
from . import sepc_sim

# x为index，舍弃前三个和后三个变量
# ref_spec为参考光谱，obj为目标光谱
def sim_var(x, ref_spec, obj_spec):
    if x < 3 or x > len(ref_spec) - 3:
        return False
    x_strat = x - 3
    x_end = x + 3 + 1
    # print(x_strat)
    # print(x_end)
    return sepc_sim.SID(ref_spec[x_strat:x_end], obj_spec[x_strat:x_end])