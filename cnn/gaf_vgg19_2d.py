from fileinput import filename
from hashlib import sha1
from time import process_time_ns
from my_model import vgg19_2d
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from keras.wrappers.scikit_learn import KerasRegressor
import os

# 锌离子浓度
zn = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# 钴离子浓度
co = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# 铁离子浓度
fe = [1.5, 3, 2, 2.5, 5, 3.5, 1, 5, 1.5, 1, 2, 3, 2.5, 4, 3.5, 1, 4.5, 1.5, 2, 4, 4.5, 4, 0.5, 3, 2.5, 0.5, 5, 4.5, 0.5, 3.5]

# 载入数据
X = []
Y = []
npy_dir = r"../data/expanded_data/obj/"
npy_list = [os.path.join(npy_dir, file) for file in os.listdir(npy_dir) if file.endswith('.npy')]
for i in npy_list:
    gaf_npy = np.load(i)
    X.append(gaf_npy)
    print(gaf_npy.shape)
    # print(gaf_npy)
    group_index = i.split("/")[-1].split(".")[0].split("_")[0]
    Y.append(co[int(group_index) - 1])
Y = np.array(Y)
print(Y.shape)
X = np.array(X)
print(X.shape)
X = np.expand_dims(X, axis=3)
print(X.shape)

# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# 模型
model = vgg19_2d.vgg19_2d()
print(model.summary())

# 训练
estimator = vgg19_2d.train(X_train, Y_train)

# 保存
struct_path = "./model_data/zn_baseline_model.json"
para_path = './model_data/zn_baseline_model.h5'
vgg19_2d.save(estimator, struct_path, para_path)

# 预测
vgg19_2d.predic(X_test, Y_test, struct_path, para_path)