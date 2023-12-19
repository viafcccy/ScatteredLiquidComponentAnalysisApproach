from operator import index
from keras import Input,layers,Model
import numpy as np
import os
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from turtle import onclick
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import my_utils.LossHistory
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 锌离子浓度
zn = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# 钴离子浓度
co = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# co = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
co_2 = [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 0.5, 1.5]
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
print(x_train.shape)

def visual(model, data, num_layer=1):
    # data:图像array数据
    # layer:第n层的输出
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    print('a')
    print(f1.shape)
    num = f1.shape[-1]
    print(num)
    plt.figure(figsize=(8, 8))
    for i in range(num):
        # plt.subplot(int(np.ceil(np.sqrt(num))), int(np.ceil(np.sqrt(num))), i+1)
        ax = plt.subplot(6, 8, i+1)
        group = int(np.ceil((i + 1)/16))
        index = (((i + 1) - (group-1)*16)+1) - 1
        ax.set_title(str(group) + '-' + str(index))
        plt.imshow(f1[1,:, :, i] * 255, cmap='gray')
        plt.axis('off')
    plt.savefig("multi_vs.png")
    plt.show()

# 加载模型用做预测
json_file = open(r"./model_data/zn_multi_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model_data/zn_multi_model.h5")
print("loaded model from disk")
loaded_model.compile(loss='mae', optimizer='adam', metrics=['mse'])

# 可视化卷积层
visual(loaded_model, x_train, 9)
