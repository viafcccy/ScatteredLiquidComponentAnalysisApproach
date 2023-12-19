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


input_shape = (256, 256, 1)

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
# npy_dir = r"../data/expanded_data_cor/"
npy_dir = r"../data/expanded_data/obj/"
npy_list = [os.path.join(npy_dir, file) for file in os.listdir(npy_dir) if file.endswith('.npy')]
for i in npy_list:
    gaf_npy = np.load(i)
    X.append(gaf_npy)
    # print(gaf_npy)
    group_index = i.split("/")[-1].split(".")[0].split("_")[0]
    Y.append(co[int(group_index) - 1]) # 根据文件名（组别），转为对应的钴离子浓度作为标签
Y = np.array(Y)
print(Y.shape)
X = np.array(X)
print(X.shape)
X = np.expand_dims(X, axis=3)
print(X.shape)

# 划分训练集，测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
print(x_train.shape)

# 注释是因为代码没有分离，注释后可以只预测，不训练
'''
inputs = Input(shape=input_shape)
# 模型结构定义
x = layers.Conv2D(32,3,strides=1,activation='relu',padding='same')(inputs)

branch_a = layers.Conv2D(16,1,activation='relu',padding='same',strides=2)(x)
branch_a = layers.Conv2D(16,3,activation='relu',padding='same',strides=2)(branch_a)

branch_b = layers.Conv2D(16,1,activation='relu',padding='same',strides=1)(x)
branch_b = layers.Conv2D(16,3,activation='relu',padding='same',strides=2)(branch_b)
branch_b = layers.Conv2D(16,3,activation='relu',padding='same',strides=2)(branch_b)

branch_c = layers.MaxPooling2D((3,3),padding='same',strides=2)(x)
branch_c = layers.Conv2D(16,1,activation='relu',padding='same',strides=2)(branch_c)

conc = layers.concatenate([branch_a,branch_b,branch_c],axis=-1)

f = layers.Flatten()(conc)

f1 = layers.Dense(256,activation='relu')(f)

outputs = layers.Dense(1,activation='linear')(f1)

# 
history = my_utils.LossHistory.LossHistory()
model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
plot_model(model, to_file='./model_multi.png', show_shapes=True)
model.compile(loss='mae', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=1, validation_data=(x_test, y_test), callbacks=[history])

def visual(model, data, num_layer=1):
    # data:图像array数据
    # layer:第n层的输出
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    print(f1.shape)
    num = f1.shape[-1]
    print(num)
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(int(np.ceil(np.sqrt(num))), int(np.ceil(np.sqrt(num))), i+1)
        plt.imshow(f1[:, :, i] * 255, cmap='gray')
        plt.axis('off')
    plt.savefig("multi_vs.png")
    plt.show()

# 将其模型转换为json
model_json = model.to_json()
with open(r"./model_data/zn_multi_model_2.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
model.save_weights('./model_data/zn_multi_model_2.h5')
'''

# 加载模型用做预测
json_file = open(r"./model_data/zn_multi_model_2.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model_data/zn_multi_model_2.h5")
print("loaded model from disk")
loaded_model.compile(loss='mae', optimizer='adam', metrics=['mse'])

Y_predicted = loaded_model.predict(x_test)
print("真实值:\n " + str(y_test))
print("预测值:\n " + str(Y_predicted))

# 可视化卷积层
# visual(model, x_train, 3)

# history.loss_plot('epoch', name = 'multi_2.png') # 每一个epoch展示一次

plt.figure()
x_plot = range(1,len(y_test)+1)
# acc
plt.plot(x_plot[:75], y_test[:75], 'r', label='真实浓度')
# loss
plt.plot(x_plot[:75], Y_predicted[:75], 'g', label='检测浓度')
plt.xlabel('测试集编号')
plt.ylabel('钴离子浓度（mg/L）')
plt.legend(loc="upper right")
plt.savefig("multi_test_2.png")
plt.show()

print(f"均方误差(MSE)：{mean_squared_error(Y_predicted, y_test)}")
print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(Y_predicted, y_test))}")
print(f"测试集R^2：{r2_score(y_test, Y_predicted)}")