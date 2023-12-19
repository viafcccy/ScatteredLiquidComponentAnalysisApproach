# -*- coding: utf8 -*-
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
co_2 = [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 0.5, 1.5]
# 铁离子浓度
fe = [1.5, 3, 2, 2.5, 5, 3.5, 1, 5, 1.5, 1, 2, 3, 2.5, 4, 3.5, 1, 4.5, 1.5, 2, 4, 4.5, 4, 0.5, 3, 2.5, 0.5, 5, 4.5, 0.5, 3.5]

# 载入数据
df = pd.read_csv(r"../data/merged_data/merged_index.csv")
X = np.expand_dims(df.values[:, 0:2048].astype(float), axis=2)
Y = []
Y_index = df.values[:, 2048].tolist()
for i in Y_index:
    print(type(i))
    Y.append(co[int(i) - 1])
Y = np.array(Y)
print(X.shape)
print(Y.shape)

# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(2048, 1)))
    model.add(Conv1D(16, 3, activation='tanh'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='tanh'))
    model.add(Conv1D(32, 3, activation='tanh'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='linear'))
    print(model.summary())
    plot_model(model, to_file='./baseline_struct.png', show_shapes=True)
    model.compile(loss='mae',optimizer='adam', metrics=['mse'])
    return model

history = my_utils.LossHistory.LossHistory()
# 训练
estimator = KerasRegressor(build_fn=baseline_model, epochs=2, batch_size=1, verbose=1)
estimator.fit(X_train, Y_train, callbacks=[history])

def visual(model, data, num_layer=1):
    # data:图像array数据
    # layer:第n层的输出
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    print(f1.shape)
    num = f1.shape[-1]
    print(num)
    plt.figure()
    for i in range(num):
        plt.subplot(int(np.ceil(np.sqrt(num))), int(np.ceil(np.sqrt(num))), i+1)
        plt.imshow(f1[:, :, i] * 255, cmap='gray')
        plt.axis('off')
    plt.savefig("baseline_vs.png")
    plt.show()

# 将其模型转换为json
model_json = estimator.model.to_json()
with open(r"./model_data/zn_baseline_model.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
estimator.model.save_weights('./model_data/zn_baseline_model.h5')

# 加载模型用做预测
json_file = open(r"./model_data/zn_baseline_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model_data/zn_baseline_model.h5")
print("loaded model from disk")
loaded_model.compile(loss='mae', optimizer='adam', metrics=['mse'])

Y_predicted = loaded_model.predict(X_test)
print("真实值:\n " + str(Y_test))
print("预测值:\n " + str(Y_predicted))

# 可视化卷积层
visual(estimator.model, X_train, 8)

history.loss_plot('epoch', name = 'baseline.png') # 每一个epoch展示一次

plt.figure(figsize=(6,6))
x_plot = range(1,len(Y_test)+1)
# acc
plt.plot(x_plot[:75], Y_test[:75], 'r', marker='o', label='真实浓度')
# loss
plt.plot(x_plot[:75], Y_predicted[:75], 'g', marker='x', label='检测浓度')
plt.xlabel('测试集编号')
plt.ylabel('钴离子浓度（mg/L）')
plt.legend(loc="upper right")
plt.savefig("baseline_test.png", dpi=500, bbox_inches="tight")
plt.show()

print(f"均方误差(MSE)：{mean_squared_error(Y_predicted, Y_test)}")
print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(Y_predicted, Y_test))}")
print(f"测试集R^2：{r2_score(Y_test, Y_predicted)}")