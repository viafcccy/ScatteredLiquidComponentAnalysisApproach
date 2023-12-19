# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
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
    # print(gaf_npy)
    group_index = i.split("/")[-1].split(".")[0].split("_")[0]
    Y.append(zn[int(group_index) - 1])
Y = np.array(Y)
X = np.array(X)
X = np.expand_dims(X, axis=3)
print(X.shape)
print(Y.shape)

# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# 定义神经网络
def vgg19_2d():
    model = Sequential()
    # 1st block
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(256, 256, 1)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # 2nd block
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # 3rd block
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # 4th block
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # 5th block
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # full connextion
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='linear'))
    print(model.summary())
    # plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    model.compile(loss='mae',optimizer='adam', metrics=['mse'])
    return model
 
# 训练分类器
estimator = KerasRegressor(build_fn=vgg19_2d, epochs=40, batch_size=1, verbose=1)
estimator.fit(X_train, Y_train)
 
# 卷积网络可视化
# def visual(model, data, num_layer=1):
#     # data:图像array数据
#     # layer:第n层的输出
#     layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
#     f1 = layer([data])[0]
#     print(f1.shape)
#     num = f1.shape[-1]
#     print(num)
#     plt.figure(figsize=(8, 8))
#     for i in range(num):
#         plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
#         plt.imshow(f1[:, :, i] * 255, cmap='gray')
#         plt.axis('off')
#     plt.show()
 
# 混淆矩阵定义
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
    plt.yticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
 
# seed = 42
# np.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
# print("Accuracy of cross validation, mean %.2f, std %.2f\n" % (result.mean(), result.std()))
 
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))
 
# 将其模型转换为json
model_json = estimator.model.to_json()
with open(r"./model_data/zn_vgg19_2d_model.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
estimator.model.save_weights('./model_data/zn_vgg19_2d_model.h5')

# 加载模型用做预测
json_file = open(r"./model_data/zn_vgg19_2d_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model_data/zn_vgg19_2d_model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# predicted_label = loaded_model.predict_classes(X)
Y_predicted = loaded_model.predict(X_test)
print("predict label:\n " + str(Y_test))
print("calsses label:\n " + str(Y_predicted))
# 显示混淆矩阵
# plot_confuse(estimator.model, X_test, Y_test)

# 可视化卷积层
# visual(estimator.model, X_train, 1)