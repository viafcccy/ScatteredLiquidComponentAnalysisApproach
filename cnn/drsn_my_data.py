#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:24:05 2019
Implemented using TensorFlow 1.0.1 and Keras 2.2.1

M. Zhao, S. Zhong, X. Fu, et al., Deep Residual Shrinkage Networks for Fault Diagnosis, 
IEEE Transactions on Industrial Informatics, 2019, DOI: 10.1109/TII.2019.2943898
There might be some problems in the Keras code. The weights in custom layers of models created using the Keras functional API may not be optimized.
https://www.reddit.com/r/MachineLearning/comments/hrawam/d_theres_a_flawbug_in_tensorflow_thats_preventing/
TensorFlow被曝存在严重bug，搭配Keras可能丢失权重
https://cloud.tencent.com/developer/news/661458
The TFLearn code is recommended for usage.
https://github.com/zhao62/Deep-Residual-Shrinkage-Networks/blob/master/DRSN_TFLearn.py
@author: super_9527
"""

from __future__ import print_function
import keras
import numpy as np
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
import os
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
K.set_learning_phase(1)

def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs,1),1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels)//2
    inputs = K.expand_dims(inputs,-1)
    inputs = K.spatial_3d_padding(inputs, ((0,0),(0,0),(pad_dim,pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)

# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    
    for i in range(nb_blocks):
        
        identity = residual
        
        if not downsample:
            downsample_strides = 1
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides), 
                          padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)
        
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal', 
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])
        
        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        
        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1,1), strides=(2,2))(identity)
            
        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)
        
        residual = keras.layers.add([residual, identity])
    
    return residual

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

# define and train a model
inputs = Input(shape=input_shape)
net = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
net = residual_shrinkage_block(net, 1, 8, downsample=True)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = GlobalAveragePooling2D()(net)
outputs = Dense(1, activation='linear', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
plot_model(model, to_file='./model_drsn.png', show_shapes=True)
model.compile(loss='mae', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, batch_size=1, epochs=1000, verbose=1, validation_data=(x_test, y_test))

# get results
K.set_learning_phase(0)
DRSN_train_score = model.evaluate(x_train, y_train, batch_size=100, verbose=0)
print('Train loss:', DRSN_train_score[0])
print('Train accuracy:', DRSN_train_score[1])
DRSN_test_score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)
print('Test loss:', DRSN_test_score[0])
print('Test accuracy:', DRSN_test_score[1])

# 将其模型转换为json
model_json = model.to_json()
with open(r"./model_data/zn_drsn_model.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
model.save_weights('./model_data/zn_drsn_model.h5')

# 加载模型用做预测
json_file = open(r"./model_data/zn_drsn_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model_data/zn_drsn_model.h5")
print("loaded model from disk")
loaded_model.compile(loss='mae', optimizer='adam', metrics=['mse'])
# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(x_test, y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# predicted_label = loaded_model.predict_classes(X)
Y_predicted = loaded_model.predict(x_test)
print("predict label:\n " + str(y_test))
print("calsses label:\n " + str(Y_predicted))