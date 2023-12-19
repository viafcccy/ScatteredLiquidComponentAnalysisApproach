import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json

# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(2048, 1)))
    model.add(Conv1D(16, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    # plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    model.compile(loss='mae',optimizer='adam', metrics=['mse'])
    return model

def train(X_train,  Y_train):
    estimator = KerasRegressor(build_fn=baseline_model, epochs=40, batch_size=1, verbose=1)
    estimator.fit(X_train, Y_train)
    return 

def save(estimator, struct_path, para_path):
# 将其模型转换为json
    model_json = estimator.model.to_json()
    with open(struct_path,'w')as json_file:
        json_file.write(model_json)# 权重不在json中,只保存网络结构
    estimator.model.save_weights(para_path)

def predic(X_test, Y_test, struct_path, para_path):
    # 加载模型用做预测
    json_file = open(struct_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(para_path)
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