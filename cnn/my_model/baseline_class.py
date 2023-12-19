from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor

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
    model.add(Dense(30, activation='softmax'))
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

def train(X_train, Y_train):
    estimator = KerasRegressor(build_fn=baseline_model, epochs=40, batch_size=1, verbose=1)
    estimator.fit(X_train, Y_train)
