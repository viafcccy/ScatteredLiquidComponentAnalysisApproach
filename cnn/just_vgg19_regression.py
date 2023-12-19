from my_model import vgg19_1d_regression
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from keras.wrappers.scikit_learn import KerasRegressor

# 锌离子浓度
zn = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# 钴离子浓度
co = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# 铁离子浓度
fe = [1.5, 3, 2, 2.5, 5, 3.5, 1, 5, 1.5, 1, 2, 3, 2.5, 4, 3.5, 1, 4.5, 1.5, 2, 4, 4.5, 4, 0.5, 3, 2.5, 0.5, 5, 4.5, 0.5, 3.5]
 
# 载入数据
df = pd.read_csv(r"../data/obj_merged_data/merged.csv")
X = np.expand_dims(df.values[:, 0:2048].astype(float), axis=2)
Y = []
Y_index = df.values[:, 2048].tolist()
# 实验组编号映射为真实的浓度值
for i in Y_index:
    Y.append(co[int(i) - 1])
Y = np.array(Y)

# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# 模型
model = vgg19_1d_regression.vgg19_1d()
print(model.summary())

# 训练
estimator = vgg19_1d_regression.train(X_train, Y_train)

# 保存
struct_path = "./model_data/zn_vgg19_model.json"
para_path = './model_data/zn_vgg19_model.h5'
vgg19_1d_regression.save(estimator, struct_path, para_path)

# 预测
vgg19_1d_regression.predic(X_test, Y_test, struct_path, para_path)