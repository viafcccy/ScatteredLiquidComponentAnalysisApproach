import matplotlib.pyplot as plt
from keras.callbacks import Callback

# 写一个LossHistory类，保存训练集的loss和acc
# 当然我也可以完全不这么做，可以直接使用model.fit()方法返回的 history对象去做
'''Callback有6个常用的方法，这里实现其中的四个
    def on_epoch_begin(self, epoch, logs=None):
    def on_epoch_end(self, epoch, logs=None):
    def on_batch_begin(self, batch, logs=None):
    def on_batch_end(self, batch, logs=None):
    def on_train_begin(self, logs=None):
    def on_train_end(self, logs=None):
'''
class LossHistory(Callback):  # 继承自Callback类
 
    '''
    在模型开始的时候定义四个属性，每一个属性都是字典类型，存储相对应的值和epoch
    '''
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        # print(self.losses)
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    # 在每一个batch结束后记录相应的值
    def on_batch_end(self, batch, logs={}):
        # print(self.losses)
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('mse'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_mse'))
    
    # 在每一个epoch之后记录相应的值
    def on_epoch_end(self, batch, logs={}):
        # print(self.losses)
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('mse'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_mse'))

    def loss_plot(self, loss_type, name):
        '''
        loss_type：指的是 'epoch'或者是'batch'，分别表示是一个batch之后记录还是一个epoch之后记录
        '''
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='Train MSE')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='Train Loss')
        #if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='Val MSE')
            # val_loss
            # plt.plot(iters, self.val_loss[loss_type], 'k', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Values') #
        plt.legend(loc="upper right")
        plt.savefig(name, dpi=500, bbox_inches="tight") #
        plt.show()