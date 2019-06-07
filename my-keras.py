# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:21:51 2019

@author: peter


import tensorflow as tf
tf.test.gpu_device_name()

from google.colab import drive
drive.mount('/content/gdrive')


import sys
!{sys.executable} -m pip install captcha


!date -R
import os
os.environ['TZ'] = "Asia/Shanghai"
!date -R

!cat "/content/gdrive/My Drive/my-project3/my-keras.py"

!python "/content/gdrive/My Drive/my-project3/my-keras.py" 2>&1 | tee  "/content/gdrive/My Drive/my-project3/my-keras/$(date).log"



tensorboard --logdir="C:/Users/peter/Google 云端硬盘/my-project3/my-keras/logs"



"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,MaxPooling2D,Dropout
from keras.layers import Dense, Flatten, Activation,Flatten
import loadData
from loadData import CAPTCHA_HEIGHT, CAPTCHA_WIDTH
import os
import tensorflow as tf
import keras
import time
import numpy as np
# loadData.loadData()

beforePath = "/content/gdrive/My Drive/my-project3/my-keras"
beforePath = "./my-keras"

def build(input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3), num_classes=40):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax') )
 #   model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return model


def build2(input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3), num_classes=40):
    model = Sequential()

    for i in range(4):
        if i==0:
            model.add(Conv2D(
            # filters：卷积核的数目（即输出的维度）
            filters=32*2**i,
            # kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。
            # 如为单个整数，则表示在各个空间维度的相同长度。
            kernel_size=(3, 3),
            # activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。
            # 如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
            activation='relu',
            input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3)))
        else:
            model.add(Conv2D(
            # filters：卷积核的数目（即输出的维度）
            filters=32*2**i,
            # kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。
            # 如为单个整数，则表示在各个空间维度的相同长度。
            kernel_size=(3, 3),
            # activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。
            # 如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
            activation='relu'))


        model.add(Conv2D(
            # filters：卷积核的数目（即输出的维度）
            filters=32*2**i,
            # kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。
            # 如为单个整数，则表示在各个空间维度的相同长度。
            kernel_size=(3, 3),
            # activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。
            # 如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
            activation='relu'))

        model.add(MaxPool2D(
            # pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，
            # 如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
            pool_size=(2, 2)))

    model.add(Flatten())  # 压平
    model.add(Dropout(0.25))
    model.add(Dense(40, activation='softmax'))  # 全连接
 #   model.add(Activation('softmax'))

    # # initiate RMSprop optimizer
    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # # Let's train the model using RMSprop
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=opt,
    #               metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

def create_model():
    model = keras.models.Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
               input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3), padding="same", strides=(1, 1)),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        #             Dropout(0.2),
        keras.layers.Dropout(0.1),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
               padding="same", strides=(1, 1)),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Dropout(0.1),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
               padding="same", strides=(1, 1)),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Dropout(0.1),
        Flatten(),
        Dense(1024, activation='relu'),
        keras.layers.Dense(40, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])


    return model


if __name__ == '__main__':
    time0 = time.time()

 #   x_train, x_test, y_train, y_test = loadData.loadData(number=10**2)

    print("start training")

    model=build2()
    print(model.summary())

    checkpoint_path = beforePath + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  save_weights_only=False,
                                                  verbose=1)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=5, verbose=0, mode='auto',
                                                   baseline=None, restore_best_weights=False)

    TensorBoardcallback = keras.callbacks.TensorBoard(
        log_dir=beforePath+'/logs',
        histogram_freq=0, batch_size=32,
        write_graph=True, write_grads=False, write_images=True,
        embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='batch'
    )

    model.fit_generator(loadData.generateKerasYieldData(),
                        # steps_per_epoch=51200,  # 一轮多少个
                        # nb_epoch=5,  # 训练 nb_epoch 轮
                        steps_per_epoch=5120,  # 一轮多少个
                        nb_epoch=5,
                        workers=1,  use_multiprocessing=False,  # 单线程
                        #            nb_worker=2, pickle_safe=True,
                        # validation_data: 它可以是以下之一： 验证数据的生成器或 Sequence 实例
                        validation_data=loadData.generateKerasYieldData(),
                        validation_steps=1280,  # 验证样本数
                        callbacks=[cp_callback,
                                   TensorBoardcallback, early_stopping]
                        )

    model.save(
        beforePath+'/model.h5')

    time1 = time.time()
    print("train : 总共花费 {0} s".format(time1-time0))
