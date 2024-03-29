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
from keras.layers import Dense, Flatten, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,MaxPooling2D,Dropout
from keras.layers import Dense, Flatten, Activation,Flatten
import loadData
from loadData import CAPTCHA_HEIGHT,CAPTCHA_WIDTH,VOCAB_LENGTH
import os
import tensorflow as tf
import keras
import time
import numpy as np
# loadData.loadData()

beforePath = "/content/gdrive/My Drive/my-project3/my-keras3"
beforePath = "./my-keras3"

def build(input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3), num_classes=VOCAB_LENGTH):
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

def build2(input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3), num_classes=VOCAB_LENGTH):
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
    model.add([Dense(num_classes, activation='softmax') for i in range(0,4) ])
 #   model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
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
        #       Dropout(0.2),

        #      keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(9600,)),
        #      keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss=tf.keras.losses.sparse_categorical_crossentropy,
    #               metrics=['accuracy'])
    return model



if __name__ == '__main__':

    for i in range(0,4):
        time0 = time.time()
        print("start training")
        j=str(i)
        beforePath=beforePath+"/"+"build"+"/"+j
        loadData.__NOW__=i
        checkpoint_path = beforePath +'/logs'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=False,
                                                      verbose=1)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=5, verbose=0, mode='auto',
                                                       baseline=None, restore_best_weights=False)

        TensorBoardcallback = keras.callbacks.TensorBoard(
            log_dir= beforePath ,
            histogram_freq=0, batch_size=32,
            write_graph=True, write_grads=False, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None, update_freq='batch'
        )
        model=build()
        print(model.summary())
        model.save(beforePath +'/model.h5')
        model.fit_generator(loadData.generateKerasYieldData3(),
                            # steps_per_epoch=51200,  # 一轮多少个
                            # nb_epoch=5,  # 训练 nb_epoch 轮
                            steps_per_epoch=5120,  # 一轮多少个
                            nb_epoch=2,
                            workers=1,  use_multiprocessing=False,  # 单线程
                            #            nb_worker=2, pickle_safe=True,
                            # validation_data: 它可以是以下之一： 验证数据的生成器或 Sequence 实例
                            validation_data=loadData.generateKerasYieldData3(),
                            validation_steps=1280,  # 验证样本数
                            callbacks=[cp_callback,
                                       TensorBoardcallback, early_stopping]
                            )
        model.save(beforePath +'/model.h5')

        time1 = time.time()
        print("train : 总共花费 {0} s".format(time1-time0))
