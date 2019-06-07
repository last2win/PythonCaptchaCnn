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
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten, Activation, Flatten

import singleCaptchaGenerate

#singleCaptchaGenerate.CAPTCHA_LIST=singleCaptchaGenerate.NUMBER+singleCaptchaGenerate.UP_CASE
#singleCaptchaGenerate.n_class=singleCaptchaGenerate.VOCAB_LENGTH = len(singleCaptchaGenerate.CAPTCHA_LIST)


import loadData
from loadData import CAPTCHA_HEIGHT, CAPTCHA_WIDTH, VOCAB_LENGTH
import os
import tensorflow as tf
import keras
import time
import numpy as np
# loadData.loadData()

beforePath = "/content/gdrive/My Drive/my-project3/oneModelOneArray"
#beforePath = "./oneModelMultArray"
beforePath = beforePath+"/number"


def createModel2():
    input_tensor = Input((CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3))

    x = input_tensor

    for i in range(4):
        x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
        x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(VOCAB_LENGTH, activation='softmax', name='c%d' % (i+1))(x)
         for i in range(4)]

    model = Model(input=input_tensor, output=x)


    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    if 1 == 1:
        time0 = time.time()
        print("start training")
        beforePath = beforePath # +"/"+"create_model"
        checkpoint_path = beforePath + '/logs/'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=False,
                                                      verbose=1)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=5, verbose=0, mode='auto',
                                                       baseline=None, restore_best_weights=False)

        TensorBoardcallback = keras.callbacks.TensorBoard(
            log_dir=beforePath,
            histogram_freq=0, batch_size=32,
            write_graph=True, write_grads=False, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None, update_freq='batch'
        )
        model = createModel2()
        print(model.summary())
#        model.save(beforePath +'/model.h5')
        model.fit_generator(loadData.generateKerasYieldData2(),
                            # steps_per_epoch=51200,  # 一轮多少个
                            # nb_epoch=5,  # 训练 nb_epoch 轮
                            steps_per_epoch=51200,  # 一轮多少个
                            nb_epoch=2,
                            workers=1,  use_multiprocessing=False,  # 单线程
                            #            nb_worker=2, pickle_safe=True,
                            # validation_data: 它可以是以下之一： 验证数据的生成器或 Sequence 实例
                            validation_data=loadData.generateKerasYieldData2(),
                            validation_steps=1280,  # 验证样本数
                            callbacks=[cp_callback,
                                       TensorBoardcallback, early_stopping], verbose=1
                            )
        model.save(beforePath + '/model.h5')

        time1 = time.time()
        print("train : 总共花费 {0} s".format(time1-time0))
