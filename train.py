# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:56:56 2019

@author: peter
"""

import loadData

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks as callbacks
import time
import numpy as np
# loadData.loadData()

tf.__version__


def create_model():
    model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(9600,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

    model = tf.keras.models.Sequential([
    keras.layers.Conv2D()
  ])


    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])


if __name__ == '__main__':
    time0 = time.time()

    x_train, x_test, y_train, y_test = loadData.loadData(number=10**2)

    print("start training")
    
    model = create_model()
    model.summary()

    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=False,
                                                     verbose=1)
    count=0
    while 1:
        count+=1
        x_train, y_train=loadData.generateData(number=200)
        model.fit(x_train,
              y_train,
              callbacks=[cp_callback])
        if count % 100 ==0:
            x_test, y_test=loadData.generateData(number=100)
            score, acc = model.evaluate(x_test, y_test, verbose=1)
            print('count is {}, Test: loss {}, acc {}'.format(count,score, acc))
            if acc >0.98:
                break

  #  score, acc = model.evaluate(x_test, y_test, verbose=1)
  #  print('Test: loss {}, acc {}'.format(score, acc))

    model.save('tf_model.h5')

    model.save_weights('tf_model_weights.h5')

    time1 = time.time()
    print("train : 总共花费 {0} s".format(time1-time0))
