# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:58:50 2019

@author: peter
"""

from keras.layers import Dense, Flatten, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten, Activation, Flatten
from keras.models import load_model
import singleCaptchaGenerate
from PIL import Image

import loadData
from loadData import CAPTCHA_HEIGHT, CAPTCHA_WIDTH, VOCAB_LENGTH
import os
import tensorflow as tf
import keras
import time
import numpy as np


def testoneModelMultArray():
    beforePath = "./oneModelMultArray"
    beforePath = beforePath+"/number"
    checkpoint_path = beforePath+ '/cp.ckpt'
    model=load_model(checkpoint_path)
    img3=Image.open("./img/0285.jpg")
    predict=model.predict(x_test)
    predict=np.array(predict)
    predict3=np.int_(predict)
    predict2=predict.astype('uint8')
    x_test, y_test=next(loadData.generateoneModelMultArray(100))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print(' Test: loss {}, acc {}'.format(score, acc))
#    model.evaluate_generator(loadData.generateoneModelMultArray(),steps=20)

if __name__ == '__main__':
 #   testoneModelMultArray()
    beforePath = "./oneModelMultArray"
    beforePath = beforePath+"/number"
    checkpoint_path = beforePath+ '/cp.ckpt'
    model=load_model(checkpoint_path)
    img3=Image.open("./img/0285.jpg")
    data=np.array(img3)
    data2=data.reshape(1, CAPTCHA_WIDTH, CAPTCHA_HEIGHT,3)
    predict=model.predict(data)