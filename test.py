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
from singleCaptchaGenerate import *
import loadData
from loadData import CAPTCHA_HEIGHT, CAPTCHA_WIDTH, VOCAB_LENGTH
import os
import tensorflow as tf
import keras
import time
import numpy as np


def decodeoneModelMultArray(vector):
    global CAPTCHA_LIST
    text = ''
    for i in range(0, 4):
        for j in range(0, VOCAB_LENGTH):
            if int(round(vector[i][0][j])) == 1:
                text += CAPTCHA_LIST[j]
    return text


def uselesstestoneModelMultArray():
    beforePath = "./oneModelMultArray"
    beforePath = beforePath+"/number"
    checkpoint_path = beforePath + '/cp.ckpt'
    model = load_model(checkpoint_path)
    img3 = Image.open("./img/0285.jpg")
    predict = model.predict(x_test)
    predict = np.array(predict)
    predict3 = np.int_(predict)
    predict2 = predict.astype('uint8')
    x_test, y_test = next(loadData.generateoneModelMultArray(100))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print(' Test: loss {}, acc {}'.format(score, acc))
#    model.evaluate_generator(loadData.generateoneModelMultArray(),steps=20)


def testoneModelMultArray():
    beforePath = "./oneModelMultArray"
    beforePath = beforePath+"/number"
    checkpoint_path = beforePath + '/cp.ckpt'
    model = load_model(checkpoint_path)
    img3 = Image.open("./img/0862.jpg")
    data = np.array(img3)
    data2 = data.reshape(1, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 3)
    predict = model.predict(data2)
    text = decodeoneModelMultArray(predict)
    import matplotlib.pyplot as plt
    #    plt.figure(figsize=(10,10))
    plt.imshow(img3)
    plt.title("predict result is "+text)
    plt.show()


if __name__ == '__main__':
    testoneModelMultArray()
