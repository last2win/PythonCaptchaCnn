# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:57:28 2019

@author: peter
"""
import os
import imageio
import numpy as np
from sklearn.model_selection import train_test_split
from singleCaptchaGenerate import CAPTCHA_LENGTH, VOCAB_LENGTH, CAPTCHA_LIST,CAPTCHA_HEIGHT,CAPTCHA_WIDTH
from singleCaptchaGenerate import generateCaptchaTextAndImage
from PIL import Image
n_class=VOCAB_LENGTH

def convert2gray(img):
    """
    图片转为黑白，3维转1维
    :param img: np
    :return:  灰度图的np
    """
    if len(img.shape) > 2:
        img2 = np.mean(img, -1)
    return img2


def text2vec(text):
    """
    text to one-hot vector
    :param text: source text
    :return: np array
    """
    if len(text) > CAPTCHA_LENGTH:
        raise ValueError('验证码长度超过'+str(CAPTCHA_LENGTH)+'个字符')
    vector = np.zeros(CAPTCHA_LENGTH * VOCAB_LENGTH, dtype=np.int8)
 #   vector = np.zeros(CAPTCHA_LENGTH * VOCAB_LENGTH)
    for i, c in enumerate(text):
        index = i * VOCAB_LENGTH + CAPTCHA_LIST.index(c)
        vector[index] = 1
    return vector


def text2vec2(text):
    """
    text to one-hot vector
    :param text: source text
    :return: np array
    """
    if len(text) > CAPTCHA_LENGTH:
        raise ValueError('验证码长度超过'+str(CAPTCHA_LENGTH)+'个字符')
    vector = np.zeros(CAPTCHA_LENGTH * VOCAB_LENGTH, dtype=np.int8)
 #   vector = np.zeros(CAPTCHA_LENGTH * VOCAB_LENGTH)
    for i, c in enumerate(text):
        index = i * VOCAB_LENGTH + CAPTCHA_LIST.index(c)
        vector[index] = 1
    return vector

def vec2text(vector):
    """
    vector to captcha text
    :param vector: np array
    :return: text
    """
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LENGTH, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_LIST[np.argmax(item)]
    return text


def loadData(openDir="./img/", number=10**2, split=0.2):
    count = 0
    data = []
    labels = []
    pass
    x_train, x_test, y_train, y_test = [], [], [], []
    fileNames = os.listdir(openDir)
    np.random.shuffle(fileNames)
    for fileName in fileNames:
        #        print(fileName)
        if fileName.endswith("jpg"):
            count += 1
            if count > number:
                break
            if count % 100 == 0:
                print("load count is ", count)
            img = imageio.imread(openDir+fileName)
            img2 = convert2gray(img)
            img3 = img2.flatten() / 255
 #           img = im2double(imageio.imread(openDir+fileName))
 #           if len(img.shape) == 2:
 #               img = img[:, :, np.newaxis]
  #          print(fileName)
            data.append(img3)
            labels.append(text2vec(fileName[:4]))
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=split)
    return x_train, x_test, y_train, y_test


def loadData2(openDir="./img/", number=10**2, split=0.2):
    count = 0
    fileNames = os.listdir(openDir)
    np.random.shuffle(fileNames)
    for fileName in fileNames:
        #        print(fileName)
        if fileName.endswith("jpg"):
            count += 1
            img = imageio.imread(openDir+fileName)
 #           img = im2double(imageio.imread(openDir+fileName))
 #           if len(img.shape) == 2:
 #               img = img[:, :, np.newaxis]
            print(fileName)
            return img


def generateData(number=10**2):
    data = []
    labels = []
    for i in range(0, number):
        if i % 100 == 0 and i > 0:
            print("load count is ", i)
        captchaText, captcha_image = generateCaptchaTextAndImage()
        img2 = convert2gray(captcha_image)
        img3 = img2.flatten() / 255
        data.append(img3)
        labels.append(text2vec(captchaText))

    return data,labels

def generateGreyKerasData(number=10**2):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH
    data = []
    labels = []
    for i in range(0, number):
 #       if i % 100 == 0 and i > 0:
 #           print("load count is ", i)
        captchaText, captcha_image = generateCaptchaTextAndImage()
        img2 = convert2gray(captcha_image)
        img3 = img2.flatten() / 255
        data.append(img3)
 #       data.append(img3)
        labels.append(text2vec(captchaText))
 #   return data,labels
    data1=np.array(data)
    labels=np.array(labels)
    data2=data1.reshape(data1.shape[0], CAPTCHA_WIDTH, CAPTCHA_HEIGHT,1)
    data3=data2.astype('float32')
 #   data3/=255
    return data3,labels

def generateKerasGreyYieldData(batch_size=32):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH
    while True:
        data = []
        labels = []
        for i in range(0, batch_size):
     #       if i % 100 == 0 and i > 0:
     #           print("load count is ", i)
            captchaText, captcha_image = generateCaptchaTextAndImage()
            img2 = convert2gray(captcha_image)
            img3 = img2.flatten() / 255
            data.append(img3)
     #       data.append(img3)
            labels.append(text2vec(captchaText))
     #   return data,labels
        data1=np.array(data)
        labels=np.array(labels)
        data2=data1.reshape(data1.shape[0], CAPTCHA_HEIGHT,CAPTCHA_WIDTH,1)
        data3=data2.astype('float32')
     #   data3/=255
        yield data3,labels

def generateKerasData(number=10**2):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH
    data = []
    labels = []
    for i in range(0, number):
 #       if i % 100 == 0 and i > 0:
 #           print("load count is ", i)
        captchaText, captcha_image = generateCaptchaTextAndImage()
        data.append(captcha_image)
        labels.append(text2vec(captchaText))
 #   return data,labels
    data1=np.array(data)
    labels=np.array(labels)
  #  data2=data1.reshape(data1.shape[0], CAPTCHA_WIDTH, CAPTCHA_HEIGHT,3)
    data3=data1.astype('float32')
    data3/=255
    return data3,labels

def generateKerasYieldData(batch_size=32):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH
    while True:
        data = []
        labels = []
        for i in range(0, batch_size):
     #       if i % 100 == 0 and i > 0:
     #           print("load count is ", i)
            captchaText, captcha_image = generateCaptchaTextAndImage()
            data.append(captcha_image)
            labels.append(text2vec(captchaText))
     #   return data,labels
        data1=np.array(data)
        labels=np.array(labels)
      #  data2=data1.reshape(data1.shape[0], CAPTCHA_WIDTH, CAPTCHA_HEIGHT,3)
        data3=data1.astype('float32')
        data3/=255
        yield data3,labels

def generateKerasYieldData2(batch_size=32):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH
    while True:
        data = []
        labels = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(CAPTCHA_LENGTH)]
        for i in range(0, batch_size):
     #       if i % 100 == 0 and i > 0:
     #           print("load count is ", i)
            captchaText, captcha_image = generateCaptchaTextAndImage()
            data.append(captcha_image)
            for j, ch in enumerate(captchaText):
                labels[j][i, :] = 0
                labels[j][i, CAPTCHA_LIST.index(ch)] = 1
 #           labels.append(text2vec(captchaText))
     #   return data,labels
        data1=np.array(data)
#        labels=np.array(labels)
      #  data2=data1.reshape(data1.shape[0], CAPTCHA_WIDTH, CAPTCHA_HEIGHT,3)
        data3=data1.astype('float32')
        data3/=255
        yield data3,labels

def generateoneModelMultArray(batch_size=32):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH
    while True:
        data = []
        labels = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(CAPTCHA_LENGTH)]
        for i in range(0, batch_size):
     #       if i % 100 == 0 and i > 0:
     #           print("load count is ", i)
            captchaText, captcha_image = generateCaptchaTextAndImage()
            data.append(captcha_image)
            for j, ch in enumerate(captchaText):
                labels[j][i, :] = 0
                labels[j][i, CAPTCHA_LIST.index(ch)] = 1
 #           labels.append(text2vec(captchaText))
     #   return data,labels
        data1=np.array(data)
#        labels=np.array(labels)
      #  data2=data1.reshape(data1.shape[0], CAPTCHA_WIDTH, CAPTCHA_HEIGHT,3)
        data3=data1.astype('float32')
  #      data3/=255
        yield data3,labels

__NOW__=-1
def generateKerasYieldData3(batch_size=32):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH,__NOW__
    while True:
        data = []
        labels = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(CAPTCHA_LENGTH)]
        for i in range(0, batch_size):
     #       if i % 100 == 0 and i > 0:
     #           print("load count is ", i)
            captchaText, captcha_image = generateCaptchaTextAndImage()
            data.append(captcha_image)
            for j, ch in enumerate(captchaText):
                labels[j][i, :] = 0
                labels[j][i, CAPTCHA_LIST.index(ch)] = 1
 #           labels.append(text2vec(captchaText))
     #   return data,labels
        data1=np.array(data)
#        labels=np.array(labels)
      #  data2=data1.reshape(data1.shape[0], CAPTCHA_WIDTH, CAPTCHA_HEIGHT,3)
        data3=data1.astype('uint8')
        data3/=255
        if __NOW__==-1:
            print("error i!")
            yield data3,labels
        else :
            yield data3,labels[__NOW__]
#        yield data3,labels

if __name__ == '__main__':
 #   img3=Image.open("0071.jpg")
 #   img = loadData2(number=1)
 #   img2 = convert2gray(img)
#    x_train, x_test, y_train, y_test=loadData(number=1)
    import matplotlib.pyplot as plt
    x_train, y_train = generateData(number=1)
    data,labels=generateGreyKerasData(number=5)
    a = generateKerasYieldData(1)
    x1, y1 = next(a)
    b = generateKerasGreyYieldData(1)
    x2, y2 = next(b)
    c = generateKerasYieldData2(1)
    x3, y3 = next(c)
    plt.imshow(x3[0])
    d = generateoneModelMultArray(1)
    x4, y4 = next(d)