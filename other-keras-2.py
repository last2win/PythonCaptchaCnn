# -*- coding: utf-8 -*-
from keras.layers import Dense, Flatten, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, Model
import keras
import time
import os
from keras.utils import Sequence
import string
import random
import numpy as np
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
print("version 1")
print("version 2:change picture width")
print("version 3:add TensorBoard")
print("version 4:change TensorBoard")
print("version 5:small steps_per_epoch")

"""
Created on Mon Jun  3 19:25:04 2019

@author: peter

4/XQH60KPVAUl10IlrM1k5_-L3LHtcHnFnmVh6BOTeqaFnvKzzhtdIrL8

import sys
!{sys.executable} -m pip install captcha

from google.colab import drive

drive.mount('/content/gdrive')

!ls

!cat "/content/gdrive/My Drive/my-project3/other-keras-2.py"


!python "/content/gdrive/My Drive/my-project3/other-keras-2.py" 2>&1 | tee  "/content/gdrive/My Drive/my-project3/keras_2/$(date)-other-keras-2.log"



tensorboard --logdir="C:/Users/peter/Google 云端硬盘/my-project3/keras_2/logs2"

"""


#characters = string.digits + string.ascii_uppercase
characters = string.digits
# print(characters)

width, height, n_len, n_class = 180, 80, 4, len(characters)

#width, height, n_len, n_class = 160, 60, 4, len(characters)

'''
generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)

plt.imshow(img)
plt.title(random_str)
'''

time0 = time.time()


class test(Sequence):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


#CAPTCHA_LIST =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CAPTCHA_LIST = list(characters)
CAPTCHA_LENGTH = len(characters)


def text2vec(text):
    """
    text to one-hot vector
    :param text: source text
    :return: np array
    """
    if len(text) > n_len:
        raise ValueError('验证码长度超过'+str(n_len)+'个字符')
    vector = np.zeros(n_len * n_class, dtype=np.int8)
 #   vector = np.zeros(CAPTCHA_LENGTH * VOCAB_LENGTH)
    for i, c in enumerate(text):
        index = i * n_class + CAPTCHA_LIST.index(c)
        vector[index] = 1
    return vector


a = text2vec("0000")


def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len * n_class), dtype=np.uint8)
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, c in enumerate(random_str):
                index = j * n_class + CAPTCHA_LIST.index(c)
                y[i][index] = 1
  #          for j, ch in enumerate(random_str):
   #             y[i]
 #               y[j][i, :] = 0
 #               y[j][i, characters.find(ch)] = 1
        yield X, y


def vec2text(vector):
    """
    vector to captcha text
    :param vector: np array
    :return: text
    """
    vector = vector[0]
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
 #   vector = np.reshape(vector, [CAPTCHA_LENGTH, -1])
    text = ''
    for i, c in enumerate(vector):
        if c == 1:
            text += CAPTCHA_LIST[i % CAPTCHA_LENGTH]
 #   for item in vector:
 #       text += CAPTCHA_LIST[np.argmax(item)]
    return text


a = gen(1)
X, y = next(a)

'''
vec2text(y)

plt.imshow(X[0])
plt.title(vec2text(y))


plt.imshow(X[0])
plt.title(vec2text(y))
plt.title(decode(y))
'''

'''
input_tensor = Input((height, width, 3))

x = input_tensor

for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
'''
'''
x = Convolution2D(32*2**1, 3, 3, activation='relu')(x)
x = Convolution2D(32*2**1, 3, 3, activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
'''


model = Sequential()

for i in range(4):
    model.add(Conv2D(
        # filters：卷积核的数目（即输出的维度）
        filters=32*2**i,
        # kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。
        # 如为单个整数，则表示在各个空间维度的相同长度。
        kernel_size=(3, 3),
        # activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。
        # 如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
        activation='relu',
        input_shape=(height,   width, 3)))

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
model.add(Dense(n_class*n_len, activation='softmax'))  # 全连接

'''
x = Flatten()(x)
x = Dropout(0.25)(x)


#x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
x = Dense(n_class*n_len, activation='softmax', name='c1')(x)

model = Model(input=input_tensor, output=x)
'''

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


print(model.summary())


checkpoint_path = "/content/gdrive/My Drive/my-project3/keras_2/check.ckpt"
#checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=False,
                                              verbose=1)

TensorBoardcallback = keras.callbacks.TensorBoard(
    log_dir='/content/gdrive/My Drive/my-project3/keras_2/logs2',
    histogram_freq=0, batch_size=32,
    write_graph=True, write_grads=False, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None, update_freq=500

)
'''
model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5, 
                    nb_worker=2, pickle_safe=True, 
                    validation_data=gen(), nb_val_samples=1280)

#steps_per_epoch = samples_per_epoch/batch_size
'''
model.fit_generator(gen(),
                    # steps_per_epoch=51200,  # 一轮多少个
                    # nb_epoch=5,  # 训练 nb_epoch 轮
                    steps_per_epoch=51200 // 10,  # 一轮多少个
                    nb_epoch=5*10,
                    workers=1,  use_multiprocessing=False,  # 单线程
                    #            nb_worker=2, pickle_safe=True,
                    # validation_data: 它可以是以下之一： 验证数据的生成器或 Sequence 实例
                    validation_data=gen(),
                    validation_steps=1280,  # 验证样本数
                    callbacks=[cp_callback, TensorBoardcallback]
                    )

model.save('/content/gdrive/My Drive/my-project3/keras_2/keras_number_model.h5')
time1 = time.time()
print("train : 总共花费 {0} s".format(time1-time0))
