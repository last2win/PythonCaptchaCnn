# -*- coding: utf-8 -*-

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from keras.utils import Sequence
import os
import time
import keras
from keras.layers import Dense, Flatten, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, Model
print("version 1")
print("version 2:change picture width")
print("version 3:add TensorBoard")
print("version 4:change TensorBoard")
print("version 5:small steps_per_epoch")

"""
Created on Mon Jun  3 16:38:53 2019

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

!cat "/content/gdrive/My Drive/my-project3/other-keras-4.py"

!python "/content/gdrive/My Drive/my-project3/other-keras-4.py" 2>&1 | tee  "/content/gdrive/My Drive/my-project3/keras_5/$(date).log"



tensorboard --logdir="C:/Users/peter/Google 云端硬盘/my-project3/keras/logs"


"""


characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
#characters = string.digits
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


def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


a = gen(1)
X, y = next(a)
'''
plt.imshow(X[0])
plt.title(decode(y))
'''




input_tensor = Input((height, width, 3))

x = input_tensor

for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d' % (i+1))(x)
     for i in range(4)]

model = Model(input=input_tensor, output=x)


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


print(model.summary())


checkpoint_path = "/content/gdrive/My Drive/my-project3/keras_5/cp.ckpt"
#checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=False,
                                              verbose=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=5, verbose=0, mode='auto',
                                               baseline=None, restore_best_weights=False)


TensorBoardcallback = keras.callbacks.TensorBoard(
    log_dir='/content/gdrive/My Drive/my-project3/keras_5/logs',
    histogram_freq=0, batch_size=32,
    write_graph=True, write_grads=False, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None, update_freq='batch'
)

# early_stopping = keras.callbacks.EarlyStopping(
#    monitor='val_loss', patience=5, min_delta=0.01)
'''
model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5, 
                    nb_worker=2, pickle_safe=True, 
                    validation_data=gen(), nb_val_samples=1280)

#steps_per_epoch = samples_per_epoch/batch_size
'''
model.fit_generator(gen(),
                    # steps_per_epoch=51200,  # 一轮多少个
                    # nb_epoch=5,  # 训练 nb_epoch 轮
                    steps_per_epoch=51200 ,  # 一轮多少个
                    nb_epoch=5,
                    workers=2,  use_multiprocessing=False,  # 单线程
                    #            nb_worker=2, pickle_safe=True,
                    # validation_data: 它可以是以下之一： 验证数据的生成器或 Sequence 实例
                    validation_data=gen(),
                    validation_steps=1280,  # 验证样本数
                    callbacks=[cp_callback,
                               TensorBoardcallback, early_stopping]
                    )

model.save(
    '/content/gdrive/My Drive/my-project3/keras_5/keras_number_ascii_model.h5')
time1 = time.time()
print("train : 总共花费 {0} s".format(time1-time0))
