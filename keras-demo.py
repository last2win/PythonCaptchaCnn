# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:54:13 2019

@author: peter
"""

import keras

from keras.datasets import mnist 

(x_train, y_train), (x_test,y_test) =mnist.load_data()

x_train=x_train.reshape(-1,28,28,1)

x_test=x_test.reshape(-1,28,28,1)

x_train=x_train/255.

x_test=x_test/255.

y_train=keras.utils.to_categorical(y_train)

y_test=keras.utils.to_categorical(y_test)

from keras.layers import Conv2D,MaxPool2D,Dense,Flatten

from keras.models import Sequential 

lenet=Sequential()

lenet.add(Conv2D(6,kernel_size=3,strides=1,padding='same',input_shape=(28, 28, 1)))

lenet.add(MaxPool2D(pool_size=2,strides=2))

lenet.add(Conv2D(16,kernel_size=5,strides=1,padding='valid'))

lenet.add(MaxPool2D(pool_size=2,strides=2))

lenet.add(Flatten())

lenet.add(Dense(120))

lenet.add(Dense(84))

lenet.add(Dense(10,activation='softmax'))

lenet.compile('sgd',loss='categorical_crossentropy',metrics=['accuracy'])


lenet.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=[x_test,y_test])

#lenet.save('myletnet.h5')









