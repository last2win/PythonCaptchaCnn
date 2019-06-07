# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:47:12 2019

@author: peter


import sys
!{sys.executable} -m pip install captcha

from google.colab import drive

drive.mount('/content/gdrive')

!ls

!touch "/content/gdrive/My Drive/my-project3/training/test.txt"

!python "/content/gdrive/My Drive/my-project3/keras-grey-train.py" | tee  "/content/gdrive/My Drive/my-project3/train.log"
"""

import loadData
import time
from singleCaptchaGenerate import  CAPTCHA_LIST,CAPTCHA_HEIGHT,CAPTCHA_WIDTH
from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten,Activation
import keras
import os
'''
# 2. 导入库和模块
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# 3. 加载数据
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 4. 数据预处理
img_x, img_y = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 5. 定义模型结构
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(img_x, img_y, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 6. 编译
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. 训练
model.fit(x_train, y_train, batch_size=128, epochs=10)

# 8. 评估模型
score = model.evaluate(x_test, y_test)
print('acc', score[1])
'''

if __name__ == '__main__':
    
    time0 = time.time()
    
 #   print("keras-grey-train v2")
    
    print("keras-grey-train v3: add callback")

    print("start training")
    
    model = Sequential()
     
    model.add(Conv2D( 
            #filters：卷积核的数目（即输出的维度）
            filters = 32,
#kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。
#如为单个整数，则表示在各个空间维度的相同长度。             
                     kernel_size=(5,5), 
#activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。
#如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）             
                     activation='relu', 
                     
                     input_shape=(
       CAPTCHA_WIDTH,  CAPTCHA_HEIGHT, 1),
#padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。
#“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
    padding='same',     # Padding method
#strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。
#如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
     strides=1,    
    ))
    
    
 #   model.add(Activation('relu')) #激活层
    
    model.add(MaxPool2D(
 #pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，
 #如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
            pool_size=(2,2),
             padding='same', 
#strides：整数或长为2的整数tuple，或者None，步长值。
            strides=(2,2)))
    
    model.add(Conv2D(3, kernel_size=(1,1), 
       padding='same', activation='relu'))
    
    model.add(MaxPool2D(pool_size=(2,2), 
       padding='same', strides=(2,2)))
    
    model.add(Flatten()) #压平
    
    model.add(Dense(500, activation='relu')) #全连接
    
    model.add(Dense(40, activation='softmax'))#输出层
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model.summary()
#    checkpoint_path = "training/cp.ckpt"
    checkpoint_path = "/content/gdrive/My Drive/my-project3/training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)

    count=0
    flag=0
    while 1:
        count+=1
        if flag==1:
            x_train, y_train=loadData.generateGreyKerasData(number=10000)
            x_test, y_test=loadData.generateGreyKerasData(number=1000)
            model.fit(x_train,y_train,batch_size=64,epochs=10
                      ,validation_data=[x_test,y_test],callbacks = [cp_callback])
            if count % 2==0:
                print(datetime.now().strftime('%c'), 'count is ', count)            
            if count % 5 ==0:
                x_test, y_test=loadData.generateGreyKerasData(number=100)
                score, acc = model.evaluate(x_test, y_test, verbose=0)
                print('count is {}, Test: loss {}, acc {}'.format(count,score, acc))
                if acc >0.98:
                    break
        else:
            x_train, y_train=loadData.generateGreyKerasData(number=1000)
#            x_test, y_test=loadData.generateGreyKerasData(number=1000)
            model.fit(x_train, y_train)
            if count % 2==0:
                print(datetime.now().strftime('%c'), 'count is ', count)
      #      model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=[x_test,y_test])
            
            if count % 5 ==0:
                x_test, y_test=loadData.generateGreyKerasData(number=100)
                score, acc = model.evaluate(x_test, y_test, verbose=0)
                print('count is {}, Test: loss {}, acc {}'.format(count,score, acc))
                if acc >0.98:
                    break

    model.save('keras_model.h5')

    time1 = time.time()
    print("train : 总共花费 {0} s".format(time1-time0))