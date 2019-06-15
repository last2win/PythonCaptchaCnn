import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_LIST = NUMBER
n_class=VOCAB_LENGTH = len(CAPTCHA_LIST)
n_len=CAPTCHA_LENGTH = 4            # 验证码长度
CAPTCHA_HEIGHT = 60        # 验证码高度
CAPTCHA_WIDTH = 160        # 验证码宽度

CAPTCHA_HEIGHT = 80        # 验证码高度
CAPTCHA_WIDTH = 180        # 验证码宽度

# CAPTCHA_LIST=["我","你","好","的","是","否"]


def randomCaptchaText(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LENGTH):
    """
    随机生成定长字符串
    :param char_set: 备选字符串列表
    :param captcha_size: 字符串长度
    :return: 字符串
    """
#    print(char_set)
    captcha_text = [random.choice(char_set) for _ in range(captcha_size)]
    return ''.join(captcha_text)


def generateCaptchaText(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT, saveDir="./img/", save=False):
    """
    生成随机验证码
    :param width: 验证码图片宽度
    :param height: 验证码图片高度
    :param save: 是否保存（None）
    :return: 验证码字符串，验证码图像np数组
    """
    image = ImageCaptcha(width=width, height=height)
    # 验证码文本
    captchaText = randomCaptchaText(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LENGTH)
    captcha = image.generate(captchaText)
    # 保存
    if save:
        image.write(captchaText, './img/' + captchaText + '.jpg')
#    captcha_image = Image.open(captcha)
    # 转化为np数组
 #   captcha_image = np.array(captcha_image)
    return captchaText


def generateCaptchaTextAndImage(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT, save=False):
    """
    生成随机验证码
    :param width: 验证码图片宽度
    :param height: 验证码图片高度
    :param save: 是否保存（None）
    :return: 验证码字符串，验证码图像np数组
    """
    image = ImageCaptcha(width=width, height=height)
    # 验证码文本
    captchaText = randomCaptchaText(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LENGTH)
    captcha = image.generate(captchaText)

    captcha_image = Image.open(captcha)
    # 转化为np数组
    captcha_image = np.array(captcha_image)
    return captchaText, captcha_image

import os
import imageio
import numpy as np
#from sklearn.model_selection import train_test_split

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
    data2=data1.reshape(data1.shape[0], CAPTCHA_HEIGHT,CAPTCHA_WIDTH,1)
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
        data3=data1.astype('uint8')
  #      data3/=255
        yield data3,labels



def oneModelOneArray(batch_size=32):
    global CAPTCHA_LIST ,n_class
    X = np.zeros((batch_size, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 3), dtype=np.uint8)
    y = np.zeros((batch_size, 40), dtype=np.uint8)
#    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            captchaText, captcha_image = generateCaptchaTextAndImage(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT)
#            random_str = captchaText
            X[i] = captcha_image
            for j, c in enumerate(captchaText):
                index = j * n_class + CAPTCHA_LIST.index(c)
                y[i][index] = 1
  #          for j, ch in enumerate(random_str):
   #             y[i]
 #               y[j][i, :] = 0
 #               y[j][i, characters.find(ch)] = 1
        yield X, y



def oneModelOneArray3(batch_size=32):
    global CAPTCHA_HEIGHT,CAPTCHA_WIDTH
    while True:
        data = []
        labels = []
        for i in range(0, batch_size):
     #       if i % 100 == 0 and i > 0:
     #           print("load count is ", i)
            captchaText, captcha_image = generateCaptchaTextAndImage(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT)
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

def oneModelOneArray2(batch_size=32):
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
 #           data.append(captcha_image)
            labels.append(text2vec(captchaText))
     #   return data,labels
        data1=np.array(data)
        labels=np.array(labels)
        data2=data1.reshape(data1.shape[0], CAPTCHA_WIDTH, CAPTCHA_HEIGHT,1)
        data3=data1.astype('float32')
#        data3/=255
        yield data3,labels
from keras.layers import Dense, Flatten, Activation, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten, Activation, Flatten


import os
import tensorflow as tf
import keras
import time
import numpy as np
# loadData.loadData()

#beforePath = "/content/gdrive/My Drive/my-project3/oneModelOneArray"
beforePath = "./oneModelOneArray"
beforePath = beforePath+"/number"



def best_model():
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
            input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 3)))

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
    model.add(Dense(40, activation='softmax'))  # 全连接

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

def create_model():
    model = keras.models.Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
               input_shape=(CAPTCHA_HEIGHT,   CAPTCHA_WIDTH, 1), padding="same", strides=(1, 1)),
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
        keras.layers.Dense(10*4, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss=tf.keras.losses.sparse_categorical_crossentropy,
    #               metrics=['accuracy'])
    return model
if 1 == 1:
        time0 = time.time()
        print("start training")
        beforePath = beforePath
        checkpoint_path = beforePath + '/cp.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=False,
                                                      verbose=1)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=5, verbose=0, mode='auto',
                                                       baseline=None)

        TensorBoardcallback = keras.callbacks.TensorBoard(
            log_dir=beforePath + '/logs/',
            histogram_freq=0, batch_size=32,
            write_graph=True, write_grads=False, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None, update_freq='batch'
        )
        model = best_model()
        print(model.summary())
#        a=loadData.oneModelOneArray(1000)
#        x_train, y_train=next(a)
#        model.fit(x_train,y_train,batch_size=64,epochs=10 )
        model.fit_generator(loadData.oneModelOneArray(),
                            # steps_per_epoch=51200,  # 一轮多少个
                            # nb_epoch=5,  # 训练 nb_epoch 轮
                            steps_per_epoch=5120,  # 一轮多少个
                            nb_epoch=2*4,
                            workers=1,  use_multiprocessing=False,  # 单线程
                            #            nb_worker=2, pickle_safe=True,
                            # validation_data: 它可以是以下之一： 验证数据的生成器或 Sequence 实例
                            validation_data=loadData.oneModelOneArray(),
                            validation_steps=1280,  # 验证样本数
                            callbacks=[cp_callback,
                                       TensorBoardcallback, early_stopping], verbose=1
                            )
        model.save(beforePath + '/model.h5')
 #       model.predict
        time1 = time.time()
        print("train : 总共花费 {0} s".format(time1-time0))