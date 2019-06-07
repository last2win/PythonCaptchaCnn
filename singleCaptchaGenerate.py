# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:04:50 2019

@author: peter
import sys
!{sys.executable} -m pip install captcha
"""

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
    captchaText = randomCaptchaText()
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
    captchaText = randomCaptchaText()
    captcha = image.generate(captchaText)

    captcha_image = Image.open(captcha)
    # 转化为np数组
    captcha_image = np.array(captcha_image)
    return captchaText, captcha_image


if __name__ == '__main__':
    text = generateCaptchaText(save=True)
    print("captcha text is: "+text)
