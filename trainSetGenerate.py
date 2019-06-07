# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:21:45 2019

@author: peter
"""

from singleCaptchaGenerate import generateCaptchaText


generateNumber = 10**2

if __name__ == '__main__':
    for i in range(0, generateNumber):
        text = generateCaptchaText(save=True)
        if i % 100 ==0:
            print("count is:  "+str(i)+"    captcha text is: "+text)
