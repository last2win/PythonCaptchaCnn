# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:21:20 2019

@author: peter
"""

import loadData

import os
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn import metrics

if __name__ == '__main__':
    time0 = time.time()

    x_train, x_test, y_train, y_test = loadData.loadData(number=9000)

    print("start training")
    
    model= MLPClassifier(solver='adam', alpha=1e-5, random_state=1
                         ,hidden_layer_sizes =(200,100)
                         )
    model.fit(x_train, y_train)
    
    
 #   joblib.dump(model, "my-mlp-module6.m")
    
    predicted = model.predict(x_test)
    rightCount=0
    for i in range(0,len(predicted)):
        flag=0
        for j in range(0,40):
            if predicted[i][j] != y_test[i][j]:
                flag=1
                break
        if flag==1:
            print("error!")
        else:
            rightCount+=1
            print("right!")
    print("total right count is:   ",rightCount)
 #   print("Accuracy: {:5.2f}%".format(100*metrics.accuracy_score(predicted, y_test)))
#    print("Accuracy :", metrics.accuracy_score(predicted, y_test))
    time1 = time.time()
    print("mlp: 总共花费 {0} s".format( time1-time0))