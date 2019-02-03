# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:04:57 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget
    
def MSE(W, b, x, y, reg):
    wxby = np.dot(x,W)+b-y
    return np.sum((1/7000)*np.dot(wxby.T,wxby) + 0.5*reg*np.dot(W.T,W))
    
def buildGraph(train_X, train_Y,test_X,test_Y,valid_X,valid_Y,beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=0.001):
    # Your implementation here
    training_epochs = 700
  
    W = tf.Variable(tf.random.truncated_normal(shape=(784,1),stddev=0.5), dtype=tf.float32, name="weight")
    b = tf.Variable(0.0, name="bias",dtype=tf.float32)
    
    x = tf.placeholder(tf.float32, shape=(784,1))
    y = tf.placeholder(tf.float32, shape=(1,1))
    #reg = tf.constant(0)   
    
    allx = tf.placeholder(tf.float32, shape=(3500,784))
    ally = tf.placeholder(tf.float32, shape=(3500,1))
    allpred=tf.matmul(allx,W)
    allcost=tf.reduce_sum(pow(allpred-ally,2))/7000

    validx = tf.placeholder(tf.float32,shape=(100,784))
    validy = tf.placeholder(tf.float32,shape=(100,1))
    validpred=tf.matmul(validx,W)
    validcost=tf.reduce_sum(pow(validpred-validy,2))/200

    testx = tf.placeholder(tf.float32,shape=(145,784))
    testy = tf.placeholder(tf.float32,shape=(145,1)) 
    testpred=tf.matmul(testx,W)
    testcost=tf.reduce_sum(pow(testpred-testy,2))/290
    
    y_pred = tf.matmul(tf.transpose(W),x) + b
    cost = tf.reduce_sum(pow(y_pred-y,2))/7000
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
   
    with tf.Session() as sess:
        
        sess.run(init)
        for epoch in range(training_epochs):
            train_X,train_Y = shuffle(train_X,train_Y)
            left,right=0,1
            while(left<7):
                for(xo,yo) in zip(train_X[left*500:right*500],train_Y[left*500:right*500]):
                    xo = np.reshape(xo,(784,1))
                    yo = np.reshape(yo,(1,1))
                    sess.run(optimizer,feed_dict={x:xo,y:yo})
                left+=1
                right+=1
            print("epoch ", epoch)
                
        print("Optimization Finished!")
        print("W=", sess.run(W), "b=", sess.run(b), '\n')

        training_cost = sess.run(allcost,feed_dict={allx:train_X, ally:train_Y})
        print("Training cost=", training_cost, '\n')
        validation_cost = sess.run(validcost,feed_dict={validx:valid_X, validy:valid_Y})
        print("Training cost=", validation_cost, '\n')
        testing_cost = sess.run(testcost,feed_dict={testx:test_X, testy:test_Y})
        print("Training cost=", testing_cost, '\n')
        
        
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape(3500,784)
validData = validData.reshape(100,784)
testData = testData.reshape(145,784)
buildGraph(train_X=trainData, train_Y=trainTarget,test_X=testData,test_Y=testTarget,valid_X=validData,valid_Y=validTarget,beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=0.01)

print("end")