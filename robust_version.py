#!/usr/bin/python
import os
from os import listdir
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from skimage.measure import block_reduce
from PIL import Image

import random
import time

# Parameters of the images
DIM_1 = 192
DIM_2 = 168
# Dense Noise
Epsilon = 1000
CLASSES = 13
SAMPLE_EACH_CLASS = 62
DOWNSAMPLE_COEFFICIENT = 2

# Find the sparse solution of SOCP
def SOCP(y, A, Epsilon):

    x_size = A.shape[1]
    print("x_size: ", x_size)

    err_size = A.shape[0]
    print("err_size:", err_size);

    x = cvx.Variable(x_size)
    err = cvx.Variable(err_size)
    # obj = cvx.norm(x,1) + cvx.norm(err,1)
    # obj = cvx.Minimize(obj)
    obj = cvx.Minimize(cvx.norm(x,1) + cvx.norm(err,1))

    constraints = [cvx.norm(y - A*x - err,2) < 1]

    start = time.time()

    prob = cvx.Problem(obj, constraints)
    prob.solve(solver='SCS', max_iters = 5, verbose = True)
    # prob.solve()

    finish = time.time()

    print("Time took: ", finish - start)

    print("Status:", prob.status)

    print("Optimal value with SCS", prob.value)


    return x.value

def LoadImage():
    # Current Path
    currPath = '/Users/LeonGong/Desktop/ELEN6886/CroppedYale/'
    # Load the first image
    X_train= []

    os.chdir(currPath)
    classDirectory = glob.glob("yale*")

    # Record the image labels
    delta = [[0 for n in range(SAMPLE_EACH_CLASS*CLASSES)] for m in range(CLASSES)]

    pos = 0
    # Load images from different classes
    for i in range(len(classDirectory)):
        # List all the class directories
        filePath = currPath + classDirectory[i]
        os.chdir(filePath)
        fileList = glob.glob("*.pgm")
        # Class i
        # Exculde 
        for file_item in fileList[2:]:
            img = Image.open(filePath+'/'+file_item)
            img = block_reduce(np.array(img), block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)

            # plt.imshow(img, cmap=plt.get_cmap('gray'))
            # plt.gca().axis('off')
            # plt.gcf().set_size_inches((5, 5))
            # plt.show()
       
            X_train.append(np.ndarray.flatten((np.array(img))))
            delta[i][pos] = 1
            pos += 1
    print("Delta, shape:", np.array(delta).shape)
    print("X_train, shape", np.array(X_train).shape) 
    return np.array(X_train).T, np.array(delta)

def classify(test, X_train, delta):
    X_hat = SOCP(test, X_train, Epsilon)
    print("X_hat shape", X_hat.shape)
    X_hat = np.array(X_hat)
    # for i in xrange(CLASSES):
    #     delta_i = np.zeros((CLASSES*SAMPLE_EACH_CLASS))
    #     delta_i[64*i:64*(i+1)] = X_hat[64*i:64*(i+1)]
    # print test.shape
    # test = np.array(test)
    print(delta.shape)
    residual = X_hat*delta.T
    # print X_train.shape
    print("residual shape", residual.shape)

    testCopy = np.tile(test,(CLASSES,1))
    testCopy = np.transpose(testCopy)

    print("test shape", test.shape, "test copy shape", testCopy.shape)

    mistake = (testCopy - np.dot(X_train, residual)) ** 2

    print("Mistake matrix: ", mistake)
    print("Mistake matrix sum: ", np.sum(mistake, axis=0))
    return np.argmin(np.sum(mistake, axis=0))

def testAlgo():
    X_train, delta = LoadImage()
    # Current Path
    currPath = '/Users/LeonGong/Desktop/ELEN6886/CroppedYale/'
    # Load the first image

    os.chdir(currPath)
    classDirectory = glob.glob("yale*")

    # Load images from different classes
    for i in range(len(classDirectory)):
        # List all the class directories
        filePath = currPath + classDirectory[i]
        os.chdir(filePath)
        fileList = glob.glob("*.pgm")
        # Class i
        # Exculde 
        img = Image.open(filePath+'/'+fileList[1])
        print(fileList[1])
        img = block_reduce(np.array(img), block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)
        img = np.ndarray.flatten((np.array(img)))
        print(classify(img.T, X_train, delta))
    # print(classify(X_train[:,0], X_train, delta))

if __name__=='__main__':
    print(testAlgo())
    





