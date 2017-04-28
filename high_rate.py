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
REDUCEDDIMENSION = 77
# Dense Noise
Epsilon = 1000
CLASSES = 38
SAMPLE_EACH_CLASS = 32
DOWNSAMPLE_COEFFICIENT = 1

CURRPATH = '/Users/LeonGong/Downloads/CroppedYale/'
STARTOFTRAIN = 25

# Find the sparse solution of SOCP
def SOCP(y, A, Epsilon):
    x_size = A.shape[1]
    err_size = A.shape[0]

    x = cvx.Variable(x_size)
    err = cvx.Variable(err_size)
    obj = cvx.Minimize(cvx.norm(x,1) + cvx.norm(err,1))

    constraints = [A*x - err == y]

    start = time.time()

    prob = cvx.Problem(obj, constraints)
    prob.solve(solver='SCS', max_iters = 10, verbose = False)
    # prob.solve(solver='ECOS_BB', max_iters = 10, verbose = False)

    finish = time.time()

    return x.value

def LoadImage():
    # Current Path
    # currPath = '/Volumes/Myspace/Courses/sparse/project/Code-Lecture1/FaceIntroDemo/CroppedYale/'
    currPath = CURRPATH
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
        for file_item in fileList[STARTOFTRAIN : STARTOFTRAIN+SAMPLE_EACH_CLASS]:
            img = Image.open(filePath+'/'+file_item)
            img = block_reduce(np.array(img), block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)
            
            # Normalization
            img = img/np.sqrt(np.sum(img**2))
            # img = (img-np.mean(img))/np.std(img)
            # print img

            # plt.imshow(img, cmap=plt.get_cmap('gray'))
            # plt.gca().axis('off')
            # plt.gcf().set_size_inches((5, 5))
            # plt.show()
       
            X_train.append(np.ndarray.flatten((np.array(img))))
            delta[i][pos] = 1
            pos += 1
    # print "Delta, shape:", np.array(delta).shape 
    # print "X_train, shape", np.array(X_train).shape
    return np.array(X_train).T, np.array(delta)

def concenIndex(estimateSolution, X_hat, k):
    max_i = np.max(np.sum(abs(estimateSolution), axis=0))
    # print max_i, np.sum(abs(X_hat))
    return (k*max_i/np.sum(abs(X_hat))-1)/(k-1)


def classify(test, X_train, delta):
    X_hat = SOCP(test, X_train, Epsilon)
    # print "X_hat shape", X_hat.shape
    X_hat = np.array(X_hat)
    
    estimateSolution = X_hat*delta.T
    # print estimateSolution.shape
    # print X_train.shape

    # SCI = concenIndex(estimateSolution, X_hat, 90)
    # print SCI

    testCopy = np.tile(test,(CLASSES,1))
    testCopy = np.transpose(testCopy)

    # print "test shape", test.shape, "test copy shape", testCopy.shape 

    mistake = (testCopy - np.dot(X_train, estimateSolution)) ** 2

    # print "Mistake matrix: ", mistake 
    # print "Mistake matrix sum: ", np.sum(mistake, axis=0)
    return np.argmin(np.sum(mistake, axis=0))

def testAlgo():
    X_train, delta = LoadImage()
    # Current Path
    # currPath = '/Volumes/Myspace/Courses/sparse/project/Code-Lecture1/FaceIntroDemo/CroppedYale/'
    currPath = CURRPATH
    # Load the first image

    os.chdir(currPath)
    classDirectory = glob.glob("yale*")


    # Generate the random face
    R_matrix = np.random.randn(REDUCEDDIMENSION, X_train.T.shape[1])
    X_train = np.dot(R_matrix, X_train)
    # print 'R matrix', R_matrix.shape
    # print 'X matrix', X_train.shape

    wrongSum = 0
    loop1 = 0
    # Load images from different classes
    for i in range(len(classDirectory)):
        # List all the class directories
        filePath = currPath + classDirectory[i]
        os.chdir(filePath)
        fileList = glob.glob("*.pgm")
        for loop in range(0, STARTOFTRAIN):
            tic = time.time()
            loop1 += 1
            img = Image.open(filePath+'/'+fileList[loop])
            # print fileList[loop] 
            img = block_reduce(np.array(img), block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)
            
            # Normalization
            img = img/np.sqrt(np.sum(img**2))
            # img = (img-np.mean(img))/np.std(img)
            img = np.ndarray.flatten((np.array(img)))

            # Calculate y_hat = Ry
            img = np.dot(R_matrix, img)
            
            predictLabel = classify(img.T, X_train, delta)
            if predictLabel != i:
                wrongSum += 1
                print(fileList[loop],' is wrongly classified')
            toc = time.time()
            # print("Predicting ", fileList[loop])
            print("Time used ", toc - tic)
            # predictLabel = classify(img.T, X_train, delta)
            # print 'The prediction result:', predictLabel
            # print 'Correct label: ', i
            wrongSum += (predictLabel != i)
    print(float(wrongSum)/loop1)


if __name__=='__main__':
    testAlgo()
