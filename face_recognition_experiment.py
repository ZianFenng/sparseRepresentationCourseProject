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

# Parameters of the images in Cropped Yale
DIM_1 = 192
DIM_2 = 168

# Dense Noise
CLASSES = 38
SAMPLE_EACH_CLASS = 32
DOWNSAMPLE_COEFFICIENT = 4
Epsilon = 200

# Mode choose
DOWNSAMPE_MODE = 1
NORMALIZE_MODE = 2


# The path of the face image dataset
# CURRPATH = '/Users/LeonGong/Desktop/ELEN6886/CroppedYale/'
CURRPATH = '/Users/LeonGong/Downloads/CroppedYale/'

# Find the sparse solution of SOCP
def SOCP(y, A, Epsilon, slv = 'SCS', maxItr = 5, robust = True):
    # slv define the choice of solver, the default solver is SCS, which uses ADMM
    # the other option is 'CVXOPT', which uses interior point method
    print('A shape', A.shape)
    if robust:
        # Define the size of the variable
        x_size = A.shape[1]
        err_size = A.shape[0]

        # Define the variables, constraints and object of the optimization problem
        x = cvx.Variable(x_size)
        err = cvx.Variable(err_size)
        obj = cvx.Minimize(cvx.norm(x,1) + cvx.norm(err,1))
        constraints = [ A*x - err == y]

    else:
        # Define the size of the variable
        x_size = A.shape[1]
        
        # Define the variables, constraints and object of the optimization problem
        x = cvx.Variable(x_size)
        obj = cvx.Minimize(cvx.norm(x,1))
        constraints = [cvx.norm(A*x - y,2) < Epsilon]

    prob = cvx.Problem(obj, constraints)
    prob.solve(solver = slv, max_iters = maxItr, verbose = False)


    return x.value

def LoadImage(DOWNSAMPLE_COEFFICIENT, newSize = [12,10]):
    # The training set, matrix A in the equation
    X_train= []
    os.chdir(CURRPATH)
    classDirectory = glob.glob("yaleB*")
    # global CLASSES = len(classDirectory)
    
    # Record the image labels
    delta = [[0 for n in range(SAMPLE_EACH_CLASS*CLASSES)] for m in range(CLASSES)]
    pos = 0
    
    # Load images from different classes
    for i in range(len(classDirectory)):
        # List all the class directories
        filePath = CURRPATH + classDirectory[i]
        os.chdir(filePath)
        fileList = glob.glob("*.pgm")
        # Class i
        # Exculde 
        for file_item in fileList[:SAMPLE_EACH_CLASS]:
            img = Image.open(filePath+'/'+file_item)
            # [size1, size2] = np.array(img).shape
            # if size1 > 196:
            #     print(file_item)
            #     continue
            # print('Imgs original size', np.array(img).shape)

            # Down sample the image
            img = downSample(img, DOWNSAMPLE_COEFFICIENT, newSize = newSize)
            # print('Img size after down:', np.array(img).shape)
            # Normalization
            img = imageNormalize(img)
            # print('Img size after norm', np.array(img).shape)
       
            X_train.append(np.ndarray.flatten((np.array(img))))
            # print(X_train)
            delta[i][pos] = 1
            pos += 1

    print('A shape load image', np.array(X_train).shape)

    return np.array(X_train).T, np.array(delta)

def downSample(img, DOWNSAMPLE_COEFFICIENT, mode = DOWNSAMPE_MODE, newSize = [12,10]):
    npimg = np.array(img)
    if mode == 1:
        return block_reduce(npimg, block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)
    else:
        [newSize1, newSize2] = newSize
        [oldSize1, oldSize2] = npimg.shape
        step1 = int(oldSize1/newSize1)
        step2 = int(oldSize2/newSize2)
        newImg = np.zeros(newSize)
        for i in range(0, oldSize1 - step1, step1):
            for j in range(0, oldSize2 - step2, step2):
                newImg[int(i/step1)][int(j/step2)] = npimg[i][j]
        # print('Img size after down sample by method 2: ', newImg.size)
        return newImg

def imageNormalize(img, mode = 2):
    if mode == 1:
        return img/np.sum(img)
    else:
        return (img-np.mean(img))/np.std(img)

def concenIndex(estimateSolution, X_hat, k):
    max_i = np.max(np.sum(abs(estimateSolution), axis=0))
    # print max_i, np.sum(abs(X_hat))
    return (k*max_i/np.sum(abs(X_hat))-1)/(k-1)


def classify(test, X_train, delta, slv = 'SCS', maxItr = 10, robust = True, k = 10, tao = 0.45):
    X_hat = SOCP(test, X_train, Epsilon, slv, maxItr, robust)
    X_hat = np.array(X_hat)
    
    estimateSolution = X_hat*delta.T

    SCI = concenIndex(estimateSolution, X_hat, k)

    testCopy = np.tile(test,(CLASSES,1))
    testCopy = np.transpose(testCopy)

    mistake = (testCopy - np.dot(X_train, estimateSolution)) ** 2
    
    return np.argmin(np.sum(mistake, axis=0))
    # if SCI > tao:
    #     return np.argmin(np.sum(mistake, axis=0))
    # else:
    #     print('Reject! SCI equals to', SCI)
    #     return -1

def testAlgo(DOWNSAMPLE_COEFFICIENT, newSize = [12,10]):
    X_train, delta = LoadImage(DOWNSAMPLE_COEFFICIENT)

    os.chdir(CURRPATH)
    classDirectory = glob.glob("yale*")


    # Generate the random face
    # R_matrix = np.random.randn(REDUCEDDIMENSION, X_train.T.shape[1])
    # X_train = np.dot(R_matrix, X_train)
    # print 'R matrix', R_matrix.shape
    # print('X matrix', X_train.shape)

    wrongSum = 0
    loop1 = 0
    # Load images from different classes
    for i in range(len(classDirectory)):
        # List all the class directories
        
        filePath = CURRPATH + classDirectory[i]
        os.chdir(filePath)
        fileList = glob.glob("*.pgm")
        for loop in range(SAMPLE_EACH_CLASS, len(fileList)):
            start = time.time()
            
            img = Image.open(filePath+'/'+fileList[loop])
            # [size1, size2] = np.array(img).shape
            # if size1 > 196:
            #     continue
            loop1 += 1
            img = downSample(img, DOWNSAMPLE_COEFFICIENT, newSize = [12,10])
            # print('Down Img size:', np.array(img).shape)
            # Normalization
            img = imageNormalize(img)
            # print('Normalized Img size:', np.array(img).shape)
            img = np.ndarray.flatten((np.array(img)))
            # print('Flat Img size:', np.array(img).shape)
            # Calculate y_hat = Ry
            # img = np.dot(R_matrix, img)

            predictLabel = classify(img.T, X_train, delta)
            print('The prediction result:', predictLabel)
            print('Correct label: ', i)
            wrongSum += (predictLabel != i)
            finish = time.time()
            print('Time for a prediction', finish - start)
    print('Correctness rate: ', float(wrongSum)/loop1)

if __name__=='__main__':
    testAlgo(DOWNSAMPLE_COEFFICIENT)
