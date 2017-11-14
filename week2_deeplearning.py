#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:35:30 2017

@author: mayank.singh
"""
## Vectorization
import numpy as np

a = np.array([1,2,3,4])
a

import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)
c
# time for vectorized product
%time c = np.dot(a,b)

# time for for loop
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

(toc - tic)*1000

# broadcasting in python

A = np.array([[56.0, 0.0, 4.4, 68.0], 
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
    
cal = A.sum(axis = 0)
cal

percentage = 100*A/cal.reshape(1,4) # python broadcasting

## NUMPY VECTORS ##

import numpy as np
a = np.random.randn(5) # 5 random variable

a.shape # Rank 1 array, neither a row vector, nor a column vector, don't use these

a.T # a transpose same as a

np.dot(a, a.T) #will just give a number, not a matrix

## recommended not to use datastructures where the shape is (x, )

## instead set a to be:

# there are 2 square brackets instead of one square bracket in a matrix
a = np.random.rand(5, 1) # column vector

a.T 

np.dot(a, a.T)

b = np.random.rand(1, 5) # row vector

assert(a.shape == (5,1)) # assert the shape of the array to double check

# can reshape to fix the array
a.reshape((1,5))


## 
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
    

np.sum(x, axis = 1, keepdims = True)

x = 2
y = 3

np.square


w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])


def sigmoid(z):
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1 + (1/np.exp(z)))
    ### END CODE HERE ###
    
    return s


A = sigmoid(np.dot(w.T,X) + b)

a = sigmoid(w*X + b)

dz = A - Y
X.shape
m = X.shape[1]

dw = (1/m)*np.dot(X,dz.T)

dw.shape

(1/m)*np.dot(np.log(A), Y.T) + np.dot(np.log((1-A)), (1-Y).T)


np.log(1-A.T)

(Y*np.log(A.T) + (1 - Y.T)*np.log(1 - A)


np.dot(Y, np.log(A.T)) + np.dot((1-Y), np.log((1-A).T))


np.dot(w.T,X)

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])

A = sigmoid(np.dot(w.T,X) + b)

np.round(A)

x, y = np.zeros([10,1]), np.zeros([10,1])
v, x = np.zeros([1,0], [1,0], dtypes = float)