# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import sklearn.model_selection
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

# helper function
# did not end up using this
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N = len(x_train)
    D = len(x_train[0])
    deltaX = x_train-test_datum.transpose()
    normSquares = -1*(np.sum(np.abs(deltaX)**2,axis=-1))/(2*tau**2)
    expNormalizedNormSquares = np.exp(normSquares - max(normSquares))
    A = np.diag(expNormalizedNormSquares/sum(expNormalizedNormSquares))
    # Recall: dC/dw = -(Y-Xw)^T AX + lam*w^T
    # Use a linear solver to solve this
    AA = np.dot(np.dot(x_train.transpose(), A), x_train) + lam*np.identity(D)
    bb = np.dot(np.dot(x_train.transpose(), A), y_train)
    wStar = np.linalg.solve(AA, bb)
    return np.dot(test_datum.transpose(), wStar)

# computes a single loss given (x,t) by training on x_train, y_train
def loss(x_train, y_train, x, t, tau, lam=1e-5):
    yhat = LRLS(x, x_train, y_train, tau, lam)
    return (0.5)*(yhat-t)**2

# compute average loss (or the cost) on (x_test, t_test) by training on (x_train, y_train)
def cost(x_train, y_train, x_test, t_test, tau, lam=1e-5):
    N = len(x_test)
    cost = 0
    for i in range(0, N):
        cost += (1./N)*loss(x_train, y_train, x_test[i], t_test[i], tau, lam)
    return cost


def run_validation(x,y,taus,val_frac,lam=1e-5):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(x, y, test_size=val_frac)
    num_iter = len(taus)
    training_cost = [0 for i in range(0, num_iter)]
    validation_cost = [0 for i in range(0, num_iter)]
    for i in range(0, num_iter):
        print("Iteration: %d" % i)
        training_cost[i] = cost(x_train, y_train, x_train, y_train, taus[i], lam)
        validation_cost[i] = cost(x_train, y_train, x_validate, y_validate, taus[i], lam)
    return (training_cost, validation_cost)

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.linspace(1,1000,300)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, train_losses)
    plt.semilogx(taus, test_losses)
    plt.legend(['training loss', 'test loss'])
    plt.show()
