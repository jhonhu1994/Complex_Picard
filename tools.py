#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:04:29 2023

@author: jhonhu
"""

# Import libraries
import numpy as np
import numexpr as ne
import math

def whitening(Y, mode, n_components):
    ''' Whitens the data Y using sphering or pca '''
    R = np.dot(Y, Y.T) / Y.shape[1]
    U, D, _ = np.linalg.svd(R)
    if mode == 'pca':
        W = U.T / np.sqrt(D)[:, None]
        W = W[0:n_components,:]
        Z = np.dot(W, Y)
    elif mode == 'sph':
        W = np.dot(U, U.T / np.sqrt(D)[:, None])
        Z = np.dot(W, Y)
    return Z, W


def laplace_rnd(N, T, mu=0., sigma=1./math.sqrt(2)):
    ''' Generates random Laplacian numbers '''
    u = np.random.rand(N, T) - 0.5
    y = mu - sigma*np.sign(u)*np.log(1. - 2.*np.abs(u))
    return y


def score(Y):
    ''' Returns the score function evaluated for each sample '''
    return ne.evaluate('tanh(Y / 2.)')


def score_der(psiY):
    ''' Returns the derivative of the score '''
    return ne.evaluate('(- psiY ** 2 + 1.) / 2.')


def loss(Y, W):
    ''' Computes the loss function for (Y, W) '''
    T = Y.shape[1]
    log_det = math.log(np.abs(np.linalg.det(W)))
    # logcoshY = np.abs(C) + np.log1p(np.exp(-np.abs(C)))
    logcoshY = np.sum((ne.evaluate('abs(Y) + 2. * log1p(exp(-abs(Y)))')))
    return - log_det + logcoshY.real / float(T)


def line_search(Y, W, direction, current_loss=None, ls_tries=10):
    '''
    Performs a simple backtracking linesearch in the direction "direction".
    Does n_ls_tries attempts before exiting.
    '''
    N = Y.shape[0]
    projected_W = np.dot(direction, W)
    alpha = 1
    if current_loss is None:
        current_loss = loss(Y,W)
    for im in range(ls_tries):
        new_Y = np.matmul(np.eye(N) + alpha*direction, Y)
        new_W = W + alpha*projected_W
        new_loss = loss(new_Y, new_W)      
        if new_loss < current_loss:
            converged = True
            rel_step = alpha
            break       
        alpha /= 2.0       
    else:
        converged = False
        rel_step = 0
        
    return converged, new_Y, new_W, new_loss, rel_step
    

