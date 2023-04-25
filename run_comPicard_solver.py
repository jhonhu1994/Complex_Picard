#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:12:50 2023

@author: jhonhu
"""

# Import libraries
import numpy as np
# import matplotlib.pyplot as plt
import copy
from tools import loss, score, score_der, line_search

def picard_complex(X, method, precon, maxiter = 150, tol = 0.000001, lambda_min = 0.01, ls_tries = 10, verbose = 1):
    '''
    Runs the Picard algorithm (the complex version)
    
    The original algorithm (the real version) is detailed in:

      Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
      Faster independent component analysis by preconditioning with Hessian
      approximations
      IEEE Transactions on Signal Processing, 2018
    
    Parameters
    ----------
    X : array, shape (N,T)
        Matrix containing the signals that have to be unmixed. N is the
        number of signals, T is the number of samples. X has to be centered.
    method : string
        DESCRIPTION.
    precon : 1 or 2
        Chooses which Hessian approximation is used.
        1 -> H1
        2 -> H2
        H2 is more costly to compute but can greatly accelerate convergence
        (See the paper for details).
    maxiter : int, optional
        Maximal number of iterations for the algorithm. The default is 150.
    tol : float, optional
        tolerance for the stopping criterion. Iterations stop when the norm
        of the gradient gets smaller than tol. The default is 0.000001.
    lambda_min : float, optional
        Constant used to regularize the Hessian approximations. The
        eigenvalues of the approximation that are below lambda_min are
        shifted to lambda_min. The default is 0.01.
    ls_tries : int, optional
        Number of tries allowed for the backtracking line-search. When that
        number is exceeded, the direction is thrown away and the gradient
        is used instead. The default is 10.
    verbose : boolean, optional
        If true, prints the informations about the algorithm. 
        The default is True.

    Returns
    -------
    Y : array, shape (N, T)
        The estimated source matrix
    W : array, shape (N, N)
        The estimated unmixing matrix, such that Y = WX.
    LossValue : vector, shape (maxiter, 1)
        The value of loss function at each iteration
    NormGradient : vector, shape (maxiter, 1)
        The l_inf norm of Gradient at each iteration
        
    Authors: Jhon Hu <hxz123@mail.ustc.edu.cn>

    '''
    
    # Init
    N, T = X.shape
    if N > T:
        print('error: There are more signals than samples')
    W = np.eye(N)
    Y = np.matmul(W,X)
    current_loss = loss(Y, W)
    
    # Iterate
    LossValue = [];
    NormGradient = [];
    for n_top in range(maxiter):
        absY = np.abs(Y)
        psiY = score(absY)
        expY = Y/absY
        # Compute the relative gradient
        G = 0.5*np.inner(psiY*expY, Y.conjugate()) / float(T) - 0.5*np.eye(N)
        
        LossValue.append(current_loss)
        G_norm = np.linalg.norm(G.ravel(), ord=np.inf)
        NormGradient.append(G_norm)
        
        # Stopping criterion
        if G_norm < tol:
            break
        
        # Find the updating direction
        if method == 'Gradient':
            direction = -G
        elif method == 'partNewton':
            direction = - solve_partHessian(G, absY, psiY, precon, lambda_min)
        elif method == 'fullNewton':
            mu = 1./2.2
            direction = - solve_fullHessian(G, Y, psiY, absY, expY, precon, lambda_min, mu)
        else:
            print('Wrong updating mode')
            
        # Do a line serach in the computed direction
        converged, new_Y, new_W, new_loss, step = line_search(Y, W, direction, current_loss, ls_tries)
        if not converged:
            if verbose:
                print('line search failed, falling back to gradient.')
                
            direction = -G
            _, new_Y, new_W, new_loss, step = line_search(Y, W, direction, current_loss, ls_tries)
            
        
        Y = copy.deepcopy(new_Y)
        W = copy.deepcopy(new_W)
        current_loss = new_loss
        if verbose:
            print('iteration '+str(n_top)+', gradient norm = '+str(G_norm))
            
            
    return Y, W, LossValue, NormGradient


def solve_partHessian(G, absY, psiY, precon, lambda_min):
    N, T = absY.shape
    psidY = score_der(psiY)
    psiY_normalized = psiY/absY
    absY_squared = absY**2
    if precon == 2:
        a = 0.25*np.inner(psidY+psiY_normalized, absY_squared)/float(T)
        tmp = np.min(a)
        a = a - float(tmp < 0)*(tmp - lambda_min)
    elif precon == 1:
        sigma2 = np.mean(absY_squared, axis=1)
        psidY_mean = np.mean(psidY + psiY_normalized, axis=1);
        a = np.matmul(psidY_mean[:, None], sigma2[None, :])
        diagonal_term = np.mean(absY_squared * (psidY + psiY_normalized), axis=1)
        a[np.diag_indices_from(a)] = diagonal_term
        a = 0.25*a
        tmp = np.min(a)
        a = a - float(tmp < 0)*(tmp - lambda_min)
    else:
        print('Wrong updating mode')
        
    # Invert the transform
    return G/a

    

def solve_fullHessian(G, Y, psiY, absY, expY, precon, lambda_min=0., mu=0.45):
    N, T = Y.shape
    # Compute the derivative of the score
    psidY = score_der(psiY)
    psiY_normalized = psiY/absY
    Y_squared = Y**2
    absY_squared = absY**2
    expY_squared = expY**2
    if precon == 2:
        a = 0.25*np.inner(psidY+psiY_normalized, absY_squared)/float(T)
        b = 0.25*np.inner((psidY-psiY_normalized)*expY_squared, Y_squared.conjugate())/float(T)
    elif precon == 1:
        sigma2 = np.mean(absY_squared, axis=1)
        psidY_mean = np.mean(psidY + psiY_normalized, axis=1);
        a = np.matmul(psidY_mean[:, None], sigma2[None, :])
        diagonal_term = np.mean(absY_squared * (psidY + psiY_normalized), axis=1)
        a[np.diag_indices_from(a)] = diagonal_term
        a = 0.25*a
        
        sigma2 = np.mean(Y_squared.conjugate(), axis=1)
        psidY_mean = np.mean((psidY-psiY_normalized)*expY_squared, axis=1);
        b = np.matmul(psidY_mean[:, None], sigma2[None, :])
        diagonal_term = np.mean(Y_squared * (psidY - psiY_normalized) * expY_squared, axis=1)
        b[np.diag_indices_from(a)] = diagonal_term
        b = 0.25*b
    else:
        print('Wrong updating mode')
        
    tmp = np.min(a)
    a = a - float(tmp < 0)*(tmp - lambda_min)
    a_1 = 1./a
    tmp = a_1 * G.conjugate()
    tmp1 = G - b*tmp
    tmp2 = mu*tmp
    tmp3 = a - b*b.conjugate()*a_1 - mu**2*a_1.T
    tmp4 = mu*a_1*b.conjugate()
    tmp4 = tmp4 + tmp4.T.conjugate()
    
    # Invert the transform
    out_part1 = (tmp1*tmp3.T + (tmp1*tmp4).T) / (tmp3*tmp3.T - tmp4*tmp4.T)
    out_part2 = ((tmp2*tmp3).T + tmp2*tmp4.T) / (tmp3*tmp3.T - tmp4*tmp4.T)
    return out_part1 - out_part2
    
    
    