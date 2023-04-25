#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:56:40 2023

@author: jhonhu
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tools import whitening, laplace_rnd
from run_comPicard_solver import picard_complex

# # Data generating
M = 10
N = M
T = 20000
mu = 0.
sigma = 1.
S = laplace_rnd(M, T, mu, sigma) + 1j*laplace_rnd(M, T, mu, sigma)  # signal
H = np.random.normal(0, 1, (N,M)) + 1j*np.random.normal(0, 1, (N,M))  # channel
X = np.dot(H,S)  # observation

# Noise adding
snr = 20
pn = 10**(-snr/10)/float(N*T)*np.linalg.norm(X, ord='fro')**2
noise = (pn/2)**0.5*(np.random.normal(0, 1, (N,T)) + 1j*np.random.normal(0, 1, (N,T)))
X = X + noise

# Preprocessing
# 1) Centering
X_mean = np.mean(X, axis=1)
X = X - X_mean[:,None]
# 2） Whitening
whitening_mode = 'sph'
n_components = X.shape[0]
X_white, W_white = whitening(X, whitening_mode, n_components)

# Blind source separation
method = 'partNewton'  # (string) 'Gradient', 'partNewton', or 'fullNewton'
precon = 2  # (int) 1 or 2
Y, W, loss, normG = picard_complex(X_white, method, precon, maxiter = 150)

plt.title("Change of the l_inf norm of Gradient at each iteration")
plt.semilogy(normG, 'b-')
plt.show()

# Performance evaluation
plt.imshow(np.abs(W@W_white@H))
plt.colorbar(cax=None,ax=None,shrink=1)
plt.show()