# A Complex Version of Preconditional ICA for Real Data (Picard) method

This GitHub repository hosts Python code of the complex extension version of Preconditional ICA for Real Data (Picard) method.


## Algorithm
Picard is an algorithm for maximum likelihood independent component analysis (ICA), proposed by Ablin P et. al. It shows state of the art speed of convergence, and solves the same problems as the widely used FastICA, Infomax and extended-Infomax, faster.  Its original version, however, only works for real variables. In this repository, Picard is extended to the complex domain.  

The original algorithm is detailed in:

`Ablin P, Cardoso J F, Gramfort A. Faster independent component analysis by preconditioning with Hessian approximations[J]. IEEE Transactions on Signal Processing, 2018, 66(15): 4040-4049.`

## Environment
In order to run the code in this repository the following software packages are needed:
* `Python 3` (for reference we use Python 3.8.8), with the following packages:`numpy`, `numexpr`, `matplotlib`, `copy`, `math`.
