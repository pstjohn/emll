import numpy as np
import scipy as sp
from theano.tests import unittest_tools as utt

from .tikhohov_solve import RegularizedSolve, LeastSquaresSolve

def test_regularized_solve():

    rng = np.random.RandomState(utt.fetch_seed())

    m = 10
    n = 3

    A_val = (rng.normal(size=(m, m)) * 0.5 + np.eye(m))
    b_val = rng.normal(size=(m, n))

    rsolve = RegularizedSolve(lambda_=1E-5)
    utt.verify_grad(rsolve, [A_val, b_val], 5, rng, eps=1.0e-7)


def test_leastsquares_solve():

    rng = np.random.RandomState(utt.fetch_seed())

    m = 10
    n = 3

    A_val = (rng.normal(size=(m, m)) * 0.5 + np.eye(m))
    b_val = rng.normal(size=(m, n))

    rsolve = LeastSquaresSolve()
    utt.verify_grad(rsolve, [A_val, b_val], 5, rng, eps=1.0e-7)


def test_leastsquares_solve_illconditioned():

    rng = np.random.RandomState(utt.fetch_seed())

    A_bad = sp.sparse.rand(50, 50, density=0.01).todense()
    A_bad = A_bad.T @ A_bad
    for i in range(40):
        A_bad[i, i] += np.random.rand()

    rsolve = LeastSquaresSolve(driver='gelsd')
    utt.verify_grad(rsolve, [A_bad, np.zeros(50)], 5, rng, eps=1.0e-7)
