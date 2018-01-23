import numpy as np
from theano.tests import unittest_tools as utt

from .tikhohov_solve import RegularizedSolve, LeastSquaresSolve

def test_regularized_solve():

    rng = np.random.RandomState(utt.fetch_seed())

    m = 10
    n = 3

    A_val = (rng.normal(size=(m, m)) * 0.5 + np.eye(m))
    b_val = rng.normal(size=(m, n))

    rsolve = RegularizedSolve(lambda_=1E-5)
    utt.verify_grad(rsolve, [A_val, b_val], 5, rng, eps=2e-8)


def test_leastsquares_solve():

    rng = np.random.RandomState(utt.fetch_seed())

    m = 10
    n = 3

    A_val = (rng.normal(size=(m, m)) * 0.5 + np.eye(m))
    b_val = rng.normal(size=(m, n))

    rsolve = LeastSquaresSolve()
    utt.verify_grad(rsolve, [A_val, b_val], 5, rng, eps=2e-8)
