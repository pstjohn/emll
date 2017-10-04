import numpy as np
import scipy.linalg

from theano import tensor
from theano.tensor.slinalg import Solve

class SymPosSolve(Solve):   
    """
    Class to allow `solve` to accept a symmetric matrix
    """

    def perform(self, node, inputs, output_storage):
        A, b = inputs
        if self.A_structure == 'lower_triangular':
            rval = scipy.linalg.solve_triangular(
                A, b, lower=True)
        elif self.A_structure == 'upper_triangular':
            rval = scipy.linalg.solve_triangular(
                A, b, lower=False)
        elif self.A_structure == 'symmetric':
            rval = scipy.linalg.solve(A, b, assume_a='pos')
        else:
            rval = scipy.linalg.solve(A, b)
        output_storage[0][0] = rval


sympos_solve = SymPosSolve(A_structure='symmetric')

class RegularizedSolve(Solve):
    """
    Solve a system of linear equations, Ax = b, while minimizing the norm of x.

    """

    __props__ = ('lambda_', *Solve.__props__)

    def __init__(self, lambda_=None):

        Solve.__init__(self)
        self.lambda_ = lambda_ if lambda_ is not None else 0.

    def __repr__(self):
        return 'RegularizedSolve{%s}' % str(self._props())

    def perform(self, node, inputs, output_storage):
        A, b = inputs

        A_hat = A.T @ A + self.lambda_ * np.eye(*A.shape)
        b_hat = A.T @ b

        cho = scipy.linalg.cho_factor(A_hat)
        rval = scipy.linalg.cho_solve(cho, b_hat)
        output_storage[0][0] = rval

    def grad(self, inputs, output_gradients):
        """
        Reverse-mode gradient updates for matrix solve operation.

        """

        A, b = inputs
        c = self(A, b)
        c_bar = output_gradients[0]

        A_hat = A.T.dot(A) + self.lambda_ * tensor.eye(*A.shape)
        x = sympos_solve(A_hat, c_bar)

        b_bar = A.dot(x)

        def force_outer(l, r):
            return tensor.outer(l, r) if r.ndim == 1 else l.dot(r.T)

        A_bar = force_outer(b - A.dot(c), x) - force_outer(b_bar, c)
        return [A_bar, b_bar]
