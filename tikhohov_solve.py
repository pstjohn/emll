import numpy as np
# import scipy.linalg

from scipy.linalg.misc import LinAlgError
from scipy.linalg.lapack import get_lapack_funcs

from theano import tensor
from theano.tensor.slinalg import Solve, solve_symmetric

class SymPosSolve(Solve):   
    """
    Class to allow `solve` to accept a symmetric matrix
    """

    def perform(self, node, inputs, output_storage):
        A, b = inputs
    
        posv, = get_lapack_funcs(('posv',), (A, b))
        c, rval, info = posv(A, b, lower=False,
                             overwrite_a=False,
                             overwrite_b=False)
    
        if info > 0:
            raise LinAlgError("singular matrix")

        output_storage[0][0] = rval


sympos_solve = SymPosSolve(A_structure='symmetric')

class RegularizedSolve(Solve):
    """
    Solve a system of linear equations, Ax = b, while minimizing the norm of x.
    Applies tikhovov regularization.

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

        posv, = get_lapack_funcs(('posv',), (A_hat, b_hat))
        c, rval, info = posv(A_hat, b_hat, lower=False,
                             overwrite_a=False,
                             overwrite_b=False)
    
        if info > 0:
            raise LinAlgError("singular matrix")

        output_storage[0][0] = rval

    def L_op(self, inputs, outputs, output_gradients):
        """
        Reverse-mode gradient updates for matrix solve operation.

        """

        A, b = inputs
        c = outputs[0]
        c_bar = output_gradients[0]

        A_hat = A.T.dot(A) + self.lambda_ * tensor.eye(*A.shape)
        x = sympos_solve(A_hat, c_bar)

        b_bar = A.dot(x)

        def force_outer(l, r):
            return tensor.outer(l, r) if r.ndim == 1 else l.dot(r.T)

        A_bar = force_outer(b - A.dot(c), x) - force_outer(b_bar, c)
        return [A_bar, b_bar]


class LeastSquaresSolve(Solve):
    """
    Solve a system of linear equations, Ax = b, while minimizing the norm of x.

    """

    def __init__(self):

        Solve.__init__(self)

    def __repr__(self):
        return 'LeastSquaresSolve{%s}' % str(self._props())

    def perform(self, node, inputs, output_storage):
        A, b = inputs

        gels, = get_lapack_funcs(('gels',), (A, b))
        c, rval, info = gels(A, b, overwrite_a=False, overwrite_b=False)
    
        if info > 0:
            raise LinAlgError("singular matrix")

        output_storage[0][0] = rval

    def L_op(self, inputs, outputs, output_gradients):
        """
        Reverse-mode gradient updates for matrix solve operation.

        """

        A, b = inputs
        c = outputs[0]
        c_bar = output_gradients[0]

        A_hat = A.T.dot(A)
        x = self(A_hat, c_bar)

        b_bar = A.dot(x)

        def force_outer(l, r):
            return tensor.outer(l, r) if r.ndim == 1 else l.dot(r.T)

        A_bar = force_outer(b - A.dot(c), x) - force_outer(b_bar, c)
        return [A_bar, b_bar]

