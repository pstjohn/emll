from functools import partial

import numpy as np
import scipy as sp

import theano
import theano.tensor as T
floatX = theano.config.floatX

class LinLogTikhonov(object):

    def __init__(self, N, Ex, Ey, v_star, lambda_=None):
        """A class to perform the linear algebra underlying the 
        decomposition method.


        Parameters
        ----------
        N : np.array
            The full stoichiometric matrix of the considered model. Must be of
            dimensions MxN
        Ex : np.array
            An NxM array of the elasticity coefficients for the given linlog
            model.
        Ey : np.array
            An NxP array of the elasticity coefficients for the external
            species.
        v_star : np.array
            A length M vector specifying the original steady-state flux
            solution of the model.
        lam : float
            The $\lambda$ value to use for tikhonov regularization
        
        """
        self.nm, self.nr = N.shape
        self.ny = Ey.shape[1]

        self.N = N
        self.Ex = Ex
        self.Ey = Ey
        self.v_star = v_star
        self.lambda_ = lambda_ if lambda_ else 0

        assert Ex.shape == (self.nr, self.nm), "Ex is the wrong shape"
        assert Ey.shape == (self.nr, self.ny), "Ey is the wrong shape"
        assert len(v_star) == self.nr, "v_star is the wrong length"
        assert self.lambda_ >= 0, "lambda must be positive"
        assert np.allclose(self.N @ v_star, 0), "reference not steady state"

    def _generate_default_inputs(self, en=None, yn=None):
        """Create matricies representing no perturbation is input is None.
           
        """
        if en is None:
            en = np.ones(self.nr)

        if yn is None:
            yn = np.zeros(self.ny)

        return en, yn

    def steady_state_mat(self, en=None, yn=None):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using a matrix solve method.
        """
        en, yn = self._generate_default_inputs(en, yn)

        # Calculate steady-state concentrations using linear solve.
        N_hat = self.N @ np.diag(self.v_star * en)
        A = N_hat @ self.Ex
        b = -N_hat @ (np.ones(self.nr) + self.Ey @ yn)
        xn = chol_solve_scipy(A, b, self.lambda_)

        # Plug concentrations into the flux equation.
        vn = self.N @ (np.ones(self.nr) + self.Ex @ xn + self.Ey @ yn)

        return xn, vn

    def steady_state_theano(self, Ex, Ey, en=None, yn=None,
                            solve_method='tikhonov'):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using theano.

        Ex and Ey should be theano matrices, en and yn should be numpy arrays.
        """

        if solve_method == 'tikhonov':
            solver = partial(chol_solve_theano, lambda_=self.lambda_)
        elif solve_method == 'direct':
            solver = direct_solve_theano

        en = np.atleast_2d(en)
        yn = np.atleast_2d(yn)

        assert en.shape[0] == yn.shape[0], "input shape mismatch"

        e_diag = en[:, np.newaxis] * np.diag(self.v_star)
        N_hat = self.N @ e_diag
        inner_v = Ey.dot(yn.T).T + np.ones(self.nr, dtype=floatX)
        As = T.dot(N_hat, Ex)
        bs = T.batched_dot(-N_hat, inner_v.dimshuffle(0, 1, 'x'))
        xn, _ = theano.scan(
            lambda A, b: solver(A, b),
            sequences=[As, bs], strict=True)

        vn = en * (np.ones(self.nr) +
                   T.dot(Ex, xn.T).T +
                   T.dot(Ey, yn.T).T)

        return xn, vn

    def calculate_jacobian_theano(self, Ex, x_star):
        """Return an expression for the jacobian matrix given a
        theano-expression for Ex and the reference metabolite concentration"""

        return T.diag(1 / x_star)\
            .dot(self.N)\
            .dot(T.diag(self.v_star))\
            .dot(Ex)




def chol_solve_scipy(A, b, lambda_=None):
    A_hat = A.T @ A + lambda_ * np.eye(*A.shape)
    b_hat = A.T @ b

    cho = sp.linalg.cho_factor(A_hat)
    return np.linalg.cho_solve(cho, b_hat)

def chol_solve_theano(A, b, lambda_=None):
    A_hat = T.dot(A.T, A) + lambda_ * T.eye(b.shape[0])
    b_hat = T.dot(A.T, b)
    
    L = T.slinalg.cholesky(A_hat)
    
    y = T.slinalg.solve_lower_triangular(L, b_hat)
    x = T.slinalg.solve_upper_triangular(L.T, y)

    return x.squeeze()

def direct_solve_theano(A, b):
    return T.slinalg.solve(A, b).squeeze()
