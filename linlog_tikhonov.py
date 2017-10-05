from functools import partial

import numpy as np
import scipy as sp

import theano
import theano.tensor as T
import theano.tensor.slinalg
floatX = theano.config.floatX

from emll.tikhohov_solve import RegularizedSolve

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

    def steady_state_mat(self, en=None, yn=None, solver=None):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using a matrix solve method.

        en: np.ndarray
            a NR vector of perturbed normalized enzyme activities
        yn: np.ndarray
            a NY vector of normalized external metabolite concentrations
        solver: function
            A function to solve Ax = b for a (possibly) singular A.

        """
        en, yn = self._generate_default_inputs(en, yn)

        if solver is None:
            solver = partial(chol_solve_scipy, lambda_=self.lambda_)

        # Calculate steady-state concentrations using linear solve.
        N_hat = self.N @ np.diag(self.v_star * en)
        A = N_hat @ self.Ex
        b = -N_hat @ (np.ones(self.nr) + self.Ey @ yn)
        xn = solver(A, b)

        # Plug concentrations into the flux equation.
        vn = en * (np.ones(self.nr) + self.Ex @ xn + self.Ey @ yn)

        return xn, vn

    def steady_state_theano(self, Ex, Ey=None, en=None, yn=None,
                            solver=None):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using theano.

        Ex and Ey should be theano matrices, en and yn should be numpy arrays.

        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept theano matrices A and b, and return a symbolic x.
        """

        if solver is None:
            rsolve_op = RegularizedSolve(self.lambda_)
            solver = partial(chol_solve_theano, rsolve_op=rsolve_op)

        if Ey is None:
            Ey = T.as_tensor_variable(Ey)

        if isinstance(en, np.ndarray):
            en = np.atleast_2d(en)
            yn = np.atleast_2d(yn)

            n_exp = en.shape[0]

        else:
            n_exp = en.tag.test_value.shape[0]

        en = T.as_tensor_variable(en)
        yn = T.as_tensor_variable(yn)

        e_diag = en.dimshuffle(0, 1, 'x') * np.diag(self.v_star)
        N_rep = self.N.reshape((-1, *self.N.shape)).repeat(n_exp, axis=0)
        N_hat = T.batched_dot(N_rep, e_diag)

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

    def control_coef_fns(self, en=None, yn=None, solver=None):
        """Construct theano functions to evaluate dxn/den and dvn/den at the
        reference state as a function of the elasticity matrix.

        en: np.ndarray
            a NR vector of perturbed normalized enzyme activities
        yn: np.ndarray
            a NY vector of normalized external metabolite concentrations
        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept theano matrices A and b, and return a symbolic x.

        Returns:

        Cx, Cv: theano.functions
            Functions which operate on Ex matrices to return the flux control
            coefficients at the desired point.

        """
        en, yn = self._generate_default_inputs(en, yn)

        if solver is None:
            rsolve_op = RegularizedSolve(self.lambda_)
            solver = partial(chol_solve_theano, rsolve_op=rsolve_op)

        Ex = T.dmatrix('Ex')
        Ex.tag.test_value = self.Ex
        en_t = T.dvector('e')
        en_t.tag.test_value = en

        N_hat = T.dot(self.N, T.diag(self.v_star * en_t))
        A = N_hat.dot(Ex)
        b = -N_hat.dot(self.Ey.dot(yn.T).T + np.ones(self.nr))

        xn = solver(A, b)
        vn = en_t * (np.ones(self.nr) + T.dot(Ex, xn))

        x_jac = T.jacobian(xn, en_t)
        v_jac = T.jacobian(vn, en_t)

        Cx = theano.function([Ex, theano.In(en_t, 'en', en)], x_jac)
        Cv = theano.function([Ex, theano.In(en_t, 'en', en)], v_jac)

        return Cx, Cv




    # def calculate_jacobian_theano(self, Ex, x_star):
    #     """Return an expression for the jacobian matrix given a
    #     theano-expression for Ex and the reference metabolite concentration"""
    #
    #     return T.diag(1 / x_star)\
    #         .dot(self.N)\
    #         .dot(T.diag(self.v_star))\
    #         .dot(Ex)




def chol_solve_scipy(A, b, lambda_=None):
    A_hat = A.T @ A + lambda_ * np.eye(*A.shape)
    b_hat = A.T @ b

    cho = sp.linalg.cho_factor(A_hat)
    return sp.linalg.cho_solve(cho, b_hat)

def chol_solve_theano_old(A, b, lambda_=None):
    A_hat = T.dot(A.T, A) + lambda_ * T.eye(b.shape[0])
    b_hat = T.dot(A.T, b)
    
    L = T.slinalg.cholesky(A_hat)
    
    y = T.slinalg.solve_lower_triangular(L, b_hat)
    x = T.slinalg.solve_upper_triangular(L.T, y)

    return x.squeeze()

def chol_solve_theano(A, b, rsolve_op=None):
    return rsolve_op(A, b).squeeze()

def direct_solve_theano(A, b):
    return T.slinalg.solve(A, b).squeeze()

def symbolic_2x2(A, bi):
    a = A[0,0]
    b = A[0,1]
    c = A[1,0]
    d = A[1,1]

    A_inv = (T.stacklists([[d, -b], [-c, a]]) / (a * d - b * c))
    return T.dot(A_inv, bi).squeeze()

class NoOpMatrixInverse(T.nlinalg.MatrixInverse):
    pass


no_op_inverse = NoOpMatrixInverse()

def direct_inverse(A, b):
    return T.dot(no_op_inverse(A), b).squeeze()
