import numpy as np
import scipy as sp

import theano
import theano.tensor as T
import theano.tensor.slinalg
floatX = theano.config.floatX

from emll.tikhohov_solve import RegularizedSolve, LeastSquaresSolve


class LinLogBase(object):

    def __init__(self, N, Ex, Ey, v_star):
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

        assert Ex.shape == (self.nr, self.nm), "Ex is the wrong shape"
        assert Ey.shape == (self.nr, self.ny), "Ey is the wrong shape"
        assert len(v_star) == self.nr, "v_star is the wrong length"
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

        en: np.ndarray
            a NR vector of perturbed normalized enzyme activities
        yn: np.ndarray
            a NY vector of normalized external metabolite concentrations
        solver: function
            A function to solve Ax = b for a (possibly) singular A.

        """
        en, yn = self._generate_default_inputs(en, yn)

        # Calculate steady-state concentrations using linear solve.
        N_hat = self.N @ np.diag(self.v_star * en)
        A = N_hat @ self.Ex
        b = -N_hat @ (np.ones(self.nr) + self.Ey @ yn)
        xn = self.solve(A, b)

        # Plug concentrations into the flux equation.
        vn = en * (np.ones(self.nr) + self.Ex @ xn + self.Ey @ yn)

        return xn, vn


    def steady_state_theano(self, Ex, Ey=None, en=None, yn=None):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using theano.

        Ex, Ey, en and yn should be theano matrices

        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept theano matrices A and b, and return a symbolic x.
        """

        if Ey is None:
            Ey = T.as_tensor_variable(Ey)

        if isinstance(en, np.ndarray):
            en = np.atleast_2d(en)
            n_exp = en.shape[0]
        else:
            n_exp = en.tag.test_value.shape[0]

        if isinstance(yn, np.ndarray):
            yn = np.atleast_2d(yn)

        en = T.as_tensor_variable(en)
        yn = T.as_tensor_variable(yn)

        e_diag = en.dimshuffle(0, 1, 'x') * np.diag(self.v_star)
        N_rep = self.N.reshape((-1, *self.N.shape)).repeat(n_exp, axis=0)
        N_hat = T.batched_dot(N_rep, e_diag)

        inner_v = Ey.dot(yn.T).T + np.ones(self.nr, dtype=floatX)
        As = T.dot(N_hat, Ex)
    
        bs = T.batched_dot(-N_hat, inner_v.dimshuffle(0, 1, 'x'))
        xn, _ = theano.scan(
            lambda A, b: self.solve_theano(A, b),
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

        Ex = T.dmatrix('Ex')
        Ex.tag.test_value = self.Ex
        en_t = T.dvector('e')
        en_t.tag.test_value = en

        N_hat = T.dot(self.N, T.diag(self.v_star * en_t))
        A = N_hat.dot(Ex)
        b = -N_hat.dot(self.Ey.dot(yn.T).T + np.ones(self.nr))

        xn = self.solve_theano(A, b)
        vn = en_t * (np.ones(self.nr) + T.dot(Ex, xn))

        x_jac = T.jacobian(xn, en_t)
        v_jac = T.jacobian(vn, en_t)

        Cx = theano.function([Ex, theano.In(en_t, 'en', en)], x_jac)
        Cv = theano.function([Ex, theano.In(en_t, 'en', en)], v_jac)

        return Cx, Cv


class LinLogSymbolic2x2(LinLogBase):
    """ Class for handling special case of a 2x2 full rank A matrix """

    def solve(self, A, bi):
        a = A[0,0]
        b = A[0,1]
        c = A[1,0]
        d = A[1,1]

        A_inv = np.array([[d, -b], [-c, a]]) / (a * d - b * c)
        return A_inv @ bi


    def solve_theano(self, A, bi):
        a = A[0,0]
        b = A[0,1]
        c = A[1,0]
        d = A[1,1]

        A_inv = (T.stacklists([[d, -b], [-c, a]]) / (a * d - b * c))
        return T.dot(A_inv, bi).squeeze()


class LinLogLeastNorm(LinLogBase):
    """ Uses dgels to solve for the least-norm solution to the linear equation """

    def solve(self, A, bi):
        x, residues, rank, s = sp.linalg.lstsq(A, bi)
        return x

    def solve_theano(self, A, bi):
        rsolve_op = LeastSquaresSolve()
        return rsolve_op(A, bi).squeeze()


class LinLogTikhonov(LinLogBase):
    """ Adds regularization to the linear solve, assumes A matrix is positive semi-definite """

    def __init__(self, N, Ex, Ey, v_star, lambda_=None):

        self.lambda_ = lambda_ if lambda_ else 0
        assert self.lambda_ >= 0, "lambda must be positive"

        LinLogBase.__init__(self, N, Ex, Ey, v_star)

    def solve(self, A, b):
        A_hat = A.T @ A + self.lambda_ * np.eye(*A.shape)
        b_hat = A.T @ b

        cho = sp.linalg.cho_factor(A_hat)
        return sp.linalg.cho_solve(cho, b_hat)

    def solve_theano(self, A, b):
        rsolve_op = RegularizedSolve(self.labmda_)
        return rsolve_op(A, b).squeeze()
