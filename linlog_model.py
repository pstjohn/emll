import copy

import numpy as np
import scipy as sp

import casadi as cs

import theano
import theano.tensor as T
from theano.tensor.slinalg import solve
floatX = theano.config.floatX


class LinLogModel(object):

    def __init__(self, N, Ex, Ey, v_star, x_star, smallbone=True,
                 metabolite_labels=None):
        """A class to handle the stochiometric analysis and matrix reduction
        required to ensure the relevant matrices are invertable.

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
        x_star : np.array
            A length N vector specifying the original steady-state metabolite
            concentrations.
        smallbone : bool
            Whether or not to apply the smallbone rank correction
        metabolite_labels : list
            A list of metabolite labels in the original ordering
        """

        self.nm, self.nr = N.shape
        self.ny = Ey.shape[1]
        self._smallbone = smallbone

        assert Ex.shape == (self.nr, self.nm), "Ex is the wrong shape"
        assert Ey.shape == (self.nr, self.ny), "Ey is the wrong shape"
        assert len(v_star) == self.nr, "v_star is the wrong length"
        assert len(x_star) == self.nm, "x_star is the wrong length"

        if smallbone:
            # q, r, p = sp.linalg.qr((N @ np.diag(v_star) @ Ex).T, pivoting=True)
            q, r, p = sp.linalg.qr((N @ np.diag(v_star) @ Ex @ N).T, pivoting=True)
        else:
            q, r, p = sp.linalg.qr((N).T, pivoting=True)

        # Construct permutation matrix
        self._p = p
        self.P = np.zeros((len(p), len(p)), dtype=int)
        for i, pi in enumerate(p):
            self.P[i, pi] = 1

        # Get the matrix rank from the r matrix
        maxabs = np.max(np.abs(np.diag(r)))
        maxdim = max(N.shape)
        self._ranktol = tol = maxabs * maxdim * np.MachAr().eps
        # Find where the rows of r are all less than tol
        self.rank = (~(np.abs(r) < tol).all(1)).sum()

        # Reorder metabolite matrices
        self.v_star = v_star
        self.x_star = self.P @ x_star
        self.z_star = self.P[:self.rank] @ x_star
        self.Ex = Ex @ self.P.T
        self.Ey = Ey

        # Construct stoichiometric and link matrices
        self.Nr = Nr = self.P[:self.rank] @ N
        self.N = self.P @ N
        self.L = (np.diag(1/self.x_star) @ self.N @ np.linalg.pinv(Nr) @
                  np.diag(self.z_star))

        # Reduced elasticity matrix
        self.Ez = self.Ex @ self.L

        # Sanity checks
        assert np.linalg.matrix_rank(Nr) == self.rank
        assert np.allclose(self.Nr @ v_star, 0)
        assert np.allclose(self.N  @ v_star, 0)
        assert np.allclose(self.L[:self.rank], np.eye(self.rank),
                           atol=1E-7 * self.rank)

        # Labeling
        if metabolite_labels:
            self.x_labels = np.asarray(metabolite_labels)[p]
            self.z_labels = np.asarray(metabolite_labels)[p[:self.rank]]

    def reorder_x(self, x_old):
        """Reorder original metabolites to the permuted order"""
        return self.P @ x_old

    def x_to_z(self, x):
        """Select only the independent species from a permuted metabolite
        vector"""
        return x[:self.rank]

    def z_to_x(self, z):
        """Expand an independent list of metabolites to a full vector using the
        $log(x) \\approx x - 1$ assumption"""
        chi = self.L @ np.log(z/self.z_star).T
        return np.exp(chi) * self.x_star

    def z_to_x_alt(self, z):
        """Expand an independent list of metabolites to a full vector using
        $(x/x0 - 1) = L @ (z/z0 - 1)$"""
        chi = self.L @ (z/self.z_star - 1).T
        return (chi + 1) * self.x_star

    def set_Ex(self, Ex_new):
        """Set the state elasticity matrix with a new matrix of parameters.

        Assumes the incoming matrix is in the original state permutation order
        and re-orders as appropriate. Also asserts the rank of the new system
        is equal to the rank of the original system.

        Parameters
        ----------
        Ex_new : np.ndarray
            The new Ex matrix, in original state perturbation order

        """

        Ex_new_perm = Ex_new @ self.P.T
        assert self.Ex.shape == Ex_new.shape
        if self._smallbone:
            new_rank = np.linalg.matrix_rank(
                    self.N @ np.diag(self.v_star) @ Ex_new_perm, tol=self._ranktol)
            assert new_rank == self.rank, "Ex_new changes the model rank"

        self.Ex = Ex_new_perm

    def calc_chi_mat(self, e_hat=None, y_hat=None):
        """Calculate a the steady-state transformed independent metabolite
        concentrations using a matrix solve method.
        """

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        N_hat = self.Nr @ np.diag(self.v_star * e_hat)

        A = N_hat @ self.Ez
        b = -N_hat @ (np.ones(self.nr) + self.Ey @ np.log(y_hat))

        chi = np.linalg.solve(A, b)

        return chi

    def calc_zs_mat(self, e_hat=None, y_hat=None):
        """Calculate a the steady-state independent metabolite concentrations
        using a matrix solve method
        """

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        chi = self.calc_chi_mat(e_hat, y_hat)
        zs = self.z_star * np.exp(chi)
        return zs

    def calc_xs_mat(self, e_hat=None, y_hat=None):
        """Calculate the complete steady-state metabolite concentrations using
        a matrix solve
        """

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        zs = self.calc_zs_mat(e_hat, y_hat)
        return self.z_to_x(zs)

    def _construct_full_ode(self, e_hat=None, y_hat=None):

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        t = cs.SX.sym('t', 1)
        x_sym = cs.SX.sym('x', self.nm)

        v = e_hat * self.v_star * (1 + self.Ex @ cs.log(x_sym/self.x_star) +
                                   self.Ey @ np.log(y_hat))

        v_fun = cs.Function('v', [x_sym], [v])
        ode = cs.DM(self.N) @ v_fun(x_sym)

        return t, x_sym, ode

    def calc_xs_full_ode(self, e_hat=None, y_hat=None):
        """Calculate the complete steady-state metabolite concentrations by
        integrating the complete ODE matrix (without a link matrix)
        """

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        t, x_sym, ode = self._construct_full_ode(e_hat, y_hat)

        integrator = cs.integrator(
            'full', 'cvodes',
            {
                't': t,
                'x': x_sym,
                'ode': ode,
            },
            {
                'abstol': 1E-9,
                'reltol': 1E-9,
                'tf': 2000,
                'regularity_check': True,
            })

        xs = np.array(integrator(x0=self.x_star)['xf']).flatten()

        return xs

    def _construct_reduced_ode(self, e_hat=None, y_hat=None):

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        t = cs.SX.sym('t', 1)
        z_sym = cs.SX.sym('z', self.rank)

        v = e_hat * self.v_star * (
            1 + self.Ez @ cs.log(z_sym/self.z_star) +
            self.Ey @ np.log(y_hat))

        v_fun = cs.Function('v', [z_sym], [v])

        ode = cs.DM(self.Nr) @ v_fun(z_sym)

        return t, z_sym, ode

    def calc_xs_reduced_ode(self, e_hat=None, y_hat=None):
        """Calculate the complete steady-state metabolite concentrations by
        integrating the stoichiometrically reduced ODE system
        """

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        t, z_sym, ode = self._construct_reduced_ode(e_hat, y_hat)

        integrator = cs.integrator(
            'full', 'cvodes',
            {
                't': t,
                'x': z_sym,
                'ode': ode,
            },
            {
                'abstol': 1E-9,
                'reltol': 1E-9,
                'tf': 10000,
                'regularity_check': True,
            })

        zs = np.array(integrator(x0=self.z_star)['xf']).flatten()

        return self.z_to_x(zs)

    def calc_xs_transformed_ode(self, e_hat=None, y_hat=None):
        """Calculate the complete steady-state metabolite concentrations by
        integrating the exp-transformed ODE system
        """

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        t = cs.SX.sym('t', 1)
        chi_sym = cs.SX.sym('z', self.rank)

        n_hat = np.diag(1/self.z_star) @ self.Nr @ np.diag(self.v_star)
        ode = cs.diag(cs.exp(-chi_sym)) @ n_hat @ np.diag(e_hat) @ (
            np.ones(self.nr) + self.Ez @ chi_sym +
            self.Ey @ np.log(y_hat))

        integrator = cs.integrator(
            'full', 'cvodes',
            {
                't': t,
                'x': chi_sym,
                'ode': ode,
            },
            {
                'abstol': 1E-9,
                'reltol': 1E-9,
                'tf': 10000,
                'regularity_check': True,
            })

        chi = np.array(integrator(x0=np.zeros(self.rank))['xf']).flatten()

        zs = np.exp(chi) * self.z_star
        xs = self.z_to_x(zs)

        return xs

    def calc_jacobian_reduced_ode(self, z, e_hat=None, y_hat=None):
        """Calculate the jacobian matrix of the reduced system using casadi at
        the given perturbated point"""

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        t, z_sym, ode = self._construct_reduced_ode(e_hat, y_hat)
        ode_f = cs.Function('v', [z_sym], [ode])
        return np.array(ode_f.jacobian(0)(z)[0])

    def calc_jacobian_full_ode(self, x, e_hat=None, y_hat=None):
        """Calculate the jacobian matrix of the full system using casadi at
        the given perturbated point"""

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        t, x_sym, ode = self._construct_full_ode(e_hat, y_hat)
        ode_f = cs.Function('v', [x_sym], [ode])
        return np.array(ode_f.jacobian(0)(x)[0])

    @staticmethod
    def is_stable(jac, tol=1E-10):
        """Check a jacobian matrix to ensure stability"""

        eigs = np.linalg.eigvals(jac)
        return np.all(np.real(eigs) <= tol)

    def calc_fluxes_from_z(self, z, e_hat=None, y_hat=None):
        """Calculate fluxes for a given steady-state and parameters"""

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        v = e_hat * self.v_star * (1 + self.Ez @
                                   np.log(z / self.z_star) +
                                   self.Ey @ np.log(y_hat))

        return v

    def calc_fluxes_from_x(self, x, e_hat=None, y_hat=None):
        """Calculate fluxes for a given steady-state and parameters"""

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        v = e_hat * self.v_star * (1 + self.Ex @ np.log(x/self.x_star) +
                                   self.Ey @ np.log(y_hat))

        return v

    def calc_steady_state_fluxes(self, e_hat=None, y_hat=None):
        """Calculate the steady-state fluxes directly"""

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        chi = self.calc_chi_mat(e_hat, y_hat)
        v = e_hat * self.v_star * (1 + self.Ez @ chi +
                                   self.Ey @ np.log(y_hat))

        return v

    def calc_jacobian_mat(self, z=None, e_hat=None, y_hat=None):
        """Calculate the jacobian matrix of the reduced system via matrix
        algebra at the given perturbated point"""

        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        if z is None:
            z = self.calc_zs_mat(e_hat, y_hat)

        n_hat = np.diag(1/self.z_star) @ self.Nr @ np.diag(self.v_star * e_hat)
        n_hat_ex_L = n_hat @ self.Ez
        chi = np.log(z/self.z_star)

        return (np.diag(np.exp(-chi)) @ n_hat_ex_L @ chi +
                n_hat_ex_L @ np.diag(np.exp(-chi)) +
                n_hat @ (np.ones(self.nr) + self.Ey @ np.log(y_hat)) @
                np.diag(np.exp(-chi)))

    def construct_control_coeff_fn(self, y_hat=None, debug=False):
        """Construct a theano function to compute the control coefficients
        ($\\frac{d \\ln v}{d \\ln e}$) at a given e_hat.

        Returns a theano.function which takes two arguments, Ex and e_hat, and
        returns the (nr)x(nr) flux control matrix.

        Parameters
        ----------
        y_hat : np.ndarray
            a perturbed external concentration vector to compile into the
            theano function. Should probably make this an explicit parameter.

        """

        if y_hat is None:
            y_hat = np.ones(self.ny)
        else:
            assert y_hat.shape == (self.ny,)


        Ex_theano = T.dmatrix('Ex')
        Ex_theano.tag.test_value = self.Ex

        e_hat = T.dvector('e')
        e_hat.tag.test_value = np.ones(self.nr)

        # Calculate chi_ss, steady state transformed metabolite concentration
        N_hat = T.dot(np.diag(1 / self.z_star) @ self.Nr,
                      T.diag(self.v_star * e_hat))
        Ex_L = T.dot(Ex_theano, self.L)

        chi_ss_left = T.dot(N_hat, Ex_L)
        chi_ss_right = T.dot(
            -N_hat, (np.ones(self.nr) + self.Ey @ np.log(y_hat)))
        chi_ss = T.dot(T.nlinalg.matrix_inverse(chi_ss_left), chi_ss_right)

        if debug:
            e_test = np.ones(self.nr)
            e_test[1] = 2.
            chi_test = theano.function([Ex_theano, e_hat], [chi_ss])
            assert np.allclose(chi_test(self.Ex, e_test),
                               self.calc_chi_mat(e_test, y_hat))

        v_ss = self.v_star * e_hat * (np.ones(self.nr) + T.dot(Ex_L, chi_ss) +
                                      self.Ey @ np.log(y_hat))

        if debug:
            v_test = theano.function([Ex_theano, e_hat], [v_ss])
            assert np.allclose(v_test(self.Ex, e_test),
                               self.calc_steady_state_fluxes(e_test, y_hat))

        v_grad = T.jacobian(v_ss, e_hat)
        rel_v_grad = v_grad * (e_hat.reshape((1, self.nr)) /
                               v_ss.reshape((self.nr, 1)))

        return theano.function([Ex_theano, e_hat], rel_v_grad)

    def calc_metabolite_control_coeff(self, e_hat=None, y_hat=None):
        """Calculate the metabolite control coefficient matrix, 
        Cx = d(ln x)/d(ln e).

        full_return: bool
            Whether or not to return v_ss, Ex_ss as well as Cx. Used for
            flux_control_coeff calculation.

        """
        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        # Calculate the steady-state flux
        v_ss = self.calc_steady_state_fluxes(e_hat, y_hat)

        # Calculate the elasticity matrix at the new steady-state
        Ez_ss = np.diag(self.v_star * e_hat / v_ss) @ self.Ez

        Cx = (-self.L @ 
              np.linalg.inv(self.Nr @ np.diag(v_ss) @ Ez_ss) @
              self.Nr @ np.diag(v_ss))
        
        return Cx

    def calc_flux_control_coeff(self, e_hat=None, y_hat=None):
        """Calculate the flux control coefficient matrix, 
        Cv = d(ln v)/d(ln e).

        """
        e_hat, y_hat = self._generate_default_inputs(e_hat, y_hat)

        # Calculate the steady-state flux
        v_ss = self.calc_steady_state_fluxes(e_hat, y_hat)

        # Calculate the elasticity matrix at the new steady-state
        Ez_ss = np.diag(self.v_star * e_hat / v_ss) @ self.Ez

        Cv = (-Ez_ss @ 
              np.linalg.inv(self.Nr @ np.diag(v_ss) @ Ez_ss) @
              self.Nr @ np.diag(v_ss)) + np.eye(self.nr)

        return Cv


    def _generate_default_inputs(self, e_hat=None, y_hat=None):
        """Create matricies of ones if input arguments are None"""

        if e_hat is None:
            e_hat = np.ones(self.nr)

        if y_hat is None:
            y_hat = np.ones(self.ny)

        return e_hat, y_hat


    def calculate_jacobian_theano(self, Ez):
        """Return an expression for the jacobian matrix given a
        theano-expression for Ex @ L"""

        N_hat_jac = T.diag(1 / self.z_star).dot(self.Nr).dot(T.diag(self.v_star))
        return T.dot(N_hat_jac, Ez)


    def calculate_steady_state_theano(self, Ez, Ey, e_hat, y_hat):
        """For a matrix of e_hat, y_hat values, calculate the chi_ss and v_hat_ss
        resulting from the relevant perturbations (using theano)"""

        e_diag = e_hat[:, np.newaxis] * np.diag(self.v_star)
    
        # Calculate the steady-state log(x/x_star) with some complicated matrix algebra...
        N_hat = (self.Nr @ e_diag).astype(floatX)
        chi_ss_left = T.dot(N_hat, Ez)
        inner_v = Ey.dot(T.log(y_hat.T)).T + np.ones(self.nr, dtype=floatX)
        chi_ss_right = T.batched_dot(-N_hat, inner_v.dimshuffle(0, 1, 'x'))
        chi_ss, _ = theano.scan(
            lambda n_left, n_right: solve(n_left, n_right),
            sequences=[chi_ss_left, chi_ss_right], strict=True)

        v_hat_ss = (e_hat) * (
            np.ones(self.nr) +
            T.dot(Ez, chi_ss).squeeze().T + 
            T.dot(Ey, np.log(y_hat)[:, :, np.newaxis]).squeeze().T)


        return chi_ss.squeeze(), v_hat_ss


    def calculate_steady_state_batch_theano(self, Ez, Ey, e_hat, y_hat):
        """For a single e_hat, y_hat perturbation (as theano variables),
        calculate the steady state"""

        e_diag = T.diag(e_hat * self.v_star.astype(floatX))   
        N_hat = T.dot(self.Nr, e_diag)
        b = -N_hat.dot(Ey.dot(T.log(y_hat.T)).T + 
                       np.ones(self.nr, dtype=floatX))
        A = N_hat.dot(Ez)
        chi_ss = solve(A, b)
        v_hat_ss = e_hat * (np.ones(self.nr) + T.dot(Ez, chi_ss) + 
                            T.dot(Ey, np.log(y_hat)))

        return chi_ss, v_hat_ss

    def calc_steady_state_casadi(self, Exz, Ey, e_hat, y_hat, method='ex'):
        """ For a single e_hat, y_hat (can be either casadi or numpy vectors)
        and a given Ex, Ey (as casadi matrices), calculate the steady-state
        concentrations and fluxes.

        method: 'ex' or 'ez'
            whether an Ex or Ez matrix was provided

        """

        if method is 'ex':
            Ez = Exz @ self.L
        else:
            Ez = Exz

        N_hat = self.Nr @ cs.diag(e_hat * self.v_star)
        
        A = N_hat @ Ez
        b = -N_hat @ (np.ones(self.nr) + Ey @ cs.log(y_hat))
        chi = cs.solve(A, b)
        
        v_hat = e_hat * (np.ones(self.nr) + Ez @ chi + 
                         Ey @ cs.log(y_hat))
        
        return chi, v_hat

    def copy(self):
        """ Return a deepcopy of the class """
        return copy.deepcopy(self)
