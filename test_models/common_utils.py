import numpy as np
import scipy as sp

import casadi as cs

class StateCompressor(object):

    def __init__(self, N, Ex, v_star, x_star):
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
        v_star : np.array
            A length M vector specifying the original steady-state flux
            solution of the model.
        x_star : np.array
            A length N vector specifying the original steady-state metabolite
            concentrations.

        """

        self.N = N
        self.x_star = x_star
        self.v_star = v_star

        self.nm, self.nr = N.shape

        assert Ex.shape == (self.nr, self.nm), "Ex is the wrong shape"
        assert len(v_star) == self.nr, "v_star is the wrong length"
        assert len(x_star) == self.nm, "x_star is the wrong length"

        q, r, p = sp.linalg.qr((N @ np.diag(v_star) @ Ex).T, pivoting=True)
        
        # # Construct permutation matrix
        # self.P = np.zeros((len(p), len(p)), dtype=int)
        # for i, pi in enumerate(p):
        #     self.P[i, pi] = 1

        # Get the matrix rank from the r matrix
        maxabs = np.max(np.abs(np.diag(r)))
        maxdim = max(N.shape)
        tol = maxabs * maxdim * np.MachAr().eps
        # Find where the rows of r are all less than tol
        self.rank = (~(np.abs(r) < tol).all(1)).sum()

        # Permutation vector
        self.p = np.sort(p[:self.rank])

        self.z_star = self.x_to_z(self.x_star)
        self.Nr = Nr = N[self.p]
        self.L = np.diag(1/self.x_star) @ N @ np.linalg.pinv(Nr) @ np.diag(self.z_star)

        # Sanity checks
        assert np.linalg.matrix_rank(Nr) == self.rank
        assert np.allclose(Nr @ v_star, 0)

    def x_to_z(self, x):
        return x[self.p]
        
    def z_to_x(self, z):
        chi = self.L @ np.log(z/self.z_star).T
        return np.exp(chi) * self.x_star

    def z_to_x_alt(self, z):
        chi = self.L @ (z/self.z_star - 1).T
        return (chi + 1) * self.x_star


def calc_xs_full_ode(Ex, Ey, e_hat, y_hat, state_compressor):

    t = cs.SX.sym('t', 1)
    x_sym = cs.SX.sym('x', sc.nm)
    y_sym = cs.SX.sym('y', ny)
    e_sym = cs.SX.sym('e', sc.nr)
    Ex_sym = cs.SX.sym('Ex', sc.nr, sc.nm)
    Ey_sym = cs.SX.sym('Ex', sc.nr, Ey.shape[1])


    v = e_sym * v_star * (1 + Ex_sym @ cs.log(x_sym/sc.x_star) + Ey @ np.log(y_hat))
    v_fun = cs.Function('v', [x_sym, e_sym, Ex_sym], [v])

    ode = cs.mtimes(cs.DM(N), v_fun(x_sym, e_sym, Ex_sym))

    integrator = cs.integrator(
        'full', 'cvodes',
        {
            't': t,
            'x': x_sym,
            'p': cs.vertcat(e_sym, Ex_sym.reshape((-1, 1))),
            'ode': ode,
        },
        {
            'tf': 2000,
            'regularity_check': True,
        })

    p = np.hstack([e_hat, Ex.reshape((-1, 1), order='F').squeeze()])
    xs = np.array(integrator(x0=sc.x_star, p=p)['xf']).flatten()

    return xs
