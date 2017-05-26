from collections import Counter

import numpy as np
import scipy
import pandas as pd
import casadi as cs
from tqdm import tqdm

import emll


class CasadiInference(object):
    def __init__(self, linlog_model, method='ex'):

        self.ll = ll = linlog_model

        if method == 'ex':
            self.ex_sparse = scipy.sparse.csc.csc_matrix(ll.Ex.round(2))
        elif method == 'ez':
            self.ex_sparse = scipy.sparse.csc.csc_matrix(ll.Ez.round(2))
            self.Linv = np.linalg.pinv(ll.L)

        self.ex_entries = cs.SX.sym('ex_i', len(self.ex_sparse.data))
        self.Ex_cs = cs.SX(cs.DM(self.ex_sparse).sparsity(), self.ex_entries)

        self.ey_sparse = scipy.sparse.csc.csc_matrix(ll.Ey.round(2))
        self.ey_entries = cs.SX.sym('ey_i', len(self.ey_sparse.data))
        self.Ey_cs = cs.SX(cs.DM(self.ey_sparse).sparsity(), self.ey_entries)

        self.method = method


    def set_data(self, e_hat, y_hat, chi, v_hat):

        assert len(e_hat) == len(y_hat) == len(chi) == len(v_hat)

        self.e_hat = e_hat
        self.y_hat = y_hat
        self.chi = chi
        self.v_hat = v_hat

        self.n_exp = len(e_hat)


    def solve_steady_state(self, Ex, Ey):
        """ Solve for chi, v_hat for the given elasticity matrices. Assumes Ex
        in the original metabolite ordering. """

        ll_new = emll.LinLogModel(
            self.ll.P.T @ self.ll.N, Ex, Ey, self.ll.v_star, 
            self.ll.P.T @ self.ll.x_star, smallbone=True)

        chi_new = np.vstack([ll_new.calc_chi_mat(self.e_hat[i], self.y_hat[i])
                             for i in range(self.n_exp)])

        v_new = np.vstack([ll_new.calc_steady_state_fluxes(
            self.e_hat[i], self.y_hat[i]) for i in range(self.n_exp)])

        v_hat_new = v_new / ll_new.v_star

        return chi_new, v_hat_new


    def create_nlp(self, print_time=True, print_level=0):

        chi_calc, v_hat_calc = zip(*(
            self.ll.calc_steady_state_casadi(
                self.Ex_cs, self.Ey_cs, e, y, method=self.method)
            for e, y in zip(self.e_hat, self.y_hat)))
            
        chi_calc = cs.horzcat(*chi_calc).T
        v_hat_calc = cs.horzcat(*v_hat_calc).T

        chi_diff = chi_calc - self.chi
        v_hat_diff = v_hat_calc - self.v_hat

        cost = (cs.sum_square(chi_diff) +
                cs.sum_square(v_hat_diff))

        # cost = (cs.sum_square(cs.diag(self.bootstrap_choice) @ chi_diff) +
        #         cs.sum_square(cs.diag(self.bootstrap_choice) @ v_hat_diff))

        nlp_x = cs.vertcat(self.ex_entries, self.ey_entries)
        self.n_var = nlp_x.shape[0]
        guess = np.hstack([self.ex_sparse.data, self.ey_sparse.data])
        self.sign = np.sign(guess)

        nlp = {
            'x': nlp_x, 
            'f': cost,
            'g': self.sign * nlp_x}
            # 'p': self.bootstrap_choice}

        self.solver = cs.nlpsol(
            'solver', 'ipopt', nlp, 
            {'print_time': print_time, 'ipopt.print_level': print_level})

        # self._cost_fn = cs.Function('cost', [nlp_x], [cost])


    def solve_nlp(self, p=None, raw=False):
        if p is None:
            p = np.ones(self.n_exp)

        sol = self.solver(x0=self.sign, lbx=-10, ubx=10,
                          lbg=np.zeros(self.n_var), ubg=10*np.ones(self.n_var),
                          p=p)

        if raw:
            return np.array(sol['x']).flatten()

        Ex_out = self.ex_sparse.copy()
        Ex_out.data = np.array(sol['x'][:len(self.ex_sparse.data)]).squeeze()
        Ex_out = np.asarray(Ex_out.todense())

        Ey_out = self.ey_sparse.copy()
        Ey_out.data = np.array(sol['x'][-len(self.ey_sparse.data):]).squeeze()
        Ey_out = np.asarray(Ey_out.todense())

        if self.method == 'ez':
            Ex_out = Ex_out @ self.Linv
        # Return the Ex matrices in the original ordering.
        return Ex_out @ self.ll.P, Ey_out


    def bootstrap(self, num, raw=False):

        if raw:
            out = np.zeros((num, self.n_var))

        else:
            ex_out = np.zeros((num, self.ll.Ex.shape[0], self.ll.Ex.shape[1]))
            ey_out = np.zeros((num, self.ll.Ey.shape[0], self.ll.Ey.shape[1]))
            chi_out = np.zeros((num, self.n_exp, self.ll.rank))
            v_hat_out = np.zeros((num, self.n_exp, self.ll.nr))

        for i in tqdm(range(num)):

            # Create a random vector to simulate bootstrap sampling
            c = Counter({i: 0 for i in range(self.n_exp)})
            c.update(np.random.randint(0, high=self.n_exp, size=self.n_exp))
            p = pd.Series(c).values
            
            if raw:
                out[i] = self.solve_nlp(p=p, raw=True)

            else:
                ex_out[i], ey_out[i] = self.solve_nlp(p=p)
                chi_out[i], v_hat_out[i] = self.solve_steady_state(ex_out[i], ey_out[i])

        if raw:
            return out
        else:
            return ex_out, ey_out, chi_out, v_hat_out

