import numpy as np
import pytest

from .linlog_model import LinLogLeastNorm, LinLogLinkMatrix, LinLogTikhonov
from .util import create_elasticity_matrix, create_Ey_matrix

from test_models import models

import theano
import theano.tensor as tt
theano.config.compute_test_value = 'ignore' 
tt.config.optimizer = 'fast_compile'


@pytest.fixture(params=['teusink', 'mendes', 'textbook',
                        'greene_small', 'greene_large', 'contador'])
def cobra_model(request):
    model, N, v_star = models[request.param]()
    Ex = create_elasticity_matrix(model)
    Ey = create_Ey_matrix(model)
    return model, N, Ex, Ey, v_star

@pytest.fixture()
def linlog_least_norm(cobra_model):
     model, N, Ex, Ey, v_star = cobra_model
     return LinLogLeastNorm(N, Ex, Ey, v_star)


@pytest.fixture()
def linlog_tikhonov(cobra_model):
     model, N, Ex, Ey, v_star = cobra_model
     return LinLogTikhonov(N, Ex, Ey, v_star, lambda_=1E-6)


@pytest.fixture()
def linlog_link(cobra_model):
     model, N, Ex, Ey, v_star = cobra_model
     return LinLogLinkMatrix(N, Ex, Ey, v_star)

@pytest.fixture(params=[linlog_least_norm, linlog_tikhonov, linlog_link])
def linlog_model(request, cobra_model):
    return request.param(cobra_model)


def test_steady_state(linlog_model):
    ll = linlog_model

    # Fake up some experiments
    n_exp = 1

    e_hat_np = 2**(0.5*np.random.randn(n_exp, ll.nr))
    y_hat_np = 2**(0.5*np.random.randn(n_exp, ll.ny))

    e_hat_t = tt.dmatrix('en')
    e_hat_t.tag.test_value = e_hat_np

    y_hat_t = tt.dmatrix('yn')
    y_hat_t.tag.test_value = y_hat_np

    Ex_t = tt.dmatrix('Ex')
    Ex_t.tag.test_value = ll.Ex

    Ey_t = tt.dmatrix('Ey')
    Ey_t.tag.test_value = ll.Ey

    chi_ss, v_hat_ss = ll.steady_state_theano(Ex_t, Ey_t, e_hat_t, y_hat_t)

    io_fun = theano.function([Ex_t, Ey_t, e_hat_t, y_hat_t], 
                             [chi_ss, v_hat_ss])
    x_theano_test, v_theano_test = io_fun(ll.Ex, ll.Ey, e_hat_np, y_hat_np)

    x_np, v_np = ll.steady_state_mat(ll.Ex, ll.Ey, e_hat_np[0], y_hat_np[0])

    np.testing.assert_allclose(x_np, x_theano_test.flatten(), atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(v_np, v_theano_test.flatten(), atol=1e-5, rtol=1e-5)


def test_control_coeff(linlog_model):

    ll = linlog_model

    fd = 1E-5
    es = np.ones((ll.nr, ll.nr)) + fd * np.eye(ll.nr)

    cx_fd = np.zeros((ll.nm, ll.nr))
    cv_fd = np.zeros((ll.nr, ll.nr))
    for i, ei in enumerate(es):
        xni, vni = ll.steady_state_mat(en=ei)
        cx_fd[:, i] = xni / fd
        cv_fd[:, i] = (vni - 1) / fd

    np.testing.assert_allclose(
        cx_fd, ll.metabolite_control_coefficient(en=ei), atol=1E-5, rtol=1E-4)
    np.testing.assert_allclose(
        cv_fd, ll.flux_control_coefficient(en=ei), atol=1E-5, rtol=1E-4)


def test_reduction_methods(cobra_model):
    model, N, Ex, Ey, v_star = cobra_model
    ll1 = LinLogLeastNorm(N, Ex, Ey, v_star, reduction_method='smallbone')
    ll2 = LinLogLeastNorm(N, Ex, Ey, v_star, reduction_method='waldherr')
    ll3 = LinLogLeastNorm(N, Ex, Ey, v_star, reduction_method=None)

    n_exp = 1
    e_hat_np = 2**(0.5*np.random.randn(n_exp, ll1.nr))
    y_hat_np = 2**(0.5*np.random.randn(n_exp, ll1.ny))

    x1, v1 = ll1.steady_state_mat(en=e_hat_np[0], yn=y_hat_np[0])
    x2, v2 = ll2.steady_state_mat(en=e_hat_np[0], yn=y_hat_np[0])
    x3, v3 = ll3.steady_state_mat(en=e_hat_np[0], yn=y_hat_np[0])

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(x2, x3)
    np.testing.assert_allclose(v1, v2)
    np.testing.assert_allclose(v2, v3)

    np.testing.assert_allclose(ll1.Nr @ (v1 * v_star), 0., atol=1E-10)
    np.testing.assert_allclose(ll2.Nr @ (v2 * v_star), 0., atol=1E-10)
    np.testing.assert_allclose(ll3.Nr @ (v3 * v_star), 0., atol=1E-10)
    
