import numpy as np
import pytest

from .linlog_model import LinLogModel
from .util import create_elasticity_matrix

from test_models import models

import theano
import theano.tensor as tt
theano.config.compute_test_value = 'ignore' 

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE"

try:
    import casadi
except ImportError:
    casadi = None

@pytest.fixture(params=['teusink', 'mendes', 'textbook',
                        'greene_small', 'greene_large', 'contador'])
def cobra_model(request):
    model, N, v_star = models[request.param]()
    return model, N, v_star


@pytest.fixture(params=['reversible', 'irreversible'])
def ex_method(request):
    return request.param 

@pytest.fixture()
def linlog_model(cobra_model, ex_method):

    model, N, v_star = cobra_model

    # Create perturbation matricies
    if ex_method is 'reversible':
        Ex = -N.T
    else:
        Ex = create_elasticity_matrix(model)

    boundary_indexes = [model.reactions.index(r) for r in
                        model.reactions.query(lambda x: x.boundary, None)]
    boundary_directions = [1 if r.products else -1 for r in
                           model.reactions.query(
                               lambda x: x.boundary, None)]
    ny = len(boundary_indexes)
    Ey = np.zeros((N.shape[1], ny))

    for i, (rid, direction) in enumerate(zip(boundary_indexes,
                                             boundary_directions)):
        Ey[rid, i] = direction

    y_hat = np.ones(ny)
    y_hat[1] -= 0.2

    e_hat = np.ones(N.shape[1])
    e_hat[v_star.argmax()] *= 1.2

    x_star = 2 * np.ones(N.shape[0])

    # Set up class
    ll = LinLogModel(N, Ex, Ey, v_star, x_star, smallbone=True,
                     metabolite_labels=[m.id for m in model.metabolites])

    return ll, e_hat, y_hat
    


def test_rank_decomposition_no_smallbone(cobra_model):

    model, N, v_star = cobra_model

    # Create perturbation matricies
    Ex = -N.T

    boundary_indexes = [model.reactions.index(r) for r in
                        model.reactions.query(lambda x: x.boundary, None)]
    boundary_directions = [1 if r.products else -1 for r in
                           model.reactions.query(
                               lambda x: x.boundary, None)]
    ny = len(boundary_indexes)
    Ey = np.zeros((N.shape[1], ny))

    for i, (rid, direction) in enumerate(zip(boundary_indexes,
                                             boundary_directions)):
        Ey[rid, i] = direction

    y_hat = np.ones(ny)
    y_hat[1] -= 0.2

    e_hat = np.ones(N.shape[1])
    e_hat[v_star.argmax()] *= 1.2

    x_star = 2 * np.ones(N.shape[0])

    # Set up class
    sc = LinLogModel(N, Ex, Ey, v_star, x_star, smallbone=False)

    # Test rank decomposition
    assert np.linalg.matrix_rank(sc.N, tol=sc._ranktol) == sc.rank

    # Test conversion methods
    v_test = 1E-1 * np.random.randn(sc.nr)
    delta_z = sc.z_star + sc.Nr @ v_test
    delta_x = sc.x_star + sc.N @ v_test
    assert np.allclose(sc.z_to_x_alt(delta_z), delta_x)


def test_rank_decomposition(linlog_model):

    ll, e_hat, y_hat = linlog_model

    # Test rank decomposition (smallbone)
    # assert np.linalg.matrix_rank(
    #         (ll.N @ np.diag(ll.v_star) @ ll.Ex), 
    #         tol=ll._ranktol) == ll.rank

    assert np.linalg.matrix_rank(ll.Ez) == ll.rank

    assert np.allclose(ll.calc_chi_mat(), 0.)


def test_steady_state_methods(linlog_model):

    ll, e_hat, y_hat = linlog_model

    # Test matrix steady-state
    x_ss_mat = ll.calc_xs_mat(e_hat, y_hat)
    z_ss_mat = ll.calc_zs_mat(e_hat, y_hat)

    assert np.all(np.isfinite(x_ss_mat))
    assert np.all(np.isfinite(z_ss_mat))

    # Test jacovian method equi valence
    ll.calc_jacobian_full_ode(x_ss_mat, e_hat, y_hat)
    pjac_red = ll.calc_jacobian_reduced_ode(z_ss_mat, e_hat, y_hat)
    pjac_mat = ll.calc_jacobian_mat(z_ss_mat, e_hat, y_hat)

    assert np.allclose(pjac_red, pjac_mat)

    ojac_red = ll.calc_jacobian_reduced_ode(ll.z_star)
    ojac_mat = ll.calc_jacobian_mat(ll.z_star)

    assert np.allclose(ojac_red, ojac_mat)

    # Test flux method equivalence
    assert np.allclose(
        ll.calc_fluxes_from_x(x_ss_mat, e_hat, y_hat),
        ll.calc_fluxes_from_z(z_ss_mat, e_hat, y_hat))

    assert np.allclose(
        ll.calc_fluxes_from_z(z_ss_mat, e_hat, y_hat),
        ll.calc_steady_state_fluxes(e_hat, y_hat))

def test_metabolite_control_coeff(linlog_model):

    ll, e_hat, y_hat = linlog_model

    if not np.any(np.isclose(ll.v_star, 0.)):
        assert np.allclose(
            ll.calc_metabolite_control_coeff(e_hat, y_hat).sum(1), 0.)


def test_flux_control_coeff(linlog_model):

    ll, e_hat, y_hat = linlog_model

    # Test flux control coefficients
    v_grad_test = ll.construct_control_coeff_fn(debug=True)
    theano_grad = v_grad_test(ll.Ex, np.ones(ll.nr))

    # Make sure the summation theorem is obeyed
    grad_sum = (theano_grad).sum(1)
    assert np.allclose(grad_sum[np.where(np.isfinite(grad_sum))], 1.)

    # Calc gradient via finite difference
    fd_grad = np.zeros((ll.nr, ll.nr))
    fd_size = 1E-5

    for i in range(ll.nr):
        e_test = np.ones(ll.nr)
        e_test[i] += fd_size
        fd_grad[:, i] = (ll.calc_steady_state_fluxes(e_test, np.ones(ll.ny)) -
                         ll.v_star) / (fd_size * ll.v_star)
        
    # Make sure fd results are close
    elems = np.where(np.isfinite(fd_grad))
    assert np.allclose(fd_grad[elems], theano_grad[elems], rtol=1e-2, atol=1E-4)

    if not np.any(np.isclose(ll.v_star, 0.)):
        mat_grad = ll.calc_flux_control_coeff(np.ones(ll.nr), np.ones(ll.ny))
        assert np.allclose(theano_grad[elems], mat_grad[elems])

    # Test the flux control coefficients at a perturbed steady state
    theano_grad = v_grad_test(ll.Ex, e_hat)

    # Make sure the summation theorem is obeyed
    grad_sum = (theano_grad).sum(1)
    assert np.allclose(grad_sum[np.where(np.isfinite(grad_sum))], 1.)

    v_ss = ll.calc_steady_state_fluxes(e_hat, np.ones(ll.ny))
    for i in range(ll.nr): 
        e_test_i = np.array(e_hat)
        e_test_i[i] += fd_size
        fd_grad[:, i] = (ll.calc_steady_state_fluxes(
            e_test_i, np.ones(ll.ny)) - v_ss) / (fd_size)

    # Scale by e_hat and v_ss
    fd_grad = (fd_grad * e_hat) / (np.atleast_2d(v_ss).T)

    # assert np.isfinite(fd_grad).all()

    # Make sure fd results are close
    elems = np.where(np.isfinite(fd_grad))
    assert np.allclose(fd_grad[elems], theano_grad[elems], rtol=1e-2, atol=1E-4)

    if not np.any(np.isclose(ll.v_star, 0.)):
        mat_grad = ll.calc_flux_control_coeff(e_hat, np.ones(ll.ny))
        assert np.allclose(theano_grad[elems], mat_grad[elems])

        # Test methods with perturned y
        control_coeffs = ll.construct_control_coeff_fn(y_hat=y_hat)
        theano_mat = control_coeffs(ll.Ex, e_hat)
        assert np.allclose(
            theano_mat, ll.calc_flux_control_coeff(e_hat, y_hat))

@pytest.mark.skipif(casadi is None, reason="casadi not found")
def test_casadi_methods(linlog_model):

    ll, e_hat, y_hat = linlog_model

    # ODE dynamic system tests
    if ll.is_stable(ll.calc_jacobian_mat(z=None, e_hat=e_hat, y_hat=y_hat)):

        # check steady-state method equivalence
        x_ss_mat = ll.calc_xs_mat(e_hat, y_hat)
        x_ss_full = ll.calc_xs_full_ode(e_hat, y_hat)
        x_ss_red = ll.calc_xs_reduced_ode(e_hat, y_hat)
        x_ss_xform = ll.calc_xs_transformed_ode(e_hat, y_hat)

        # assert np.allclose(x_ss_mat, x_ss_full)
        assert np.allclose(x_ss_mat, x_ss_red)
        assert np.allclose(x_ss_mat, x_ss_xform)

        # Make sure fluxes are equal

        # This one needs a lower tolerance since x_ss_full != x_ss_mat from
        # the linlog assumption
        from scipy.stats import pearsonr

        assert pearsonr(
            ll.calc_fluxes_from_x(x_ss_full, e_hat, y_hat),
            ll.calc_fluxes_from_x(x_ss_mat, e_hat, y_hat))[0] > .99

        assert np.allclose(
            ll.calc_fluxes_from_x(x_ss_red, e_hat, y_hat),
            ll.calc_fluxes_from_x(x_ss_mat, e_hat, y_hat))

        assert np.allclose(
            ll.calc_fluxes_from_x(x_ss_xform, e_hat, y_hat),
            ll.calc_fluxes_from_x(x_ss_mat, e_hat, y_hat))


def test_theano_jac(linlog_model):

    ll, e_hat, y_hat = linlog_model
    
    Ez_t = tt.dmatrix('Ez')
    Ez_t.tag.test_value = ll.Ez
    
    jac_theano_sym = ll.calculate_jacobian_theano(Ez_t)
    jac_theano_fun = theano.function([Ez_t], [jac_theano_sym])

    jac_mat = ll.calc_jacobian_mat(ll.z_star)
    jac_theano = jac_theano_fun(ll.Ez)

    assert np.allclose(jac_mat, jac_theano)


def test_theano_steady_state(linlog_model):

    ll, e_hat, y_hat = linlog_model

    # Fake up some experiments
    n_exp = 10
    e_hat = 2**(0.5*np.random.randn(n_exp, ll.nr))
    y_hat = 2**(0.5*np.random.randn(n_exp, ll.ny))

    # Calculate the numpy-expected results
    chi_test = np.array([ll.calc_chi_mat(e_hat[i], y_hat[i]) for i in range(n_exp)])
    v_test = np.array([ll.calc_steady_state_fluxes(ei, yi)
                       for chi_i, ei, yi in zip(chi_test, e_hat, y_hat)])

    Ez_t = tt.dmatrix('Ez')
    Ez_t.tag.test_value = ll.Ez

    Ey_t = tt.dmatrix('Ey')
    Ey_t.tag.test_value = ll.Ey

    chi_ss, v_hat_ss = ll.calculate_steady_state_theano(Ez_t, Ey_t, e_hat, y_hat)

    io_fun = theano.function([Ez_t, Ey_t], [chi_ss, ll.v_star * v_hat_ss])
    x_theano_test, v_theano_test = io_fun(ll.Ez, ll.Ey)

    assert np.allclose(chi_test, x_theano_test)
    assert np.allclose(v_test, v_theano_test)


def test_calculate_steady_state_batch_theano(linlog_model):

    ll, e_hat, y_hat = linlog_model

    # Fake up some experiments
    e_hat_np = 2**(0.5*np.random.randn(ll.nr))
    y_hat_np = 2**(0.5*np.random.randn(ll.ny))

    e_hat_t = tt.vector()
    e_hat_t.tag.test_value = e_hat_np

    y_hat_t = tt.vector()
    y_hat_t.tag.test_value = y_hat_np

    Ez_t = tt.dmatrix('Ez')
    Ez_t.tag.test_value = ll.Ez

    Ey_t = tt.dmatrix('Ey')
    Ey_t.tag.test_value = ll.Ey

    chi_np = ll.calc_chi_mat(e_hat_np, y_hat_np)
    v_hat_np = ll.calc_steady_state_fluxes(e_hat_np, y_hat_np) / ll.v_star

    chi_ss, v_hat_ss = ll.calculate_steady_state_batch_theano(
        Ez_t, Ey_t, e_hat_t, y_hat_t)
    io_fun = theano.function([Ez_t, Ey_t, e_hat_t, y_hat_t], 
                             [chi_ss, v_hat_ss])

    x_theano_test, v_theano_test = io_fun(ll.Ez, ll.Ey, e_hat_np, y_hat_np)

    assert np.allclose(chi_np, x_theano_test)
    assert np.allclose(v_hat_np[np.isfinite(v_hat_np)],
                       v_theano_test[np.isfinite(v_hat_np)])
