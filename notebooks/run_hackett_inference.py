# Code to run the ADVI inference with a near-genome scale model and relative
# omics data.

# So I've found that for certain hardware (the intel chips on the cluster here,
# for instance) the intel python and mkl-numpy are about 2x as fast as the
# openblas versions. You can delete a bunch of this stuff if it doesn't work
# for you. This example is a lot slower than some of the other ones though, but
# I guess that's expected

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as T

import cobra
import emll

# Load model and data
model = cobra.io.load_yaml_model('data/jol2012.yaml')

r_compartments = [
    r.compartments if 'e' not in r.compartments else 't'
    for r in model.reactions
]

r_compartments[model.reactions.index('SUCCt2r')] = 'c'
r_compartments[model.reactions.index('ACt2r')] = 'c'

for rxn in model.exchanges:
    r_compartments[model.reactions.index(rxn)] = 't'

m_compartments = [
    m.compartment for m in model.metabolites
]

v_star = pd.read_csv('data/v_star.csv', header=None, index_col=0)[1]

x = pd.read_csv('data/metabolite_concentrations.csv', index_col=0)
v = pd.read_csv('data/boundary_fluxes.csv', index_col=0)
e = pd.read_csv('data/enzyme_measurements.csv', index_col=0)

# Reindex arrays to have the same column ordering
to_consider = v.columns
v = v.loc[:, to_consider]
x = x.loc[:, to_consider]
e = e.loc[:, to_consider]

n_exp = len(to_consider) - 1
ref_state = 'P0.11'

xn = (x.subtract(x['P0.11'], 0) * np.log(2)).T
en = (2 ** e.subtract(e['P0.11'], 0)).T

# To calculate vn, we have to merge in the v_star series and do some
# calculations.
v_star_df = pd.DataFrame(v_star).reset_index().rename(columns= {0: 'id', 1:'flux'})
v_merge = v.merge(v_star_df, left_index=True, right_on='id').set_index('id')
vn = v_merge.divide(v_merge.flux, 0).drop('flux', 1).T

# Drop reference state
vn = vn.drop(ref_state)
xn = xn.drop(ref_state)
en = en.drop(ref_state)

# Get indexes for measured values
x_inds = np.array([model.metabolites.index(met) for met in xn.columns])
e_inds = np.array([model.reactions.index(rxn) for rxn in en.columns])
v_inds = np.array([model.reactions.index(rxn) for rxn in vn.columns])

e_laplace_inds = []
e_zero_inds = []

for i, rxn in enumerate(model.reactions):
    if rxn.id not in en.columns:
        if ('e' not in rxn.compartments) and (len(rxn.compartments) == 1):
            e_laplace_inds += [i]
        else:
            e_zero_inds += [i]

e_laplace_inds = np.array(e_laplace_inds)
e_zero_inds = np.array(e_zero_inds)
e_indexer = np.hstack([e_inds, e_laplace_inds, e_zero_inds]).argsort()

N = cobra.util.create_stoichiometric_matrix(model)
Ex = emll.util.create_elasticity_matrix(model)
Ey = emll.util.create_Ey_matrix(model)

Ex *= 0.1 + 0.8 * np.random.rand(*Ex.shape)

ll = emll.LinLogLeastNorm(N, Ex, Ey, v_star.values, driver='gelsy')

np.random.seed(1)


# Define the probability model
from emll.util import initialize_elasticity

with pm.Model() as pymc_model:

    # Priors on elasticity values       
    Ex_t = pm.Deterministic('Ex', initialize_elasticity(
        ll.N,  b=0.01, sd=1, alpha=None,
        m_compartments=m_compartments,
        r_compartments=r_compartments
    ))

    Ey_t = T.as_tensor_variable(Ey)

    e_measured = pm.Normal('log_e_measured', mu=np.log(en), sd=0.2,
                           shape=(n_exp, len(e_inds)))
    e_unmeasured = pm.Laplace('log_e_unmeasured', mu=0, b=0.1,
                              shape=(n_exp, len(e_laplace_inds)))
    log_en_t = T.concatenate(
        [e_measured, e_unmeasured,
         T.zeros((n_exp, len(e_zero_inds)))], axis=1)[:, e_indexer]

    pm.Deterministic('log_en_t', log_en_t)

    # Priors on external concentrations
    yn_t = pm.Normal('yn_t', mu=0, sd=10, shape=(n_exp, ll.ny),
                     testval=0.1 * np.random.randn(n_exp, ll.ny))


    chi_ss, vn_ss = ll.steady_state_theano(Ex_t, Ey_t, T.exp(log_en_t), yn_t)
    pm.Deterministic('chi_ss', chi_ss)
    pm.Deterministic('vn_ss', vn_ss)

    log_vn_ss = T.log(T.clip(vn_ss[:, v_inds], 1E-8, 1E8))
    log_vn_ss = T.clip(log_vn_ss, -1.5, 1.5)
    
    chi_clip = T.clip(chi_ss[:, x_inds], -1.5, 1.5)

    chi_obs = pm.Normal('chi_obs', mu=chi_clip, sd=0.2,
                        observed=xn.clip(lower=-1.5, upper=1.5))
    log_vn_obs = pm.Normal('vn_obs', mu=log_vn_ss, sd=0.1,
                           observed=np.log(vn).clip(lower=-1.5, upper=1.5))


if __name__ == "__main__":

    with pymc_model:

        approx = pm.ADVI()
        hist = approx.fit(
            n=40000,
            obj_optimizer=pm.adagrad_window(learning_rate=0.005),
            total_grad_norm_constraint=100
        )

        # trace = hist.sample(500)
        # ppc = pm.sample_ppc(trace)


    import gzip
    import pickle
    with gzip.open('data/hackett_advi.pgz', 'wb') as f:
        pickle.dump({'approx': approx,
         'hist': hist,
         'trace': trace}, f)
