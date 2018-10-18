import os
import cobra
import numpy as np

import pandas as pd

currdir = os.path.dirname(os.path.abspath(__file__))

def get_N_v(model):

    solution = model.optimize()

    N = cobra.util.create_stoichiometric_matrix(model)
    v_star = solution.fluxes.values

    for i, v in enumerate(v_star):
        if v < 0:
            N[:, i] *= -1
            v_star[i] *= -1

    assert np.all(v_star >= 0)

    return N, v_star


def load_contador():
    model = cobra.io.load_json_model(currdir + '/contador.json')
    model.reactions.EX_glc.bounds = (-1.243, 1000)
    model.reactions.EX_lys.lower_bound = .139
    model.reactions.zwf.lower_bound = .778

    N, v_star = get_N_v(model)

    return model, N, v_star

def load_teusink():
    model = cobra.io.read_sbml_model(currdir + '/BIOMD0000000064.xml')
    model.reactions.vGLT.bounds = (-88.1, 88.1)
    for rxn in model.reactions:
        rxn.lower_bound = 0.1

    model.objective = model.reactions.vATP

    N, v_star = get_N_v(model)

    return model, N, v_star
    
def load_mendes():
    from .mendes_model import model
    N = np.array(model.to_array_based_model().S.todense())
    v_star = np.array([0.0289273 ,  0.0289273 ,  0.01074245,  0.01074245,  0.01074245,
                       0.01818485,  0.01818485,  0.01818485])
    return model, N, v_star
    
def load_textbook():

    model = cobra.io.load_json_model(currdir + '/textbook_reduced.json')
    N, v_star = get_N_v(model)

    return model, N, v_star

def load_greene_small():

    N = np.array([
        [-1, 0, 0, 1, 0, 0],
        [1, 1, -1, 0, 0, 0],
        [1, -1, 0, 0, 0, -1],
        [0, 0, 1, 0, -1, 0]])

    v_star = np.array([1, 0.5, 1.5, 1, 1.5, 0.5])

    rxn_names = ['V1', 'V2', 'V3', 'Vin', 'Vout', 'Voutx3']
    met_names = ['x1', 'x2', 'x3', 'x4']

    assert np.allclose(N @ v_star, 0)

    return construct_model_from_mat(N, rxn_names, met_names), N, v_star


def load_greene_large():

    N = np.array([
        [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, -1, 0],
        [-1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0],
        [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0]])

    v_star = np.array([1, 0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0, 1.5, 0, 1.5,
                       0, 1.5, 0, 1, 0, 1, 0, 1, 0, 1.5, 0.5])

    rxn_names = ['v1_1', 'v1_2', 'v1_3', 'v1_4', 'v1_5', 'v1_6', 'v2_1',
                 'v2_2', 'v2_3', 'v2_4', 'v2_5', 'v2_6', 'v3_1', 'v3_2', 'v3_3',
                 'v3_4', 'v3_5', 'v3_6', 'vin_1', 'vin_2', 'vin_3', 'vin_4',
                 'vin_5', 'vin_6', 'vout', 'voutx3']

    met_names = ['x1', 'x2', 'x3', 'x4', 'E1', 'E2', 'E3', 'Ein', 'x1E1',
                 'x3E1', 'x3E2', 'x2E2', 'x2E3', 'x4E3', 'xoutEin', 'xinEin']

    assert np.allclose(N @ v_star, 0)

    return construct_model_from_mat(N, rxn_names, met_names), N, v_star


def load_jol2012_edit():

    model = cobra.io.load_json_model(currdir + '/jol2012_trimmed.json')
    v_star = pd.read_pickle(currdir + '/jol2012_vstar.p').values

    N = cobra.util.create_stoichiometric_matrix(model)

    assert np.allclose(N @ v_star, 0)

    return model, N, v_star


def construct_model_from_mat(N, rxn_names, met_names):

    model = cobra.Model('test_model')

    model.add_metabolites([cobra.Metabolite(id=met_name) for met_name in met_names])
            
    for row, rxn_name in zip(N.T, rxn_names):
        reaction = cobra.Reaction(id=rxn_name)
        model.add_reaction(reaction)
        reaction.add_metabolites({
            met_id: float(stoich) for met_id, stoich in zip(met_names, row)
            if abs(stoich) > 1E-6})

    return model


models = {
    'teusink': load_teusink,
    'mendes': load_mendes,
    'textbook': load_textbook,
    'greene_small': load_greene_small,
    'greene_large': load_greene_large,
    'contador': load_contador,
    'jol2012': load_jol2012_edit,
}
