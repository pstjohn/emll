import os
import cobra
import numpy as np

currdir = os.path.dirname(os.path.abspath(__file__))

def load_contador():
    model = cobra.io.load_json_model(currdir + '/contador.json')
    N = np.array(model.to_array_based_model().S.
                 todense())
    model.optimize()
    v_star = np.array(list(model.solution.x))

    return model, N, v_star

def load_teusink():
    model = cobra.io.read_sbml_model(currdir + '/BIOMD0000000064.xml')
    N = np.array(model.to_array_based_model().S.
                 todense())

    model.reactions.vGLT.bounds = (-88.1, 88.1)

    for rxn in model.reactions:
        rxn.lower_bound = 0.1

    model.objective = model.reactions.vATP
    model.optimize()
    v_star = np.array(list(model.solution.x))

    return model, N, v_star
    
def load_mendes():
    from .mendes_model import model
    N = np.array(model.to_array_based_model().S.todense())
    v_star = np.array([0.0289273 ,  0.0289273 ,  0.01074245,  0.01074245,  0.01074245,
                       0.01818485,  0.01818485,  0.01818485])
    return model, N, v_star
    
def load_textbook():
    import cobra.test
    model = cobra.test.create_test_model('textbook')
    
    # model.reactions.Biomass_Ecoli_core.remove_from_model()
    # model.objective = model.reactions.ATPM
    
    model.optimize()
    
    N = np.array(model.to_array_based_model().S.todense())
    v_star = np.array(list(model.solution.x))

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

    rxn_names = ['v1,1', 'v1,2', 'v1,3', 'v1,4', 'v1,5', 'v1,6', 'v2,1',
            'v2,2', 'v2,3', 'v2,4', 'v2,5', 'v2,6', 'v3,1', 'v3,2', 'v3,3',
            'v3,4', 'v3,5', 'v3,6', 'vin,1', 'vin,2', 'vin,3', 'vin,4',
            'vin,5', 'vin,6', 'vout', 'voutx3']

    met_names = ['x1', 'x2', 'x3', 'x4', 'E1', 'E2', 'E3', 'Ein', 'x1E1',
            'x3E1', 'x3E2', 'x2E2', 'x2E3', 'x4E3', 'xoutEin', 'xinEin']

    assert np.allclose(N @ v_star, 0)

    return construct_model_from_mat(N, rxn_names, met_names), N, v_star


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
