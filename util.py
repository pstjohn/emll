import numpy as np

def create_elasticity_matrix(model):
    """Create an elasticity matrix given the model in model.

    E[j,i] represents the elasticity of reaction j for metabolite i.

    """

    n_metabolites = len(model.metabolites)
    n_reactions = len(model.reactions)
    array = np.zeros((n_reactions, n_metabolites), dtype=float)

    m_ind = model.metabolites.index
    r_ind = model.reactions.index

    for reaction in model.reactions:
        for metabolite, stoich in reaction.metabolites.items():

            # Reversible reaction, assign all elements to -stoich
            if reaction.reversibility:
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

            # Irrevesible in forward direction, only assign if met is reactant
            elif ((not reaction.reversibility) & 
                  (reaction.upper_bound > 0) &
                  (stoich < 0)):
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

            # Irreversible in reverse direction, only assign in met is product
            elif ((not reaction.reversibility) & 
                  (reaction.lower_bound < 0) &
                  (stoich > 0)):
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

    return array

def create_Ey_matrix(model):

    boundary_indexes = [model.reactions.index(r) for r in model.exchanges]
    boundary_directions = [1 if r.products else -1 for r in
                           model.reactions.query(
                               lambda x: x.boundary, None)]
    ny = len(boundary_indexes)
    Ey = np.zeros((len(model.reactions), ny))

    for i, (rid, direction) in enumerate(zip(boundary_indexes,
                                             boundary_directions)):
        Ey[rid, i] = direction

    return Ey
