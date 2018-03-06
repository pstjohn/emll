import numpy as np
import scipy as sp

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
    """ This function should return a good guess for the Ey matrix. This
    essentially requires considering the effects of the reactants / products
    for the unbalanced exchange reactions, and is probably best handled
    manually for now. """

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


def compute_waldherr_reduction(N, tol=1E-8):
    """ Uses the SVD to calculate a reduced stoichiometric matrix, link, and
    conservation matrices.
    
    Returns:
    Nr, L, G
    
    """
    u, e, vh = sp.linalg.svd(N)
    E = sp.linalg.diagsvd(e, *N.shape)

    Nr = (E[e > tol] @ vh)
    L = u[:, e > tol]
    G = u[:, e >= tol]

    return Nr, L, G


def compute_smallbone_reduction(N, Ex, v_star, tol=1E-8):
    """ Uses the SVD to calculate a reduced stoichiometric matrix, then
    calculates a link matrix as described in Smallbone *et al* 2007.
    
    Returns:
    Nr, L, P
    
    """
    q, r, p = sp.linalg.qr((N @ np.diag(v_star) @ Ex).T,
                           pivoting=True)

    # Construct permutation matrix
    P = np.zeros((len(p), len(p)), dtype=int)
    for i, pi in enumerate(p):
        P[i, pi] = 1

    # Get the matrix rank from the r matrix
    maxabs = np.max(np.abs(np.diag(r)))
    maxdim = max(N.shape)
    tol = maxabs * maxdim * np.MachAr().eps
    # Find where the rows of r are all less than tol
    rank = (~(np.abs(r) < tol).all(1)).sum()

    Nr = P[:rank] @ N
    L = N @ np.linalg.pinv(Nr)

    return Nr, L, P


import theano.tensor as T
import pymc3 as pm

def initialize_elasticity(N, name=None, b=0.1):
    
    if name is None:
        name = 'ex'
    
    e_guess = -N.T

    e_flat = e_guess.flatten()
    nonzero_inds = np.where(e_flat != 0)[0]
    zero_inds = np.where(e_flat == 0)[0]
    
    e_sign = np.sign(e_flat[nonzero_inds])
    flat_indexer = np.hstack([nonzero_inds, zero_inds]).argsort()
    num_zero = len(zero_inds)
    num_nonzero = len(nonzero_inds)
        
    e_kin_entries = pm.HalfNormal(name + '_kinetic_entries', sd=1, shape=num_nonzero,
                                  testval=np.abs(np.random.randn(num_nonzero)))
    
    e_cap_entries = pm.Laplace(name + '_capacity_entries', mu=0, b=b, shape=num_zero,
                               testval=b * np.random.randn(num_zero))
    
    flat_e_entries = T.concatenate([e_kin_entries * e_sign, e_cap_entries])
        
    E = flat_e_entries[flat_indexer].reshape(N.T.shape)
    
    return E
