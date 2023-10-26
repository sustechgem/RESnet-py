import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse import triu
from PyPardiso import PyPardiso


def solveRESnet(edges, C, sources):
    """
    Solve an arbitrary 3D resistor network circuit problem using the potential's
    formulation and Kirchoff's current law.

    Parameters
    ----------
    edges: numpy.ndarray
        A 2-column matrix of node index for the edges (branches) that describes
        topology of the network; the 1st column for starting node and the 2nd
        column for ending node.
    C: numpy.ndarray
        A vector of conductance values on edges.
    sources: numpy.ndarray
        A vector for the source (current injection amplitude at each node).

    Returns
    -------
    potentials : numpy.ndarray
        Electric potentials on each node (assume zero potential at the first node).
    potentialDiffs : numpy.ndarray
        Potential drops across each edge (branch).
    currents : numpy.ndarray
        Current flowing along each edge (branch).
    """

    Nnodes = np.max(edges)  # # of nodes
    Nedges = edges.shape[0]  # # of edges

    # Form potential difference matrix (node to edge), a.k.a. gradient operator
    I = np.kron(np.arange(1, Nedges+1), [[1], [1]])
    J = edges.T
    S = np.kron(np.ones(Nedges), [[1], [-1]])
    G = csr_matrix((S.flatten(), (I.flatten()-1, J.flatten()-1)), shape=(Nedges, Nnodes))

    Cdiag = spdiags(C, 0, Nedges, Nedges)
    E = csr_matrix(([1], ([0], [0])), shape=(Nnodes, Nnodes))
    G_csc = G.tocsc()
    E_csc = E.tocsc()
    A_csr = G_csc.T @ Cdiag @ G_csc + E_csc
    A_upper_triangular = triu(A_csr, format='csr')

    # Matrix factorization and Solve for multiple rhs
    pardiso_solver = PyPardiso(A_upper_triangular, matrix_type=2)
    if sources.shape[1] == 1:
        potentials = pardiso_solver.solve(sources)
    else:
        potentials_tmp = []
        for i in range(sources.shape[1]):
            source = sources[:, i]
            source = source.astype(np.float64)
            potentials_tmp.append(pardiso_solver.solve(source))
        potentials = [list(column) for column in zip(*potentials_tmp)]
        potentials = np.array(potentials)
    pardiso_solver.release()  # Release memory

    # Compute potential difference (E field) on all edges
    potentialDiffs = G @ potentials

    # Compute current on all edges
    currents = Cdiag @ potentialDiffs

    return potentials, potentialDiffs, currents
