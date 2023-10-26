import numpy as np
from scipy.sparse import spdiags, coo_matrix


def formCell2EdgeMatrix(edges, lengths, faces, cells, volumes):
    """
    Form the mapping matrix that transforms cell conductivity model (cellCon in S/m) to conductance on edges.

    Parameters:
    -----------
    edges: numpy.ndarray
        a 2-column matrix of node index for the edges; 1st column for
        starting node and 2nd column for ending node
    lengths: numpy.ndarray
        a vector of the edges' lengths in meter
    faces: numpy.ndarray
        a 4-column matrix of edge index for the faces
    cells: numpy.ndarray
        a 6-column matrix of face index for the cells
    volumes: numpy.ndarray
        a vector of the cells' volume in cubic meter

    Returns:
    --------
    Cell2Edge: scipy.sparse.csr_matrix
        a Nedges x Ncells matrix that acts as volume/4/length/length

    Note:
    -----
        Suppose cellCon is the cell conductivity vector defined on all the
        cells. Cc = Cell2Edge * cellCon gets the conductances of the
        equivalent resistors defined in the cells.
    """

    Nnodes = 8
    Nedges = edges.shape[0]
    Nepf = faces.shape[1]
    Ncells = cells.shape[0]
    Nfpc = cells.shape[1]

    J = faces[np.subtract(cells.reshape(-1, 1), 1), :]
    J = J.reshape(1, -1)
    temp1 = J.reshape(-1, Nepf * Nfpc).T
    old_idx = np.unique(temp1, return_index=True, axis=0)[1]
    temp2 = np.array([temp1[index] for index in sorted(old_idx)]).T
    J = temp2.ravel(order='F')
    I = np.tile(np.arange(1, Ncells + 1).reshape(-1, 1), Nfpc + Nnodes - 2).ravel(order='F')
    J = np.subtract(J, 1)
    I = np.subtract(I, 1)

    Cell2Edge = spdiags(1 / lengths ** 2, 0, Nedges, Nedges) @ coo_matrix((np.ones(len(J)), (J, I)),
                                                                          shape=(Nedges, Ncells),
                                                                          dtype=np.int64).tocsr() @ spdiags(volumes / 4,
                                                                                                          0, Ncells,
                                                                                                          Ncells)

    return Cell2Edge
