import numpy as np
from scipy.sparse import spdiags, coo_matrix


def formFace2EdgeMatrix(edges, lengths, faces, areas):
    """
    Form the mapping matrix that transforms face conductivity model (faceCon in S)
    to conductance on edges.

    Parameters:
    -----------
    edges: numpy.ndarray
        a 2-column matrix of node index for the edges;
        1st column for starting node and 2nd column for ending node
    lengths: numpy.ndarray
        a vector of the edges' lengths in meter
    faces: numpy.ndarray
        a 4-column matrix of edge index for the faces
    areas: numpy.ndarray
        a vector of the faces' area in square meter

    Returns:
    --------
    Face2Edge: scipy.sparse.csr_matrix
        a Nedges x Nfaces matrix that acts as area/2/length/length

    Note:
    -----
        Suppose faceCon is the face conductivity vector defined on all of the
        faces. Cf = Face2Edge * faceCon gets the conductances of the
        equivalent resistors defined on the faces.
    """
    # # of edges and # of faces
    Nedges, _ = edges.shape     # # of edges
    Nfaces, Nepf = faces.shape  # # of faces, # of edges per face

    I = np.subtract(np.repeat(np.arange(Nfaces + 1)[1:, np.newaxis], Nepf, axis=1).flatten(), 1)
    J = np.subtract(faces.flatten(), 1)
    Face2Edge = spdiags(1 / lengths ** 2, 0, Nedges, Nedges) @ coo_matrix((np.ones(len(J)), (J, I)),
                                                                          shape=(Nedges, Nfaces),
                                                                          dtype=np.int64).tocsr() @ spdiags(areas / 2, 0,
                                                                                                          Nfaces,
                                                                                                          Nfaces)
    return Face2Edge
