from scipy.sparse import spdiags


def formEdge2EdgeMatrix(edges, lengths):
    """
    Form the mapping matrix that transforms edge conductivity model (edgeCon in S*m)
    to conductance on edges.

    Parameters:
    -----------
    edges: numpy.ndarray
        A 2-column matrix of node index for the edges; 1st column for starting node and 2nd column for ending node
    lengths: numpy.ndarray
        A vector of the edges' lengths in meter

    Returns:
    --------
    Edge2Edge: scipy.sparse.csr_matrix
        A Nedges x Nedges matrix that acts as 1/length

    Note:
    -----
    Suppose edgeCon is the edge conductivity vector defined on all of the edges. Ce = Edge2Edge * edgeCon
    gets the conductances of the equivalent resistors defined on the edges.
    """

    Nedges = edges.shape[0]
    Edge2Edge = spdiags(1 / lengths, 0, Nedges, Nedges, format='csr')
    return Edge2Edge
