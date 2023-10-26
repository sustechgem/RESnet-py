import numpy as np
from scipy.sparse import csc_matrix


def calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, points):
    """
    Core function of the trilinear interpolation. Calculate the weights of
    the eight neighboring nodes surrounding a given point in a lattice grid.

    Parameters:
    -----------
    nodeX, nodeY, nodeZ: numpy.ndarray
        node locations in X, Y, Z of a rectilinear mesh
    points: numpy.ndarray
        a Npoints x 3 matrix specifying the X, Y, Z coordinates of points

    Returns:
    --------
    weights: scipy.sparse.csc_matrix
        a Nnodes x Npoints sparse matrix of the calculated weights

    Note:
    -----
    weights: distribute the point's contrbution to neighboring grid nodes
    weights' (transpose): estimate the point's value from the neighboring nodes
    """

    # Prep
    Nnx = len(nodeX)
    Nny = len(nodeY)
    Nnz = len(nodeZ)
    Npoint = points.shape[0]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Snap out-of-region points to the nearest boundary
    x = np.maximum(x, nodeX[0])
    x = np.minimum(x, nodeX[-1])
    y = np.maximum(y, nodeY[0])
    y = np.minimum(y, nodeY[-1])
    z = np.minimum(z, nodeZ[0])
    z = np.maximum(z, nodeZ[-1])

    # Nodes-to-points distance in x, y, z-direction
    nxd = np.abs(nodeX.reshape(1, -1) - x.reshape(-1, 1))
    nyd = np.abs(nodeY.reshape(1, -1) - y.reshape(-1, 1))
    nzd = np.abs(nodeZ.reshape(1, -1) - z.reshape(-1, 1))

    # Find the two the nearest nodes and the distances to the nodes in x, y, z-direction
    xd, xn = nearest_nodes(nxd)
    yd, yn = nearest_nodes(nyd)
    zd, zn = nearest_nodes(nzd)

    # Calculate weights of eight neighboring nodes
    sxd = np.sum(xd, axis=1)
    syd = np.sum(yd, axis=1)
    szd = np.sum(zd, axis=1)
    w1 = xd[:, 1] / sxd * yd[:, 1] / syd * zd[:, 1] / szd
    w2 = xd[:, 1] / sxd * yd[:, 1] / syd * zd[:, 0] / szd
    w3 = xd[:, 0] / sxd * yd[:, 1] / syd * zd[:, 1] / szd
    w4 = xd[:, 0] / sxd * yd[:, 1] / syd * zd[:, 0] / szd
    w5 = xd[:, 1] / sxd * yd[:, 0] / syd * zd[:, 1] / szd
    w6 = xd[:, 1] / sxd * yd[:, 0] / syd * zd[:, 0] / szd
    w7 = xd[:, 0] / sxd * yd[:, 0] / syd * zd[:, 1] / szd
    w8 = xd[:, 0] / sxd * yd[:, 0] / syd * zd[:, 0] / szd

    # Calculate indices of eight neighboring nodes
    n1 = (Nnx * Nnz) * (yn[:, 0]) + Nnz * (xn[:, 0]) + zn[:, 0]
    n2 = (Nnx * Nnz) * (yn[:, 0]) + Nnz * (xn[:, 0]) + zn[:, 1]
    n3 = (Nnx * Nnz) * (yn[:, 0]) + Nnz * (xn[:, 1]) + zn[:, 0]
    n4 = (Nnx * Nnz) * (yn[:, 0]) + Nnz * (xn[:, 1]) + zn[:, 1]
    n5 = (Nnx * Nnz) * (yn[:, 1]) + Nnz * (xn[:, 0]) + zn[:, 0]
    n6 = (Nnx * Nnz) * (yn[:, 1]) + Nnz * (xn[:, 0]) + zn[:, 1]
    n7 = (Nnx * Nnz) * (yn[:, 1]) + Nnz * (xn[:, 1]) + zn[:, 0]
    n8 = (Nnx * Nnz) * (yn[:, 1]) + Nnz * (xn[:, 1]) + zn[:, 1]

    # Put weights in a sparse matrix: Nnodes x Npoints
    s1 = csc_matrix((w1, (n1, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))
    s2 = csc_matrix((w2, (n2, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))
    s3 = csc_matrix((w3, (n3, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))
    s4 = csc_matrix((w4, (n4, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))
    s5 = csc_matrix((w5, (n5, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))
    s6 = csc_matrix((w6, (n6, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))
    s7 = csc_matrix((w7, (n7, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))
    s8 = csc_matrix((w8, (n8, np.arange(Npoint))), shape=(Nnx * Nny * Nnz, Npoint))

    # Assemble
    weights = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8
    return weights.tocsr()


def nearest_nodes(direction):
    dd = np.apply_along_axis(lambda i: np.sort(i, kind='stable')[:2], axis=1, arr=direction)
    nn = np.apply_along_axis(lambda i: np.argsort(i, kind='stable')[:2], axis=1, arr=direction)
    ind = nn[:, 0] > nn[:, 1]
    nn = conversion(ind, nn)
    dd = conversion(ind, dd)
    return dd, nn


def conversion(logical_index, array):
    tmp1 = array[logical_index, 0]
    tmp2 = array[logical_index, 1]
    array[logical_index, 0] = tmp2
    array[logical_index, 1] = tmp1
    return array
