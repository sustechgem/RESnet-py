import numpy as np


def formRectMeshConnectivity(nodeX, nodeY, nodeZ):
    """
    Form the connectivity information for a given rectilinear mesh

    Parameters:
    -----------
    nodeX, nodeY, nodeZ: numpy.ndarray
        node locations in X, Y, Z of a rectilinear mesh


    Returns:
    --------
    nodes: numpy.ndarray
        a 3-column matrix of X-Y-Z locations for the nodes (grid conjunction)
    edges: numpy.ndarray
        a 2-column matrix of node index for the edges; the 1st column for the starting nodes and
        the 2nd column for ending nodes
    lengths: numpy.ndarray
        a vector of the edges' lengths in meter
    faces: numpy.ndarray
        a 4-column matrix of edge index for the faces
    areas: numpy.ndarray
        a vector of the faces' area in square meter
    cells: numpy.ndarray
        a 6-column matrix of face index for the cells
    volumes: numpy.ndarray
        a vector of the cells' volume in cubic meter

    Note:
    -----
    First level ordering: for directional objects (edge and face's normal),
        follow x, y, z-orientation ordering
    Second level ordering: for non-directional objects (cell) and within
        a particular orientation, count in z (top to bottom), then x
        (left to right), then y (front to back)
    """

    # Create nodes lists
    # number of nodes
    Nx = len(nodeX)
    Ny = len(nodeY)
    Nz = len(nodeZ)
    """
    The indexing='ij' parameter in meshgrid() is used to match the indexing convention of MATLAB.
    my_ravel() is used to flatten the arrays a, c, and b into 1D arrays.
    """
    a, b, c = np.meshgrid(nodeX, nodeZ, nodeY)
    a, b, c = np.ravel(a, order='F'), np.ravel(b, order='F'), np.ravel(c, order='F')
    nodes = np.column_stack((a, c, b))  # X-Y-Z location (note ordering)

    # Create edges list (index to nodes)
    # x-direction edges
    a, b, c = np.meshgrid(range(1, Nx), range(1, Nz + 1), range(1, Ny + 1))
    x_cell, y_node, z_node = np.ravel(a, order='F'), np.ravel(c, order='F'), np.ravel(b, order='F')
    x1 = (Nx * Nz) * (y_node - 1) + Nz * (x_cell - 1) + z_node
    x2 = x1 + Nz

    # y-direction edges
    a, b, c = np.meshgrid(range(1, Nx + 1), range(1, Nz + 1), range(1, Ny))
    x_node, y_cell, z_node = np.ravel(a, order='F'), np.ravel(c, order='F'), np.ravel(b, order='F')
    y1 = (Nx * Nz) * (y_cell - 1) + Nz * (x_node - 1) + z_node
    y2 = y1 + Nx * Nz

    # z-direction edges
    a, b, c = np.meshgrid(range(1, Nx + 1), range(1, Nz), range(1, Ny + 1))
    x_node, y_node, z_cell = np.ravel(a, order='F'), np.ravel(c, order='F'), np.ravel(b, order='F')
    z1 = (Nx * Nz) * (y_node - 1) + Nz * (x_node - 1) + z_cell
    z2 = z1 + 1

    # Assemble in order of x-, y-, z-oriented edge
    n1 = np.concatenate((x1, y1, z1))
    n2 = np.concatenate((x2, y2, z2))
    edges = np.column_stack((n1, n2))

    # Create lengths list (in meter)
    lengths = np.sqrt(np.sum((nodes[edges[:, 0] - 1, :] - nodes[edges[:, 1] - 1, :]) ** 2, axis=1))

    # Create faces list (index to edges)
    NEdgesX = (Nx - 1) * Ny * Nz
    NEdgesY = Nx * (Ny - 1) * Nz
    NEdgesZ = Nx * Ny * (Nz - 1)

    # x-face built with y-edge and z-edge
    tmp = np.reshape(np.arange(1, NEdgesY + 1), (Ny - 1, Nx, Nz)).transpose(0, 2, 1)
    tmp = np.delete(tmp, Nz - 1, axis=1)
    xfye1 = my_ravel(tmp) + NEdgesX
    xfye2 = xfye1 + 1
    tmp = np.reshape(np.arange(1, NEdgesZ + 1), (Ny, Nx, Nz - 1)).transpose(0, 2, 1)
    tmp = np.delete(tmp, Ny - 1, axis=0)
    xfze1 = my_ravel(tmp) + NEdgesX + NEdgesY
    xfze2 = xfze1 + (Nz - 1) * Nx

    # y-face built with x-edge and z-edge
    tmp = np.reshape(np.arange(1, NEdgesX + 1), (Ny, Nx - 1, Nz)).transpose(0, 2, 1)
    tmp = np.delete(tmp, Nz - 1, axis=1)
    yfxe1 = my_ravel(tmp)
    yfxe2 = yfxe1 + 1
    tmp = np.reshape(np.arange(1, NEdgesZ + 1), (Ny, Nx, Nz - 1)).transpose(0, 2, 1)
    tmp = np.delete(tmp, Nx - 1, axis=2)
    yfze1 = my_ravel(tmp) + NEdgesX + NEdgesY
    yfze2 = yfze1 + Nz - 1

    # z-face built with x-edge and y-edge
    tmp = np.reshape(np.arange(1, NEdgesX + 1), (Ny, Nx - 1, Nz)).transpose(0, 2, 1)
    tmp = np.delete(tmp, Ny - 1, axis=0)
    zfxe1 = my_ravel(tmp)
    zfxe2 = zfxe1 + Nz * (Nx - 1)
    tmp = np.reshape(np.arange(1, NEdgesY + 1), (Ny - 1, Nx, Nz)).transpose(0, 2, 1)
    tmp = np.delete(tmp, Nx - 1, axis=2)
    zfye1 = my_ravel(tmp) + NEdgesX
    zfye2 = zfye1 + Nz

    # Assemble: four edges per face, faces in x,y,z orientation
    faces = np.hstack([[xfye1, xfye2, xfze1, xfze2],
                       [yfxe1, yfxe2, yfze1, yfze2],
                       [zfxe1, zfxe2, zfye1, zfye2]]).T

    # Create areas list (in meter squared)
    areas = np.multiply(lengths[faces[:, 0] - 1], lengths[faces[:, 2] - 1])  # the 1st and 3rd edges are perpendicular

    # Create cells list (index to faces)
    NFacesX = Nx * (Ny - 1) * (Nz - 1)
    NFacesY = (Nx - 1) * Ny * (Nz - 1)
    NFacesZ = (Nx - 1) * (Ny - 1) * Nz
    # x-face
    tmp = np.arange(1, NFacesX + 1).reshape(Ny - 1, Nx, Nz - 1).transpose(0, 2, 1)
    tmp = np.delete(tmp, Nx - 1, axis=2)
    xf1 = my_ravel(tmp)
    xf2 = xf1 + Nz - 1
    # y-face
    tmp = np.arange(1, NFacesY + 1).reshape(Ny, Nx - 1, Nz - 1).transpose(0, 2, 1)
    tmp = np.delete(tmp, Ny - 1, axis=0)
    yf1 = my_ravel(tmp + NFacesX)
    yf2 = yf1 + (Nz - 1) * (Nx - 1)
    # z-face
    tmp = np.arange(1, NFacesZ + 1).reshape(Ny - 1, Nx - 1, Nz).transpose(0, 2, 1)
    tmp = np.delete(tmp, Nz - 1, axis=1)
    zf1 = my_ravel(tmp + NFacesX + NFacesY)
    zf2 = zf1 + 1

    # assembly: six faces per cell
    cells = np.array([xf1, xf2, yf1, yf2, zf1, zf2]).T

    # Create volumes list (in meter cubed)
    volumes = np.sqrt(areas[cells[:, 0] - 1] * areas[cells[:, 2] - 1] * areas[cells[:, 4] - 1])

    return nodes, edges, lengths, faces, areas, cells, volumes


def my_ravel(ndarray):
    temp = None
    for i in range(len(ndarray)):
        if temp is None:
            temp = ndarray[i].ravel(order='F')
        else:
            temp = np.concatenate((temp, ndarray[i].ravel(order='F')))
    return temp
