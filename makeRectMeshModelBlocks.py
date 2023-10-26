import numpy as np

from formRectMeshConnectivity import formRectMeshConnectivity


def makeRectMeshModelBlocks(nodeX, nodeY, nodeZ, blkLoc, blkVal, bkgCellVal=0, bkgFaceVal=0, bkgEdgeVal=0):
    """
    Make edgeCon, faceCon, cellCon models using blocks

    Parameters:
    -----------
    nodeX, nodeY, nodeZ: numpy.ndarray
        Node locations in X, Y, Z of a rectilinear mesh
    blkLoc: numpy.ndarray
        A Nblock x 6 matrix whose columns are [xmin xmax ymin ymax zmax zmin]
        specifying the range of the block; if the range of any dimension is zero,
        that dimension vanishes to represent a thin object; one dimension vanishes
        for 2D sheet object; two dimensions vanish for 1D line object; point object
        not allowed.
    blkVal: numpy.ndarray
        A vector specifying the physical property values of the blocks.
    bkgCellVal: list
        Background model values defined at cell centers; for the
        description of existing model structure; can be a scalar for the
        whole-space or a vector; treat empty [] as 0. (default is 0)
    bkgFaceVal: list
        Background model values defined at cell faces; for the
        description of existing model structure; can be a scalar for the
        whole-space or a vector; treat empty [] as 0. (default is 0)
    bkgEdgeVal: list
        Background model values defined at cell edges; for the
        description of existing model structure; can be a scalar for the
        whole-space or a vector; treat empty [] as 0. (default is 0)

    Returns:
    --------
    cellVal: numpy.ndarray
        A vector of physical property values defined on all cells
    faceVal: numpy.ndarray
        A vector of physical property values defined on all faces (cellVal * thickness)
    edgeVal: numpy.ndarray
        A vector of physical property values defined on all edges (cellVal * cross-sectional area)

    Notes:
    ------
    Entries in edgeCon and faceCon are directional.
    First level ordering: for directional objects (edge and face's normal),
        follow x, y, z-orientation ordering
    Second level ordering: for non-directional objects (cell) and within
        a particular orientation, count in z (top to bottom), then x
        (left to right), then y (front to back)
    """

    # Prep
    Nx = len(nodeX)
    Ny = len(nodeY)
    Nz = len(nodeZ)
    Nedges = (Nx - 1) * Ny * Nz + Nx * (Ny - 1) * Nz + Nx * Ny * (Nz - 1)
    Nfaces = (Nx - 1) * Ny * (Nz - 1) + Nx * (Ny - 1) * (Nz - 1) + (Nx - 1) * (Ny - 1) * Nz
    Ncells = (Nx - 1) * (Ny - 1) * (Nz - 1)

    if not bkgCellVal:
        bkgCellVal = 0
    if not bkgFaceVal:
        bkgFaceVal = 0
    if not bkgEdgeVal:
        bkgEdgeVal = 0
    edgeVal = np.zeros(Nedges) + bkgEdgeVal
    faceVal = np.zeros(Nfaces) + bkgFaceVal
    cellVal = np.zeros(Ncells) + bkgCellVal

    # Get connectivity lists from mesh definition
    nodes, edges, _, faces, _, cells, _ = formRectMeshConnectivity(nodeX, nodeY, nodeZ)

    # Replace inf with the outmost boundary
    blkLoc = np.atleast_2d(blkLoc)  # Converts a one-dimensional array to a two-dimensional array
    blkLoc[blkLoc[:, 0] == -np.inf, 0] = nodeX[0]
    blkLoc[blkLoc[:, 1] == np.inf, 1] = nodeX[-1]
    blkLoc[blkLoc[:, 2] == -np.inf, 2] = nodeY[0]
    blkLoc[blkLoc[:, 3] == np.inf, 3] = nodeY[-1]
    blkLoc[blkLoc[:, 4] == np.inf, 4] = nodeZ[0]
    blkLoc[blkLoc[:, 5] == -np.inf, 5] = nodeZ[-1]

    # Pre-screen to identify object types: 1D, 2D or 3D
    dim = np.vstack((~(abs(blkLoc[:, 0] - blkLoc[:, 1]) == 0), ~(abs(blkLoc[:, 2] - blkLoc[:, 3]) == 0),
                     ~(abs(blkLoc[:, 4] - blkLoc[:, 5]) == 0))).T
    # 0 indicates that dimension vanished
    objType = np.sum(dim, axis=1)  # dimensionality = 3 for volume, 2 for sheet, 1 for string

    # Object center positions
    edgesCenter = 1 / 2 * (nodes[edges[:, 0] - 1, :] + nodes[edges[:, 1] - 1, :])
    facesCenter = 1 / 4 * (edgesCenter[faces[:, 0] - 1, :] + edgesCenter[faces[:, 1] - 1, :] +
                           edgesCenter[faces[:, 2] - 1, :] + edgesCenter[faces[:, 3] - 1, :])

    cellsCenter = 1 / 6 * (facesCenter[cells[:, 0] - 1, :] + facesCenter[cells[:, 1] - 1, :] +
                           facesCenter[cells[:, 2] - 1, :] + facesCenter[cells[:, 3] - 1, :] +
                           facesCenter[cells[:, 4] - 1, :] + facesCenter[cells[:, 5] - 1, :])

    # Loop over blocks to make additions
    Nblk = blkLoc.shape[0]
    tol = 0.001  # allow small inaccuracy when locating sheets and lines

    for i in range(Nblk):
        xmin = np.min(blkLoc[i, 0:2])
        xmax = np.max(blkLoc[i, 0:2])
        ymin = np.min(blkLoc[i, 2:4])
        ymax = np.max(blkLoc[i, 2:4])
        zmax = np.max(blkLoc[i, 4:6])
        zmin = np.min(blkLoc[i, 4:6])

        xminInd = np.argmin(np.abs(nodeX - xmin))
        xmaxInd = np.argmin(np.abs(nodeX - xmax))
        yminInd = np.argmin(np.abs(nodeY - ymin))
        ymaxInd = np.argmin(np.abs(nodeY - ymax))
        zminInd = np.argmin(np.abs(nodeZ - zmin))
        zmaxInd = np.argmin(np.abs(nodeZ - zmax))

        if objType[i] == 3:  # volume -> add to cellCon
            ind = np.logical_and(
                np.logical_and(cellsCenter[:, 0] >= nodeX[xminInd], cellsCenter[:, 0] <= nodeX[xmaxInd]),
                np.logical_and(cellsCenter[:, 1] >= nodeY[yminInd], cellsCenter[:, 1] <= nodeY[ymaxInd])
            )
            ind = np.logical_and(ind, cellsCenter[:, 2] <= nodeZ[zmaxInd])
            ind = np.logical_and(ind, cellsCenter[:, 2] >= nodeZ[zminInd])
            cellVal[ind] = blkVal[i]

        elif objType[i] == 2:  # sheet -> add to faceCon
            ind = np.logical_and(
                np.logical_and(facesCenter[:, 0] >= nodeX[xminInd] - tol, facesCenter[:, 0] <= nodeX[xmaxInd] + tol),
                np.logical_and(facesCenter[:, 1] >= nodeY[yminInd] - tol, facesCenter[:, 1] <= nodeY[ymaxInd] + tol)
            )
            ind = np.logical_and(ind, facesCenter[:, 2] <= nodeZ[zmaxInd] + tol)
            ind = np.logical_and(ind, facesCenter[:, 2] >= nodeZ[zminInd] - tol)
            faceVal[ind] = blkVal[i]

        elif objType[i] == 1:  # string -> add to edgeCon
            ind = np.logical_and(
                np.logical_and(edgesCenter[:, 0] >= nodeX[xminInd] - tol, edgesCenter[:, 0] <= nodeX[xmaxInd] + tol),
                np.logical_and(edgesCenter[:, 1] >= nodeY[yminInd] - tol, edgesCenter[:, 1] <= nodeY[ymaxInd] + tol)
            )
            ind = np.logical_and(ind, edgesCenter[:, 2] <= nodeZ[zmaxInd] + tol)
            ind = np.logical_and(ind, edgesCenter[:, 2] >= nodeZ[zminInd] - tol)
            edgeVal[ind] = blkVal[i]

        elif objType[i] == 0:  # point -> no action
            pass

    return cellVal, faceVal, edgeVal
