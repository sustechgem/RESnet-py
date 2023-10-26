import time

import numpy as np

from calcTrilinearInterpWeights import calcTrilinearInterpWeights
from formCell2EdgeMatrix import formCell2EdgeMatrix
from formEdge2EdgeMatrix import formEdge2EdgeMatrix
from formFace2EdgeMatrix import formFace2EdgeMatrix
from formRectMeshConnectivity import formRectMeshConnectivity
from makeRectMeshModelBlocks import makeRectMeshModelBlocks
from solveRESnet import solveRESnet

if __name__ == '__main__':
    """
    A minimum working template for running RESnet
    For demonstration and testing only. Results not physically meaningful.
    """

    '''Set up the geo-electrical model'''
    # Create a 3D rectilinear mesh
    nodeX = np.linspace(-100, 100, num=21)  # node locations in X
    nodeY = np.linspace(-100, 100, num=21)  # node locations in Y
    nodeZ = np.linspace(0, -100, num=21)    # node locations in Z

    # Define the model using combination of blocks
    # % A model includes some blocks that can represent objects like sheets or lines when one or two dimensions vanish.
    blkLoc = np.array(
        [[-np.inf, np.inf, -np.inf, np.inf, 0, -np.inf],
         # a volumetric object defined by [xmin xmax ymin ymax zmax zmin]
         [-20, 20, -20, 20, -50, -50],  # a sheet object whose zmax = zmin
         [0, 0, 0, 0, 0, -80]])  # a line object whose xmin = xmax and ymin = ymax

    blkCon = np.array([[1e-2],  # conductive property of the volumetric object (S/m)
                       [10],  # conductive property of the sheet object (S)
                       [5e4]])  # conductive property of the line object (S*m)

    '''Setup the electric surveys'''
    # Define the current sources in the format of [x y z current(Ampere)]
    tx = np.array([np.array([[0, 0, 0, 1],  # first set of source (two electrodes)
                             [40, 0, 0, -1]]),
                   np.array([[-10, 10, 0, 0.5],  # second set of source (three electrodes)
                             [10, 10, 0, 0.5],
                             [0, 0, -50, -1]])], dtype=object)

    # Define the receiver electrodes in the format of [Mx My Mz Nx Ny Nz]
    rx = np.array([np.array([[10, 0, 0, 20, 0, 0],  # first set of receivers (yielding three data values)
                             [20, 0, 0, 30, 0, 0],
                             [30, 0, 0, 40, 0, 0]]),
                   np.array([[0, 0, -10, 0, 0, -20],  # second set of receivers (yielding two data values)
                             [0, 0, -20, 0, 0, -30]])], dtype=object)

    '''Form a resistor network'''
    # Get connectivity properties of nodes, edges, faces, cells
    nodes, edges, lengths, faces, areas, cells, volumes = formRectMeshConnectivity(nodeX, nodeY, nodeZ)

    # Get conductive property model vectors (convert the block-model description to values on edges, faces and cells)
    cellCon, faceCon, edgeCon = makeRectMeshModelBlocks(nodeX, nodeY, nodeZ, blkLoc, blkCon, [], [], [])

    # Convert all conductive objects to conductance on edges
    Edge2Edge = formEdge2EdgeMatrix(edges, lengths)
    Face2Edge = formFace2EdgeMatrix(edges, lengths, faces, areas)
    Cell2Edge = formCell2EdgeMatrix(edges, lengths, faces, cells, volumes)

    Ce = Edge2Edge @ edgeCon  # conductance from edges
    Cf = Face2Edge @ faceCon  # conductance from faces
    Cc = Cell2Edge @ cellCon  # conductance from cells
    C = Ce + Cf + Cc  # total conductance

    '''Solve the resistor network problem'''
    # Calculate current sources on the nodes using info in tx
    Ntx = len(tx)  # number of tx-rx sets
    sources = np.zeros((nodes.shape[0], Ntx))
    for i in range(Ntx):
        # weights for the distribution of point current source to the neighboring nodes
        weights = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, tx[i][:, :3])
        sources[:, i] = weights.dot(tx[i][:, 3])  # total current intensities at all the nodes

    # Solve multiple tx-rx sets for the same model
    # Obtain potentials at the nodes, potential differences and current along the edges
    start_time = time.time()
    potentials, potentialDiffs, currents = solveRESnet(edges, C, sources)
    end_time = time.time()
    print(f"Time: {(end_time - start_time):.6f} seconds")

    # Get simulated data
    data = []
    for i in range(Ntx):
        Mw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, :3])
        # weights for the interpolation of potential data at the M-electrode location
        Nw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, 3:6])
        # weights for the interpolation of potential data at the N-electrode location
        data.append(np.dot((Mw.T - Nw.T), potentials[:, i]))  # calculate the potential difference data as "M - N"

    print('RESnet-m-py Test Passed!')