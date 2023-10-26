import numpy as np
import time

from matplotlib import pyplot as plt

from calcTrilinearInterpWeights import calcTrilinearInterpWeights
from formCell2EdgeMatrix import formCell2EdgeMatrix
from formEdge2EdgeMatrix import formEdge2EdgeMatrix
from formFace2EdgeMatrix import formFace2EdgeMatrix
from formRectMeshConnectivity import formRectMeshConnectivity
from makeRectMeshModelBlocks import makeRectMeshModelBlocks
from solveRESnet import solveRESnet

if __name__ == '__main__':
    """
    Demonstration of complex infrastructure modeling
    In the following, a dipole-dipole survey is simulated for 
    Model #1: A uniform half-space
    Model #2: Model #1 with two anomalous blocks buried
    Model #3: Model #2 with infrastructure (a steel cased well and a steel-sheet warehouse)
    Model #4: Model #3 with an above-ground pipe connecting the well and the warehouse
    """

    '''Setup the 3D mesh'''
    # Create a 3D rectilinear mesh
    h = 1  # smallest horizontal grid size
    ratio = 1.4  # expansion rate
    nctbc = 15  # number of cells to boundary condition
    nodeX = np.round(np.concatenate(
        (-np.cumsum(h * np.power(ratio, (np.arange(nctbc, -1, -1)))[::-1])[::-1] - 45,
         np.arange(-45, 45 + h, h),
         45 + np.cumsum(h * np.power(ratio, np.arange(nctbc + 1)))),
        axis=None))  # node locations in X
    nodeY = np.round(np.concatenate(
        (-np.cumsum(h * np.power(ratio, (np.arange(nctbc, -1, -1)))[::-1])[::-1] - 12,
         np.arange(-12, 12 + h, h),
         12 + np.cumsum(h * np.power(ratio, np.arange(nctbc + 1)))),
        axis=None))  # node locations in Y
    h = 1  # smallest vertical grid size
    ratio = 1.4  # expansion rate
    nctbc = 15  # number of cells to boundary condition
    nodeZ = np.concatenate((np.arange(4, -11, -h), -10 - np.round(np.cumsum(h * ratio ** (np.arange(nctbc + 1))))),
                           axis=0)  # node locations in Z (top of mesh at 4 m above the surface)

    '''Setup the electric surveys (dipole-dipole)'''
    spacing = 4  # meters between two electrodes
    Aloc = np.arange(-40, 29, spacing)
    Ntx = len(Aloc)  # number of source combos
    tx = [None] * Ntx
    rx = [None] * Ntx

    for i in range(Ntx):
        tx[i] = np.array([[Aloc[i], 0, 0, 1],  # A electrode
                          [Aloc[i] + spacing, 0, 0, -1]])  # B electrode
        Mloc = np.arange(Aloc[i] + spacing * 2, 37, spacing)  # M electrodes
        Nloc = np.arange(Aloc[i] + spacing * 3, 41, spacing)  # N electrodes
        rx[i] = np.column_stack((Mloc, np.zeros_like(Mloc), np.zeros_like(Mloc),
                                 Nloc, np.zeros_like(Nloc), np.zeros_like(Nloc)))  # Mx My(=0) Mz(=0) Nx Ny(=0) Nz(=0)
    Ndata = sum(range(1, Ntx + 1))  # total number of data

    '''Define Model #1: Uniform half-space'''
    blkLoc = [[-np.inf, np.inf, -np.inf, np.inf, np.inf, 0],  # a uniform layer for the air (above surface)
              [-np.inf, np.inf, -np.inf, np.inf, 0, -np.inf]]  # a uniform half-space below surface

    blkCon = [[1e-5],  # air conductivity
              [0.01]]  # earth half-space conductivity

    '''Form a resistor network'''
    # Get connectivity properties of nodes, edges, faces, cells
    nodes, edges, lengths, faces, areas, cells, volumes = formRectMeshConnectivity(nodeX, nodeY, nodeZ)

    # Get conductive property model vectors (convert the block-model description to values on edges, faces and cells)
    cellCon, faceCon, edgeCon = makeRectMeshModelBlocks(nodeX, nodeY, nodeZ, blkLoc, blkCon, [], [], [])

    # Convert conductive objects to conductance on edges
    Edge2Edge = formEdge2EdgeMatrix(edges, lengths)
    Face2Edge = formFace2EdgeMatrix(edges, lengths, faces, areas)
    Cell2Edge = formCell2EdgeMatrix(edges, lengths, faces, cells, volumes)
    Ce = Edge2Edge.dot(edgeCon)  # conductance from edges
    Cf = Face2Edge.dot(faceCon)  # conductance from faces
    Cc = Cell2Edge.dot(cellCon)  # conductance from cells
    C = Ce + Cf + Cc  # total conductance

    '''Solve the resistor network problem'''
    # Calculate current sources on the nodes using info in tx
    Ntx = len(tx)  # number of tx-rx sets
    sources = np.zeros((nodes.shape[0], Ntx))  # current source intensity at all nodes (for all source configurations)
    for i in range(Ntx):
        # weights for the distribution of point current source to the neighboring nodes
        weights = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, tx[i][:, :3])
        # total current intensities at all the nodes
        sources[:, i] = weights.dot(tx[i][:, 3])

    # Obtain potentials at the nodes, potential differences and current along the edges
    start_time = time.time()
    potentials, potentialDiffs, currents = solveRESnet(edges, C, sources)
    end_time = time.time()
    print(f"Time: {(end_time - start_time):.6f} seconds")

    # Get simulated data
    data = []
    for i in range(Ntx):
        # weights for the interpolation of potential data at the M-electrode location
        Mw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, :3])
        # weights for the interpolation of potential data at the N-electrode location
        Nw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, 3:6])
        # calculate the potential difference data as "M - N"
        data.append((Mw.T - Nw.T) @ potentials[:, i])
    data1 = data  # save to data1

    '''Plot Model #1's apparent resistivity'''
    ABMNKVR1 = np.zeros((Ndata, 7))
    count = 0

    for i in range(Ntx):
        A = tx[i][0, 0]
        B = tx[i][1, 0]

        for j in range(rx[i].shape[0]):
            M = rx[i][j, 0]
            N = rx[i][j, 3]
            K = 2 * np.pi / (1 / (M - A) - 1 / (M - B) - 1 / (N - A) + 1 / (N - B))  # geometric factor
            count += 1
            # [A B M N geometric_factor potential_difference apparent_resistivity]
            ABMNKVR1[count - 1, :] = [A, B, M, N, K, data1[i][j], K * data1[i][j]]

    x = np.sum(ABMNKVR1[:, :4], axis=1) / 4  # mid-point of four electrodes
    n = (ABMNKVR1[:, 2] - ABMNKVR1[:, 1]) / spacing  # n-spacing as pseudo-depth

    plt.figure()
    plt.scatter(x, n, 500, np.log10(ABMNKVR1[:, 6]), marker='s', cmap='jet')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.colorbar(label='Apparent resistivity log(Ohm*m)')
    plt.xlabel('X (m)')
    plt.ylabel('n-spacing')
    plt.title('(1) Model #1: Half-space')

    '''Define Model #2: Two blocks in half-space'''
    blkLoc = np.array([[-np.inf, np.inf, -np.inf, np.inf, np.inf, 0],  # a uniform layer for the air (above surface)
                       [-np.inf, np.inf, -np.inf, np.inf, 0, -np.inf],  # a uniform half-space below surface
                       [-30, -20, -4, 4, -6, -10],  # an anomalous conductive block
                       [0, 8, -8, 2, -2, -6]])  # an anomalous resistive block

    blkCon = np.array([1e-5,  # air conductivity
                       0.01,  # earth conductivity
                       0.1,  # block conductivity
                       0.001])  # block conductivity

    '''Form a resistor network'''
    # Get connectivity properties of nodes, edges, faces, cells
    nodes, edges, lengths, faces, areas, cells, volumes = formRectMeshConnectivity(nodeX, nodeY, nodeZ)

    # Get conductive property model vectors (convert the block-model description to values on edges, faces, and cells)
    cellCon, faceCon, edgeCon = makeRectMeshModelBlocks(nodeX, nodeY, nodeZ, blkLoc, blkCon, None, None, None)

    # Convert all conductive objects to conductance on edges
    Edge2Edge = formEdge2EdgeMatrix(edges, lengths)
    Face2Edge = formFace2EdgeMatrix(edges, lengths, faces, areas)
    Cell2Edge = formCell2EdgeMatrix(edges, lengths, faces, cells, volumes)
    Ce = Edge2Edge.dot(edgeCon)  # conductance from edges
    Cf = Face2Edge.dot(faceCon)  # conductance from faces
    Cc = Cell2Edge.dot(cellCon)  # conductance from cells
    C = Ce + Cf + Cc  # total conductance

    '''Solve the resistor network problem'''
    # Calculate current sources on the nodes using info in tx
    Ntx = len(tx)  # number of tx-rx sets
    sources = np.zeros((nodes.shape[0], Ntx))  # current source intensity at all nodes (for all source configurations)
    for i in range(Ntx):
        # weights for the distribution of point current source to the neighboring nodes
        weights = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, tx[i][:, :3])
        # total current intensities at all the nodes
        sources[:, i] = weights.dot(tx[i][:, 3])

    # Obtain potentials at the nodes, potential differences and current along the edges
    start_time = time.time()
    potentials, potentialDiffs, currents = solveRESnet(edges, C, sources)
    end_time = time.time()
    print(f"Time: {(end_time - start_time):.6f} seconds")

    # Get simulated data
    data = []
    for i in range(Ntx):
        # weights for the interpolation of potential data at the M-electrode location
        Mw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, :3])
        # weights for the interpolation of potential data at the N-electrode location
        Nw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, 3:6])
        # calculate the potential difference data as "M - N"
        data.append((Mw.T - Nw.T) @ potentials[:, i])
    data2 = data  # save to data2

    '''Plot Model #2's apparent resistivity'''
    ABMNKVR2 = np.zeros((Ndata, 7))
    count = 0

    for i in range(Ntx):
        A = tx[i][0, 0]
        B = tx[i][1, 0]

        for j in range(rx[i].shape[0]):
            M = rx[i][j, 0]
            N = rx[i][j, 3]
            K = 2 * np.pi / (1 / (M - A) - 1 / (M - B) - 1 / (N - A) + 1 / (N - B))  # geometric factor
            count += 1
            # [A B M N geometric_factor potential_difference apparent_resistivity]
            ABMNKVR2[count - 1, :] = [A, B, M, N, K, data2[i][j], K * data2[i][j]]

    x = np.sum(ABMNKVR2[:, :4], axis=1) / 4  # mid-point of four electrodes
    n = (ABMNKVR2[:, 2] - ABMNKVR2[:, 1]) / spacing  # n-spacing as pseudo-depth

    plt.figure()
    plt.scatter(x, n, s=500, c=np.log10(ABMNKVR2[:, 6]), marker='s', cmap='viridis', alpha=1.0)
    plt.gca().invert_yaxis()
    plt.colorbar(label='Apparent resistivity log(Ohm*m)')
    plt.xlabel('X (m)')
    plt.ylabel('n-spacing')
    plt.title('(2) Model #2: Two blocks')
    plt.grid(True)

    '''Define Model #3: Infrastructure'''
    blkLoc = np.array([[-np.inf, np.inf, -np.inf, np.inf, np.inf, 0],  # a uniform layer for the air (above surface)
                       [-np.inf, np.inf, -np.inf, np.inf, 0, -np.inf],  # a uniform half-space below surface
                       [-30, -20, -4, 4, -6, -10],  # an anomalous conductive block
                       [0, 8, -8, 2, -2, -6],  # an anomalous resistive block
                       [-18, -18, -6, -6, 0, -100],  # a steel cased well
                       [6, 16, 4, 4, 6, -2],  # southern wall of steel-sheet warehouse
                       [6, 16, 10, 10, 6, -2],  # northern wall of steel-sheet warehouse
                       [6, 6, 4, 10, 6, -2],  # western wall of steel-sheet warehouse
                       [16, 16, 4, 10, 6, -2],  # eastern wall of steel-sheet warehouse
                       [6, 16, 4, 10, 6, 6]])  # roof

    blkCon = np.array([1e-5,  # air conductivity
                       0.01,  # earth conductivity
                       0.1,  # block conductivity
                       0.001,  # block conductivity
                       np.pi * (0.05 ** 2 - 0.04 ** 2) * 5e6,
                       # steel casing's edgeCon (cross-sectional area times intrinsic conductivity)
                       0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
                       0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
                       0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
                       0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
                       0.005 * 5e6])  # steel sheet's faceCon (thickness times intrinsic conductivity)

    '''Form a resistor network'''
    # Get connectivity properties of nodes, edges, faces, cells
    nodes, edges, lengths, faces, areas, cells, volumes = formRectMeshConnectivity(nodeX, nodeY, nodeZ)

    # Get conductive property model vectors (convert the block-model description to values on edges, faces and cells)
    cellCon, faceCon, edgeCon = makeRectMeshModelBlocks(nodeX, nodeY, nodeZ, blkLoc, blkCon, [], [], [])

    # Convert all conductive objects to conductance on edges
    Edge2Edge = formEdge2EdgeMatrix(edges, lengths)
    Face2Edge = formFace2EdgeMatrix(edges, lengths, faces, areas)
    Cell2Edge = formCell2EdgeMatrix(edges, lengths, faces, cells, volumes)
    Ce = Edge2Edge.dot(edgeCon)  # conductance from edges
    Cf = Face2Edge.dot(faceCon)  # conductance from faces
    Cc = Cell2Edge.dot(cellCon)  # conductance from cells
    C = Ce + Cf + Cc  # total conductance

    '''Solve the resistor network problem'''
    # Calculate current sources on the nodes using info in tx
    Ntx = len(tx)  # number of tx-rx sets
    sources = np.zeros((nodes.shape[0], Ntx))  # current source intensity at all nodes (for all source configurations)
    for i in range(Ntx):
        # weights for the distribution of point current source to the neighboring nodes
        weights = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, tx[i][:, :3])
        # total current intensities at all the nodes
        sources[:, i] = weights.dot(tx[i][:, 3])

    # Obtain potentials at the nodes, potential differences, and current along the edges
    start_time = time.time()
    potentials, potentialDiffs, currents = solveRESnet(edges, C, sources)
    end_time = time.time()
    print(f"Time: {(end_time - start_time):.6f} seconds")

    # Get simulated data
    data = []
    for i in range(Ntx):
        # weights for the interpolation of potential data at the M-electrode location
        Mw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, :3])
        # weights for the interpolation of potential data at the N-electrode location
        Nw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, 3:6])
        # calculate the potential difference data as "M - N"
        data.append((Mw.T - Nw.T) @ potentials[:, i])
    data3 = data  # save to data3

    '''Plot Model #3's apparent resistivity'''
    ABMNKVR3 = np.zeros((Ndata, 7))
    count = 0

    for i in range(Ntx):
        A = tx[i][0, 0]
        B = tx[i][1, 0]

        for j in range(rx[i].shape[0]):
            M = rx[i][j, 0]
            N = rx[i][j, 3]
            K = 2 * np.pi / (1 / (M - A) - 1 / (M - B) - 1 / (N - A) + 1 / (N - B))  # geometric factor
            count += 1
            # [A B M N geometric_factor potential_difference apparent_resistivity]
            ABMNKVR3[count - 1, :] = [A, B, M, N, K, data3[i][j], K * data3[i][j]]

    x = np.sum(ABMNKVR3[:, :4], axis=1) / 4  # mid-point of four electrodes
    n = (ABMNKVR3[:, 2] - ABMNKVR3[:, 1]) / spacing  # n-spacing as pseudo-depth

    plt.figure()
    plt.scatter(x, n, s=500, c=np.log10(ABMNKVR3[:, 6]), marker='s', cmap='viridis')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Apparent resistivity log(Ohm*m)')
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('n-spacing')
    plt.title('(3) Model #3: Infrastructure')

    '''Define Model #4: Above-ground pipe'''
    blkLoc = np.array([
        [-np.inf, np.inf, -np.inf, np.inf, np.inf, 0],  # a uniform layer for the air (above surface)
        [-np.inf, np.inf, -np.inf, np.inf, 0, -np.inf],  # a uniform half-space below surface
        [-30, -20, -4, 4, -6, -10],  # an anomalous conductive block
        [0, 8, -8, 2, -2, -6],  # an anomalous resistive block
        [-18, -18, -6, -6, 4, -100],  # a steel cased well
        [6, 16, 4, 4, 6, -2],  # southern wall of steel-sheet warehouse
        [6, 16, 10, 10, 6, -2],  # northern wall of steel-sheet warehouse
        [6, 6, 4, 10, 6, -2],  # western wall of steel-sheet warehouse
        [16, 16, 4, 10, 6, -2],  # eastern wall of steel-sheet warehouse
        [6, 16, 4, 10, 6, 6]  # roof
    ])

    blkCon = np.array([
        1e-5,  # air conductivity
        0.01,  # earth conductivity
        0.1,  # block conductivity
        0.001,  # block conductivity
        np.pi * (0.05 ** 2 - 0.04 ** 2) * 5e6,
        # steel casing's edgeCon (cross-sectional area times intrinsic conductivity)
        0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
        0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
        0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
        0.005 * 5e6,  # steel sheet's faceCon (thickness times intrinsic conductivity)
        0.005 * 5e6  # steel sheet's faceCon (thickness times intrinsic conductivity)
    ])

    '''Form a resistor network'''
    # Get connectivity properties of nodes, edges, faces, cells
    nodes, edges, lengths, faces, areas, cells, volumes = formRectMeshConnectivity(nodeX, nodeY, nodeZ)

    # Get conductive property model vectors (convert the block-model description to values on edges, faces, and cells)
    cellCon, faceCon, edgeCon = makeRectMeshModelBlocks(nodeX, nodeY, nodeZ, blkLoc, blkCon, [], [], [])

    # Convert all conductive objects to conductance on edges
    Edge2Edge = formEdge2EdgeMatrix(edges, lengths)
    Face2Edge = formFace2EdgeMatrix(edges, lengths, faces, areas)
    Cell2Edge = formCell2EdgeMatrix(edges, lengths, faces, cells, volumes)
    Ce = Edge2Edge.dot(edgeCon)  # conductance from edges
    Cf = Face2Edge.dot(faceCon)  # conductance from faces
    Cc = Cell2Edge.dot(cellCon)  # conductance from cells
    C = Ce + Cf + Cc  # total conductance

    # Add above-ground pipe
    pipeStart = np.array([-18, -6, 4])  # where the pipe starts
    pipeEnd = np.array([6, 4, 4])  # where the pipe ends
    pipeLength = np.linalg.norm(pipeStart - pipeEnd)  # length of the pipe
    pipeStartNode = np.where((nodes[:, 0] == pipeStart[0]) & (nodes[:, 1] == pipeStart[1]) &
                             (nodes[:, 2] == pipeStart[2]))[0]  # search for the starting node
    pipeEndNode = np.where((nodes[:, 0] == pipeEnd[0]) & (nodes[:, 1] == pipeEnd[1]) &
                           (nodes[:, 2] == pipeEnd[2]))[0]  # search for the ending node
    tmp = [pipeStartNode[0] + 1, pipeEndNode[0] + 1]
    edges = np.vstack((edges, tmp))  # append an additional edge representing the pipe
    # append an additional conductance representing the pipe
    C = np.append(C, np.pi * (0.05 ** 2 - 0.04 ** 2) * 5e6 / pipeLength)

    '''Solve the resistor network problem'''
    # Calculate current sources on the nodes using info in tx
    Ntx = len(tx)  # number of tx-rx sets
    sources = np.zeros((nodes.shape[0], Ntx))  # current source intensity at all nodes (for all source configurations)
    for i in range(Ntx):
        # weights for the distribution of point current source to the neighboring nodes
        weights = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, tx[i][:, :3])
        # total current intensities at all the nodes
        sources[:, i] = weights.dot(tx[i][:, 3])

    # Obtain potentials at the nodes, potential differences, and current along the edges
    start_time = time.time()
    potentials, potentialDiffs, currents = solveRESnet(edges, C, sources)
    end_time = time.time()
    print(f"Time: {(end_time - start_time):.6f} seconds")

    # Get simulated data
    data = []
    for i in range(Ntx):
        # weights for the interpolation of potential data at the M-electrode location
        Mw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, :3])
        # weights for the interpolation of potential data at the N-electrode location
        Nw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, 3:6])
        # calculate the potential difference data as "M - N"
        data.append((Mw.T - Nw.T) @ potentials[:, i])
    data4 = data  # save to data4

    '''Plot Model #4's apparent resistivity'''
    ABMNKVR4 = np.zeros((Ndata, 7))
    count = 0

    for i in range(Ntx):
        A = tx[i][0, 0]
        B = tx[i][1, 0]

        for j in range(rx[i].shape[0]):
            M = rx[i][j, 0]
            N = rx[i][j, 3]
            K = 2 * np.pi / (1 / (M - A) - 1 / (M - B) - 1 / (N - A) + 1 / (N - B))  # geometric factor
            count += 1
            # [A, B, M, N, geometric_factor, potential_difference, apparent_resistivity]
            ABMNKVR4[count - 1, :] = [A, B, M, N, K, data4[i][j], K * data4[i][j]]

    x = np.sum(ABMNKVR4[:, 0:4], axis=1) / 4  # mid-point of four electrodes
    n = (ABMNKVR4[:, 2] - ABMNKVR4[:, 1]) / spacing  # n-spacing as pseudo-depth

    plt.figure()
    plt.scatter(x, n, s=500, c=np.log10(ABMNKVR4[:, 6]), marker='s', cmap='viridis')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Apparent resistivity log(Ohm*m)')
    plt.xlabel('X (m)')
    plt.ylabel('n-spacing')
    plt.title('(4) Model #4: Above-ground pipe')
    plt.grid(True)
    plt.show()
