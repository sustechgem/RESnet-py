import time

import numpy as np
from matplotlib import pyplot as plt

import calcTrilinearInterpWeights
import formCell2EdgeMatrix
import formEdge2EdgeMatrix
import formFace2EdgeMatrix
import formRectMeshConnectivity
import makeRectMeshModelBlocks
import solveRESnet

if __name__ == '__main__':
    """
    An example for testing the solution of half-space
    """

    '''Set up the geo-electrical model'''
    # Create a 3D rectilinear mesh
    h = 2
    ratio = 1.14
    nctbc = 30
    tmp = np.cumsum(h * np.power(ratio, np.arange(nctbc + 1)))
    nodeX = np.round(np.concatenate((-tmp[::-1], [0], tmp)))  # node locations in X
    nodeY = np.round(np.concatenate((-tmp[::-1], [0], tmp)))  # node locations in Y
    nodeZ = np.round(np.concatenate(([0], -tmp)))  # node locations in Z

    # Define the model using combination of blocks
    # A model includes some blocks that can represent objects like sheets or lines when one or two dimensions vanish.
    blkLoc = np.array([-np.inf, np.inf, -np.inf, np.inf, 0, -np.inf])  # a uniform half-space

    blkCon = np.array([1e-2])  # conductive property of the volumetric object (S/m)

    '''Setup the electric surveys'''
    # Define the current sources in the format of [x y z current(Ampere)]
    tx = np.array([[(0, 0, 0, 1),  # first set of source (two electrodes)
                    [-np.inf, 0, 0, -1]]])

    # Define the receiver electrodes in the format of [Mx My Mz Nx Ny Nz]
    rx = np.array([[[10, 0, 0, 20, 0, 0],  # first set of receivers (yielding three data values)
                    [20, 0, 0, 30, 0, 0],
                    [30, 0, 0, 40, 0, 0],
                    [40, 0, 0, 50, 0, 0],
                    [50, 0, 0, 60, 0, 0],
                    [60, 0, 0, 70, 0, 0],
                    [70, 0, 0, 80, 0, 0],
                    [80, 0, 0, 90, 0, 0],
                    [90, 0, 0, 100, 0, 0]]])

    '''Form a resistor network'''
    # Get connectivity properties of nodes, edges, faces, cells
    nodes, edges, lengths, faces, areas, cells, volumes = formRectMeshConnectivity.function(nodeX, nodeY, nodeZ)

    # Get conductive property model vectors (convert the block-model description to values on edges, faces and cells)
    cellCon, faceCon, edgeCon = makeRectMeshModelBlocks.function(nodeX, nodeY, nodeZ, blkLoc, blkCon, [], [], [])

    # Convert all conductive objects to conductance on edges
    Edge2Edge = formEdge2EdgeMatrix.function(edges, lengths)
    Face2Edge = formFace2EdgeMatrix.function(edges, lengths, faces, areas)
    Cell2Edge = formCell2EdgeMatrix.function(edges, lengths, faces, cells, volumes)
    Ce = Edge2Edge.dot(edgeCon)  # conductance from edges
    Cf = Face2Edge.dot(faceCon)  # conductance from faces
    Cc = Cell2Edge.dot(cellCon)  # conductance from cells
    C = Ce + Cf + Cc  # total conductance

    '''Solve the resistor network problem'''
    # Calculate current sources on the nodes using info in tx
    tx = np.array(tx)
    Ntx = len(tx)  # number of tx-rx sets
    sources = np.zeros((nodes.shape[0], Ntx))
    for i in range(Ntx):
        weights = calcTrilinearInterpWeights.function(nodeX, nodeY, nodeZ, tx[i][:, 0:3])
        # weights for the distribution of point current source to the neighboring nodes
        sources[:, i] = weights.dot(tx[i][:, 3])  # total current intensities at all the nodes

    # Obtain potentials at the nodes, potential differences and current along the edges
    start_time = time.time()
    potentials, potentialDiffs, currents = solveRESnet.function(edges, C, sources)
    end_time = time.time()
    print(f"Time: {(end_time - start_time):.6f} seconds")

    # Get simulated data
    potentials = potentials.reshape((-1, 1))
    data = []
    for i in range(Ntx):
        Mw = calcTrilinearInterpWeights.function(nodeX, nodeY, nodeZ, rx[i][:, :3])
        Nw = calcTrilinearInterpWeights.function(nodeX, nodeY, nodeZ, rx[i][:, 3:6])
        data.append((Mw.T - Nw.T) @ potentials[:, i])

    '''Analytic solutions'''
    rAM = rx[0][:, 0] - 0
    rAN = rx[0][:, 3] - 0
    rho = 100
    I = 1
    dV = rho * I / 2 / np.pi * (1 / rAM - 1 / rAN)
    X = 0.5 * (rx[0][:, 0] + rx[0][:, 3])

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].semilogy(X, data[0], '.-', label='RESnet-m')
    axs[0].plot(X, dV, 'o-', label='Analytic', markerfacecolor='none')
    axs[0].set_title('Uniform half-space under excitation of a point current source')
    axs[0].set_xlabel('Distance (m)')
    axs[0].set_ylabel('Potential difference (Volt)')
    axs[0].set_xlim(10, 100)
    axs[0].set_ylim(0.01, 1)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(X, ((data[0] - dV) / dV) * 100, 'k.-')
    axs[1].set_xlabel('Distance (m)')
    axs[1].set_ylabel('Relative Error (%)')
    axs[1].set_xlim(10, 100)
    axs[1].set_ylim(-3, 2)
    axs[1].grid(True)

    plt.show()
