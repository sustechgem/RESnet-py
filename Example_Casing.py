import numpy as np
import time
import matplotlib.pyplot as plt

from calcTrilinearInterpWeights import calcTrilinearInterpWeights
from formCell2EdgeMatrix import formCell2EdgeMatrix
from formEdge2EdgeMatrix import formEdge2EdgeMatrix
from formFace2EdgeMatrix import formFace2EdgeMatrix
from formRectMeshConnectivity import formRectMeshConnectivity
from makeRectMeshModelBlocks import makeRectMeshModelBlocks
from solveRESnet import solveRESnet


def draw_figure(E, title):
    Ex = np.reshape(E[:N], (Ndatagridy, Ndatagridx))
    Ey = np.reshape(E[N:2 * N], (Ndatagridy, Ndatagridx))
    Etotal = np.sqrt(Ex ** 2 + Ey ** 2)
    Ex = np.rot90(Ex)
    Ey = np.rot90(Ey)
    Ey = np.flipud(Ey)
    Etotal = np.rot90(Etotal)
    plt.imshow(np.log10(Etotal), extent=(datagridx[0], datagridx[-1], datagridy[0], datagridy[-1]))
    plt.gca().invert_yaxis()
    hc = plt.colorbar()
    hc.set_label('primary electric field log(V/m)')
    plt.clim([-7, -4])
    plt.streamplot(datagridx, datagridy, Ex, Ey, density=1, color=Etotal, linewidth=1, cmap='jet')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)


if __name__ == '__main__':
    """
    Demonstration of steel casing modeling
    This script reproduces the top row panels (a, b, c, d) in Figure 5 in
    Heagy, L.J. and Oldenburg, D.W., 2019. Direct current resistivity with
    steel-cased wells. Geophysical Journal International, 219(1), pp.1-26.
    doi:10.1093/gji/ggz281  
    """

    '''Setup the geo-electrical model'''
    # Create a 3D rectilinear mesh
    h = 50  # smallest horizontal grid size
    ratio = 1.4  # expansion rate
    nctbc = 11  # number of cells to boundary condition
    nodeX = np.round(np.concatenate(
        (-np.cumsum((h * np.power(ratio, np.arange(nctbc, -1, -1)))[::-1])[::-1] - 1300,
         np.arange(-1300, 1300 + h, h),
         1300 + np.cumsum(h * np.power(ratio, np.arange(nctbc + 1)))),
        axis=0))  # node locations in X
    nodeY = np.round(np.concatenate(
        (-np.cumsum((h * np.power(ratio, np.arange(nctbc, -1, -1)))[::-1])[::-1] - 1300,
         np.arange(-1300, 1300 + h, h),
         1300 + np.cumsum(h * np.power(ratio, np.arange(nctbc + 1)))),
        axis=0))  # node locations in Y

    h = 50  # smallest vertical grid size
    ratio = 1.4  # expansion rate
    nctbc = 12  # number of cells to boundary condition
    nodeZ = np.concatenate((np.arange(0, -1001, -h), -1000 - np.round(np.cumsum(h * ratio ** (np.arange(nctbc + 1))))),
                           axis=0)  # node locations in Z

    # Define the model using combination of blocks
    # A model includes some blocks that can represent objects like sheets or lines when one or two dimensions vanish.
    blkLoc = np.array([[-np.inf, np.inf, -np.inf, np.inf, 0, -np.inf],  # a uniform half-space
                       [0, 0, 0, 0, 0, -1000]])  # a steel casing
    blkCon = np.array([[0.1],  # conductive property of the half-space earth (S/m)
                       [np.pi * (0.05 ** 2 - 0.04 ** 2) * 5e6]])
    # conductive property of the casing (S*m) = cross-sectional area (m^2) * steel's conductivity (S/m)

    '''Setup the electric surveys'''
    # Define the current sources in the format of [x y z current(Ampere)]
    tx = [
        np.array([[0, 0, 0, 1],  # first set of source (two electrodes)
                  [-2000, 0, 0, -1]]),  # return electrode 2000 m away
        np.array([[0, 0, 0, 1],  # second set of source (two electrodes)
                  [-750, 0, 0, -1]]),  # return electrode 750 m away
        np.array([[0, 0, 0, 1],  # third set of source (two electrodes)
                  [-500, 0, 0, -1]]),  # return electrode 500 m away
        np.array([[0, 0, 0, 1],  # fourth set of source (two electrodes)
                  [-250, 0, 0, -1]])  # return electrode 250 m away
    ]

    # Define the receiver electrodes in the format of [Mx My Mz Nx Ny Nz]
    spacing = 20  # M-N distance
    datagridx = np.arange(-1250, 1251, spacing)  # data grid in X
    Ndatagridx = len(datagridx)  # number of data grid in X
    datagridy = np.arange(-1250, 1251, spacing)  # data grid in Y
    Ndatagridy = len(datagridy)  # number of data grid in Y
    N = Ndatagridx * Ndatagridy  # total number of receivers
    datax, datay = np.meshgrid(datagridx, datagridy)
    rx = np.concatenate((
        np.vstack((datax.flatten('F') - spacing / 2, datay.flatten('F'), np.zeros(N), datax.flatten('F') + spacing / 2,
                   datay.flatten('F'), np.zeros(N))),  # measuring Ex
        np.vstack((datax.flatten('F'), datay.flatten('F') - spacing / 2, np.zeros(N), datax.flatten('F'),
                   datay.flatten('F') + spacing / 2, np.zeros(N)))  # measuring Ey
    ), axis=1)
    rx = rx.T
    rx = [rx, rx, rx, rx]  # same receiver locations for four different source configurations

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
    tx = np.array(tx)
    Ntx = len(tx)  # number of tx-rx sets
    sources = np.zeros((nodes.shape[0], Ntx))
    for i in range(Ntx):
        # weights for the distribution of point current source to the neighboring nodes
        weights = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, tx[i][:, 0:3])
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
        Mw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, 0:3])
        # weights for the interpolation of potential data at the N-electrode location
        Nw = calcTrilinearInterpWeights(nodeX, nodeY, nodeZ, rx[i][:, 3:6])
        # calculate the potential difference data as "M - N"
        data.append((Mw.T - Nw.T) @ potentials[:, i])
    data = np.array(data)

    '''Plot the results'''
    # The top row in Figure 5 (Heagy and Oldenburg, 2019)
    fig1 = plt.figure()
    E1 = data[0] / spacing  # first set of source
    draw_figure(E1, '(a) Return electrode offset = 2000 m')

    fig2 = plt.figure()
    E2 = data[1] / spacing  # second set of source
    draw_figure(E2, '(b) Return electrode offset = 750 m')

    fig3 = plt.figure()
    E3 = data[2] / spacing  # third set of source
    draw_figure(E3, '(c) Return electrode offset = 500 m')

    fig4 = plt.figure()
    E4 = data[3] / spacing  # forth set of source
    draw_figure(E4, '(d) Return electrode offset = 250 m')

    plt.show()
