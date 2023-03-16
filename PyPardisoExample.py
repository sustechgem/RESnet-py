
import numpy as np
import scipy.sparse as sp
from PyPardiso import PyPardiso
import time
if __name__ == '__main__':
    t1 = time.time()
    a=500
    A=sp.rand(a, a, density=0.5, format='csr')*1j+sp.rand(a, a, density=0.5, format='csr')
    b=np.random.rand(a)+np.random.rand(a)*1j
    pardiso_solver=PyPardiso(A , matrix_type = 13) # Analysis, numerical factorization
    x=pardiso_solver.solve(b)# Solve, iterative refinement
    pardiso_solver.release()# Release memory
    t2=time.time()
    print('solver took %1.3f s ' %(t2 - t1))

