#!/usr/bin/env python
# coding: utf-8

# In[5]:



import os

# 1. Restrict threads BEFORE importing netket/jax/numpy
os.environ["OMP_NUM_THREADS"] = "2" # Example: Restrict to 2 threads per node
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
# Tell JAX to use exactly 2 threads for its CPU operations
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

import numpy as np
import netket as nk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse.linalg import eigsh,expm_multiply
import sys
import os
sys.path.append("../netket_with_NQS")
from class_WF import rotated_IsingModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def Fast_Hadamard(a):
    ordering = 1.0    
    a = a.copy()
    
    # Target the last dimension's length instead of len(a)
    L = a.shape[-1]
    h = 1
    
    while h < L:
        # Reshape only the last axis into: (blocks, 2, h)
        # a.shape[:-1] keeps all leading dimensions intact
        a2 = a.reshape(a.shape[:-1] + (L // (2 * h), 2, h))

        # Slice along the newly created axis of size 2
        # Ellipsis (...) automatically applies to all leading dimensions
        x = a2[..., 0, :]
        y = a2[..., 1, :]

        # Butterfly
        t = x.copy()
        a2[..., 0, :] = 0.5 * (t + y)
        a2[..., 1, :] = 0.5 * (t - y) * ordering

        h *= 2

    return a


L = 10
gi = 1.5
gf = 0.5
angle=0.0

eps = 1e-16

t_start = 0.0
dt = 0.01
Ndt = 100



hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)
H_in = rotated_IsingModel(angle*np.pi/180,gi,L,hi)
H_in = H_in.to_sparse()
E,psi = eigsh(H_in, k=1, which='SA')
GS = np.log(psi[:,0]+1e-14+1j*1e-14)
psi_initial = psi[:,0]


if angle == 90.0:
    # 1. Summing booleans is orders of magnitude faster than np.prod on floats.
    # 2. An even number of negative spins guarantees the product is 1.
    available_states = (hi.all_states() < 0).sum(axis=-1) % 2 == 0
else:
    # Directly allocate a boolean array. This uses 8x less RAM and is instantaneous.
    available_states = np.ones(2**L, dtype=bool)


# Here,we quantify the entire evolution...


#We need to built psi_t:

time = t_start+np.arange(Ndt+1)*dt
output_dir = "wavefunctions_data"
os.makedirs(output_dir, exist_ok=True)


# 1. Get the Hamiltonian as a sparse matrix (do NOT use to_dense())
H_out = rotated_IsingModel(angle*np.pi/180, gf, L, hi).to_sparse()


psi_in  = psi_initial.copy()
psi_t  = psi_in.copy()
    
A = -1j * H_out

base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"

filename = os.path.join(output_dir, base_name)
psi_history = np.lib.format.open_memmap(
        filename, 
        mode='w+', 
        dtype=np.complex128, 
        shape=(Ndt+1,len(psi_t))
)
psi_history[:,:] = expm_multiply(A,psi_t,start=time[0],stop=time[-1],num=Ndt+1,endpoint=True)
           
            
psi_history.flush()
del psi_history

print(gf)





