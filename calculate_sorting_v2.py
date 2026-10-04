#!/usr/bin/env python
# coding: utf-8

# In[5]:



import os
import argparse
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
    L = a.shape[-1]
    h = 1
    
    while h < L:
        a2 = a.reshape(a.shape[:-1] + (L // (2 * h), 2, h))
        x = a2[..., 0, :]
        y = a2[..., 1, :]

        t = x.copy()
        a2[..., 0, :] = 0.5 * (t + y)
        a2[..., 1, :] = 0.5 * (t - y) * ordering

        h *= 2

    return a

    
def single_sweep(exact,psi,mask,past_inf,hwt,idx_arr,nstates,continuous):

    a = exact * psi
    Csquare_a = Fast_Hadamard(a)
    Csquare = Fast_Hadamard(exact**2)

    A_plus = (1+nstates*Csquare_a[mask])/2.0
    A_minus = (1-nstates*Csquare_a[mask])/2.0
    B_plus = (1+nstates*Csquare[mask])/2.0
    B_minus = (1-nstates*Csquare[mask])/2.0

    square_overlaps = A_plus*np.exp(-hwt[mask])+A_minus*np.exp(hwt[mask])
    norm_square = B_plus*np.exp(-2 * hwt[mask])+B_minus*np.exp(2 * hwt[mask])

    infs = 1 - square_overlaps**2/norm_square

    if not continuous:
        idx_inf = np.argmin((infs),axis=-1)
    else:
        idx_inf = np.argmin(np.abs(past_inf-infs),axis=-1)
        
    min_inf = infs[idx_inf]
    global_idxs_inf = idx_arr[mask][idx_inf]

    aux = np.zeros_like(hwt)
    id_coeff_expanded = global_idxs_inf[..., np.newaxis]
    vals = np.take_along_axis(hwt, id_coeff_expanded, axis=-1)
    np.put_along_axis(aux, id_coeff_expanded, -vals, axis=-1)
    np.put_along_axis(mask, id_coeff_expanded, False, axis=-1)
    
    new_psi = psi * np.exp(Fast_Hadamard(aux) * nstates)
    new_psi = new_psi / np.linalg.norm(new_psi,axis=-1)

    
    return new_psi,mask,min_inf,idx_inf


def sweeps(exact, hwt, initial_mask,nstates, hi, past_inf,continuous=False):
    batch_dims = exact.shape[:-1]
    infs = np.ones_like(hwt[initial_mask], dtype=float)
    order = np.ones_like(hwt[initial_mask])

    mask = initial_mask
    psi = exact
    
    indices = np.arange(nstates)
    leading_shape = hwt.shape
    idx_arr = np.broadcast_to(indices, leading_shape)
    
    for kk in range(int(nstates/2)):
        new_psi, new_mask, new_inf, idx_inf = single_sweep(exact,psi,mask,past_inf, hwt,idx_arr,nstates,continuous)
        infs[..., kk] = new_inf
        order[..., kk] = idx_inf
        
        psi = new_psi
        past_inf = new_inf
        mask = new_mask
        
    return infs, order

# Initial parameters ....
parser = argparse.ArgumentParser()

parser.add_argument("--L", type=int, default=12)
parser.add_argument("--candidates", type=int, default=10)
parser.add_argument("--gi", type=float, default=1.5)
parser.add_argument("--gf", type=float, default=0.5)
parser.add_argument("--angle", type=float, default=0.0)
parser.add_argument("--t_start", type=float, default=0.0)
parser.add_argument("--t_idx", type=int, default=10)

args_list=None
args, _ = parser.parse_known_args(args_list)
params = vars(args)



eps = 1e-16
dt = 0.01
Ndt = 100

t_idx = params["t_idx"]
L = params["L"]
gi = params["gi"]
gf = params["gf"]
angle= params["angle"]
t_start = params["t_start"]
candidates=params["candidates"]

hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)

output_dir = "wavefunctions_data"
os.makedirs(output_dir, exist_ok=True)

base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
filename = os.path.join(output_dir, base_name)
psi_history = np.lib.format.open_memmap(
        filename, 
        mode='r+', 
        dtype=np.complex128, 
        shape=(Ndt+1,2**L)
)
psi_t_idx = psi_history[t_idx]
print(psi_t_idx.shape)


continuous_list=[False,True]

for continuous in continuous_list:

    log_WHT = Fast_Hadamard(np.log(psi_t_idx))
    exact = np.abs(psi_t_idx)
    initial_mask = np.arange(2**L,dtype=int) % 2 == 0
    past_inf = 0.0

    inf_exact,order_exact = sweeps(exact,log_WHT.real,initial_mask,2**L,hi,past_inf,continuous)

    print(inf_exact.shape,order_exact.shape)
    base_name_inf = f"inf_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
    filename_inf = os.path.join(output_dir, base_name_inf)
    psi_inf = np.lib.format.open_memmap(
        filename_inf, 
        mode='w+', 
        dtype=np.complex128, 
        shape=(2**(L-1),)
    )
    psi_inf[:] = inf_exact[::-1]

    base_name_sort = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
    filename_sort = os.path.join(output_dir, base_name_sort)
    psi_sort = np.lib.format.open_memmap(
        filename_sort, 
        mode='w+', 
        dtype=int, 
        shape=(2**(L-1),)
    )
    psi_sort[:] = order_exact[::-1]


    psi_sort.flush()
    psi_inf.flush()
    del psi_sort
    del psi_inf

psi_history.flush()
del psi_history

