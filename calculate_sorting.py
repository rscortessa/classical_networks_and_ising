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

def update(id_coeff, hwt, nstates):
    aux = np.zeros_like(hwt)
    id_coeff_expanded = id_coeff[..., np.newaxis]
    
    vals = np.take_along_axis(hwt, id_coeff_expanded, axis=-1)
    np.put_along_axis(aux, id_coeff_expanded, -vals, axis=-1)
    
    return np.exp(Fast_Hadamard(aux) * nstates)

def inf(psi, phi):
    return 1 - (np.einsum('...m, ...m -> ...', psi, phi, optimize=True))**2
 
def step(exact, psi, past_inf, id_coeff, hwt, nstates):
    delta_psi = update(id_coeff, hwt, nstates)
    new_psi = psi * delta_psi
    new_psi = new_psi / np.linalg.norm(new_psi, axis=-1)[..., None]
    new_inf = inf(exact, new_psi)
    return new_psi, new_inf

def single_sweep(exact, psi, past_inf, sorting, hwt, nstates,candidates):
    batch_dims = psi.shape[:-1]

    infs = np.ones_like(sorting, dtype=float)
    new_psis = np.zeros(sorting.shape + (nstates,))
    past_inf_arr = np.asarray(past_inf).reshape(batch_dims)
    
    loop_idxs = np.min([sorting.shape[-1],candidates])
    for idx_i in range(loop_idxs):
        i = sorting[..., idx_i]
        new_psi, new_inf = step(exact, psi, past_inf, i, hwt, nstates)
        infs[..., idx_i] = new_inf
        new_psis[..., idx_i, :] = new_psi

    min_inf = np.argmin(np.abs(infs - past_inf_arr[..., np.newaxis]), axis=-1)
    min_idx_expanded = min_inf[..., np.newaxis]
    
    new_inf = np.take_along_axis(infs, min_idx_expanded, axis=-1).squeeze(-1)
    new_i = np.take_along_axis(sorting, min_idx_expanded, axis=-1).squeeze(-1)
    
    min_idx_psi = np.broadcast_to(min_inf[..., np.newaxis, np.newaxis], batch_dims + (1, nstates))
    new_psi = np.take_along_axis(new_psis, min_idx_psi, axis=-2).squeeze(-2)

    return new_psi, new_inf, new_i


def sweeps(exact, psi, sorting, hwt, nstates, hi, past_inf,candidates, parity=True):
    batch_dims = psi.shape[:-1]
    sorting_ = sorting.copy()
    infs = np.ones_like(sorting, dtype=float)
    order = np.ones_like(sorting)
    
    states_it = range(sorting.shape[-1])
    
    for kk in states_it:
        aux_psi, aux_inf, new_i = single_sweep(exact, psi, past_inf, sorting_, hwt, nstates,candidates)
        infs[..., kk] = aux_inf
        order[..., kk] = new_i
        
        psi = aux_psi
        past_inf = aux_inf
        
        mask = sorting_ != new_i[..., np.newaxis]
        sorting_ = sorting_[mask].reshape(*batch_dims, -1)
        
    return infs, order

# Initial parameters ....
parser = argparse.ArgumentParser()

parser.add_argument("--L", type=int, default=12)
parser.add_argument("--candidates", type=int, default=10)
parser.add_argument("--gi", type=float, default=1.5)
parser.add_argument("--gf", type=float, default=0.5)
parser.add_argument("--angle", type=float, default=0.0)
parser.add_argument("--t_start", type=float, default=0.0)
parser.add_argument("--t_idx", type=int, default=60)

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

log_WHT = Fast_Hadamard(np.log(psi_t_idx))
bare_sorting = np.argsort(np.abs(log_WHT.real))[2**(L-1):]
exact = np.abs(psi_t_idx)
psi = exact.copy()
past_inf = 0.0
inf_exact,order_exact = sweeps(exact, psi, bare_sorting, log_WHT.real,2**L,hi,past_inf,candidates, parity=True)


print(inf_exact.shape,order_exact.shape)
base_name_inf = f"inf_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}candidates{candidates}Nt{Ndt}.npy"
filename_inf = os.path.join(output_dir, base_name_inf)
psi_inf = np.lib.format.open_memmap(
        filename_inf, 
        mode='w+', 
        dtype=np.complex128, 
        shape=(2**(L-1),)
)
psi_inf[:] = inf_exact[::-1]

base_name_sort = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}candidates{candidates}Nt{Ndt}.npy"
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
psi_history.flush()
del psi_history
del psi_sort
del psi_inf
