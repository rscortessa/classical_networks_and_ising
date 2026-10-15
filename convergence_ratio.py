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

    
def single_sweep(a,c_square_psi,wht_log_psi,mask,past_inf,idx_arr,nstates,continuous):


    all_idx = np.arange(nstates)
    sum_a = a[0] * nstates
    sum_b = c_square_psi[0] * nstates
    
    A_plus = (sum_a+nstates*a[mask])/2.0
    A_minus = (sum_a-nstates*a[mask])/2.0
    B_plus = (sum_b+nstates*c_square_psi[mask])/2.0
    B_minus = (sum_b-nstates*c_square_psi[mask])/2.0

    square_overlaps = A_plus*np.exp(-wht_log_psi[mask])+A_minus*np.exp(wht_log_psi[mask])
    norm_square = B_plus*np.exp(-2 * wht_log_psi[mask])+B_minus*np.exp(2 * wht_log_psi[mask])

    infs = 1 - square_overlaps**2/norm_square

    if not continuous:
        idx_inf = np.argmin((infs),axis=-1)
    else:
        idx_inf = np.argmin(np.abs(past_inf-infs),axis=-1)
        
    min_inf = infs[idx_inf]
    global_idxs_inf = idx_arr[mask][idx_inf]

    
    
    id_coeff_expanded = global_idxs_inf[..., np.newaxis]


    aux = np.zeros_like(wht_log_psi)
    vals = np.take_along_axis(wht_log_psi, id_coeff_expanded, axis=-1)
    
    np.put_along_axis(wht_log_psi, id_coeff_expanded,0.0, axis=-1)
    np.put_along_axis(mask, id_coeff_expanded, False, axis=-1)

    ch = np.cosh(vals) ; ch2 = np.cosh(2* vals)
    sh = np.sinh(vals) ; sh2 = np.sinh(2* vals)

    perm = all_idx^global_idxs_inf

    a = ch * a -sh * a[perm]
    c_square_psi = ch2 * c_square_psi -sh2 * c_square_psi[perm]

    Z_sq = c_square_psi[0] * nstates
    c_square_psi /= Z_sq
    a /= np.sqrt(Z_sq)
    
    return a,c_square_psi,wht_log_psi,mask,min_inf,global_idxs_inf


def sweeps(target_psi,psi,initial_mask,nstates, hi, past_inf,continuous=False):

    a = Fast_Hadamard(target_psi * psi)
    c_square_psi = Fast_Hadamard(psi**2)
    wht_log_psi = Fast_Hadamard(np.log(psi))
    
    batch_dims = target_psi.shape[:-1]
    infs = np.ones_like(wht_log_psi[initial_mask], dtype=float)
    order = np.ones_like(wht_log_psi[initial_mask],dtype=int)

    mask = initial_mask.copy()
    
    indices = np.arange(nstates)
    leading_shape = wht_log_psi.shape
    idx_arr = np.broadcast_to(indices, leading_shape)


    
    for kk in range(int(nstates/2)):
        new_a,new_c_square_psi,updated_wht_log_psi,new_mask, new_inf, idx_inf = single_sweep(a,c_square_psi,wht_log_psi,mask,past_inf,idx_arr,nstates,continuous=continuous)

        infs[..., kk] = new_inf
        order[..., kk] = idx_inf
        
        a = new_a
        c_square_psi = new_c_square_psi
        wht_log_psi = updated_wht_log_psi
        past_inf = new_inf
        mask = new_mask
        
    return infs, order

# Initial parameters ....
parser = argparse.ArgumentParser()

parser.add_argument("--L", type=int, default=16)
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

L = params["L"]
gi = params["gi"]
gf = params["gf"]
angle= params["angle"]
t_start = params["t_start"]
candidates=params["candidates"]
pbc=True
hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)
add=""
if pbc:
    add+="_pbc"
if angle != 0.0:
    add+="_"+str(angle)
    

output_dir = "wavefunctions_data"+add
os.makedirs(output_dir, exist_ok=True)



base_name_ratio = f"ratio_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
filename = os.path.join(output_dir, base_name_ratio)
psi_ratio = np.lib.format.open_memmap(
        filename, 
        mode='r+', 
        dtype=np.complex128, 
        shape=(Ndt+1,2**(L)-1,2)
)


for t_idx in range(Ndt+1):
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


    continuous = True

    log_WHT = Fast_Hadamard(np.log(psi_t_idx))   
    log_sorting = np.sort(np.abs(log_WHT.real))[::-1]
    
    print(inf_exact.shape,order_exact.shape)

    base_name_sort = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
    filename_sort = os.path.join(output_dir, base_name_sort)
    psi_sort = np.lib.format.open_memmap(
        filename_sort, 
        mode='r+', 
        dtype=int, 
        shape=(2**(L),)
    )

    
    psi_ratio[t_idx:,0] = log_WHT[psi_sort[:-1]]/log_WHT[psi_sort[1:]]
    psi_ratio[t_idx:,1] = log_sorting[:-1]/log_sorting[1:]
    
        

    psi_sort.flush()
    del psi_sort

psi_ratio.flush()
psi_history.flush()

del psi_history
del psi_ratio
