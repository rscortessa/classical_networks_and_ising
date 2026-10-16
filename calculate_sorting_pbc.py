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


import netket as nk
import numpy as np

def group_equivalent_states(L, hi, states, pad_to_dense=False):
    """
    Groups Hilbert space states of a 1D spin-1/2 chain based on 
    translation (cyclic) and reflection symmetries.
    
    Args:
        L (int): Length of the chain.
        hi: NetKet Hilbert space object.
        states: The specific batch/subset of states to group.
        pad_to_dense (bool): If True, returns 2D numpy arrays padded with -1. 
                             If False, returns lists of lists.
                             
    Returns:
        tuple: (groups_local, groups_global)
            - groups_local: 2D grouping of state indices relative to the input `states`.
            - groups_global: 2D grouping of state indices relative to the full Hilbert space.
    """
    # 1. Map global Hilbert indices to local array indices
    batch_global_indices = hi.states_to_numbers(states)
    
    # Fast lookup array: lookup[global_index] = local_index
    lookup = np.full(hi.n_states, -1, dtype=int)
    lookup[batch_global_indices] = np.arange(len(states))
    
    visited = np.zeros(len(states), dtype=bool)
    
    groups_local = []
    groups_global = []
    
    for idx, state in enumerate(states):
        if visited[idx]:
            continue
            
        # Generate all symmetric variations of the current state
        symm_states = np.empty((2 * L, L))
        for shift in range(L):
            rolled = np.roll(state, shift)
            symm_states[shift] = rolled
            symm_states[L + shift] = rolled[::-1] # Reflection
            
        # Get the unique integer indices of these states in the global Hilbert space
        symm_global_all = np.unique(hi.states_to_numbers(symm_states))
        
        # Map global indices back to local indices
        symm_local_all = lookup[symm_global_all]
        
        # Filter out symmetric states that aren't in our input batch (-1)
        valid_mask = symm_local_all != -1
        symm_local = symm_local_all[valid_mask]
        symm_global = symm_global_all[valid_mask]
        
        groups_local.append(symm_local.tolist())
        groups_global.append(symm_global.tolist())
        
        # Mark equivalent states as visited using local indices
        visited[symm_local] = True
        
    if pad_to_dense:
        max_len = max(len(g) for g in groups_local)
        
        dense_local = np.full((len(groups_local), max_len), -1, dtype=int)
        dense_global = np.full((len(groups_global), max_len), -1, dtype=int)
        
        for i in range(len(groups_local)):
            dense_local[i, :len(groups_local[i])] = groups_local[i]
            dense_global[i, :len(groups_global[i])] = groups_global[i]
            
        return dense_local, dense_global
        
    return groups_local, groups_global


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

def deleting_group(a,c_square_psi,wht_log_psi,idxs_group,mask,past_inf,idx_arr,nstates,continuous):


    all_idx = np.arange(nstates)
    
    for idx in idxs_group:
        sum_a = a[0] * nstates
        sum_b = c_square_psi[0] * nstates
    
        A_plus = (sum_a+nstates*a[idx])/2.0
        A_minus = (sum_a-nstates*a[idx])/2.0
        B_plus = (sum_b+nstates*c_square_psi[idx])/2.0
        B_minus = (sum_b-nstates*c_square_psi[idx])/2.0

        square_overlaps = A_plus*np.exp(-wht_log_psi[idx])+A_minus*np.exp(wht_log_psi[idx])
        norm_square = B_plus*np.exp(-2 * wht_log_psi[idx])+B_minus*np.exp(2 * wht_log_psi[idx])

        infs = 1 - square_overlaps**2/norm_square

        aux = np.zeros_like(wht_log_psi)
        vals = np.take_along_axis(wht_log_psi, idx, axis=-1)
    
        np.put_along_axis(wht_log_psi, idx,0.0, axis=-1)
        np.put_along_axis(mask, idx, False, axis=-1)
        
        ch = np.cosh(vals) ; ch2 = np.cosh(2* vals)
        sh = np.sinh(vals) ; sh2 = np.sinh(2* vals)

        perm = all_idx^global_idxs_inf

        a = ch * a -sh * a[perm]
        c_square_psi = ch2 * c_square_psi -sh2 * c_square_psi[perm]

        Z_sq = c_square_psi[0] * nstates
        c_square_psi /= Z_sq
        a /= np.sqrt(Z_sq)

    return infs,a,c_square_psi,wht_log_psi,mask

def group_sweep(a,c_square_psi,wht_log_psi,mask,idx_groups,past_inf,idx_arr,nstates,continuous):

    n_groups = len(idx_groups)

    groups_infs = []
    groups_a = []
    groups_c_square_psi = []
    groups_wht_log_psi = []
    groups_mask = []
    
    for idxs_group in idx_groups:
        idx_group_infs,idx_group_a,idx_group_c_square_psi,idx_group_wht_log_psi,idx_group_mask = deleting_group(a,c_square_psi,wht_log_psi,idxs_group,mask,past_inf,idx_arr,nstates,continuous)

        groups_infs.append(idx_group_infs)
        groups_a.append(idx_group_a)
        groups_c_square_psi.append(idx_group_c_square_psi) 
        groups_wht_log_psi.append(idx_group_wht_log_psi)
        groups_mask.append(idx_group_mask) 

    groups_infs = np.array(groups_infs)
        
    if not continuous:
        idx_inf = np.argmin((groups_infs),axis=-1)
    else:
        idx_inf = np.argmin(np.abs(past_inf-group_infs),axis=-1)

    mask = np.ones(n_groups,dtype=bool)
    mask[idx_inf] = False
    idx_groups = idx_groups[mask]
    
    return group_infs[idx_inf],groups_a[idx_inf],groups_c_square_psi[idx_inf],groups_wht_log_psi[idx_inf],groups_mask[idx_inf],idx_groups


def sweeps(target_psi,psi,initial_mask,nstates, hi, past_inf,L,continuous=False):

    states = hi.all_states()[mask]
    batch_dims = target_psi.shape[:-1]
    
    groups_local, groups_global = group_equivalent_states(L, hi, states, pad_to_dense=False)
    ngroups= len(groups_local)
    
    a = Fast_Hadamard(target_psi * psi)
    c_square_psi = Fast_Hadamard(psi**2)
    wht_log_psi = Fast_Hadamard(np.log(psi))
    
   
    infs = np.ones(batch_dims+(ngroups,), dtype=float)
    order = np.ones(batch_dims+(ngroups,), dtype=float)

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

t_idx = params["t_idx"]
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
    initial_mask  = np.ones(2**L,dtype=int)
else:
    initial_mask = np.arange(2**L,dtype=int) % 2 == 0

len_mask = np.sum(initial_mask)

    
output_dir = "wavefunctions_data"+add
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
    
    past_inf = 0.0
    inf_exact,order_exact = sweeps(exact,exact,initial_mask,2**L, hi, past_inf,continuous=continuous)
   
    print(inf_exact.shape,order_exact.shape)
    base_name_inf = f"inf_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
    filename_inf = os.path.join(output_dir, base_name_inf)
    psi_inf = np.lib.format.open_memmap(
        filename_inf, 
        mode='w+', 
        dtype=np.complex128, 
        shape=(len_mask,)
    )
    psi_inf[:] = inf_exact[::-1]

    base_name_sort = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
    filename_sort = os.path.join(output_dir, base_name_sort)
    psi_sort = np.lib.format.open_memmap(
        filename_sort, 
        mode='w+', 
        dtype=int, 
        shape=(len_mask,)
    )
    psi_sort[:] = order_exact[::-1]


    psi_sort.flush()
    psi_inf.flush()
    del psi_sort
    del psi_inf

psi_history.flush()
del psi_history

