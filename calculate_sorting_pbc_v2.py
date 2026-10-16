#!/usr/bin/env python
# coding: utf-8

import os
import argparse
# 1. Restrict threads BEFORE importing netket/jax/numpy
os.environ["OMP_NUM_THREADS"] = "2"
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

# sys.path.append("../netket_with_NQS")
# from class_WF import rotated_IsingModel


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


def sweeps(target_psi, psi, nstates, hi, past_inf, L, idx_groups, ngroups, continuous=False):
    batch_dims = target_psi.shape[:-1]

    # We only need the WHT once to precompute the exponent matrices
    wht_log_psi = Fast_Hadamard(np.log(psi))

    print("Precomputing exponent matrices for all groups...")
    exp_minus_E = np.zeros((ngroups, nstates), dtype=float)
    exp_minus_2E = np.zeros((ngroups, nstates), dtype=float)

    for i, idxs_group in enumerate(idx_groups):
        V = np.zeros(nstates, dtype=wht_log_psi.dtype)
        for g in idxs_group:
            V[g] = wht_log_psi[g]

        E = Fast_Hadamard(V) * nstates
        exp_minus_E[i] = np.exp(-E)
        exp_minus_2E[i] = np.exp(-2 * E)

    infs = np.ones(batch_dims + (ngroups,), dtype=float)
    order = np.ones(batch_dims + (ngroups,), dtype=int)

  
    active_mask = np.ones(ngroups, dtype=bool)
    global_idx_array_group = np.arange(0,ngroups,dtype=int)
    # Maintain the state directly in real space
    current_psi = psi.copy()

    # Sweep exactly ngroups times
    for kk in range(ngroups):
        # 1. A_real and C_real are calculated instantly in real space (NO transforms needed)
        A_real = target_psi * current_psi
        C_real = current_psi ** 2

        # 2. SIMULTANEOUS EVALUATION: Evaluate all groups instantly
        sum_a_0 = np.dot(exp_minus_E[active_mask], A_real)
        sum_c_0 = np.dot(exp_minus_2E[active_mask], C_real)

        overlap_sq = sum_a_0**2
        norm_sq = sum_c_0

        group_infs = 1.0 - (overlap_sq / norm_sq)

        # 3. Find the best candidate
        if not continuous:
            best_idx = np.argmin(group_infs, axis=-1)
        else:
            diff = np.abs(past_inf - group_infs)
            best_idx = np.argmin(diff, axis=-1)

        # 4. Apply the exact multi-state update via real-space multiplication!
        # This completely replaces the `deleting_group` function and the internal transforms.
        current_psi = current_psi * exp_minus_E[best_idx]
        best_inf = group_infs[best_idx]
        
        global_idx_group = global_idx_array_group[active_mask][best_idx]
        chosen_group = idx_groups[global_idx_group]

        active_mask[global_idx_group] = False

        
  
        infs[..., kk] = best_inf
        order[..., kk] = chosen_group[0]
        past_inf = best_inf

    return infs, order

def sweeps(target_psi, psi, nstates, hi, past_inf, L, idx_groups, ngroups, continuous=False):
    batch_dims = target_psi.shape[:-1]

    # Ensure initial wavefunctions are strictly normalized in real space
    target_psi = target_psi / np.sqrt(np.sum(target_psi ** 2))
    psi = psi / np.sqrt(np.sum(psi ** 2))

    # We only need the WHT once to precompute the exponent matrices
    wht_log_psi = Fast_Hadamard(np.log(psi))

    print("Precomputing exponent matrices for all groups...")
    exp_minus_E = np.zeros((ngroups, nstates), dtype=float)
    exp_minus_2E = np.zeros((ngroups, nstates), dtype=float)

    for i, idxs_group in enumerate(idx_groups):
        V = np.zeros(nstates, dtype=wht_log_psi.dtype)
        for g in idxs_group:
            # Optional safeguard: if g == 0 is in a group, its WHT value is just the global norm offset.
            # Keeping it or setting V[0] = 0 prevents exp(-E) from blowing up numerically.
            if g != 0:
                V[g] = wht_log_psi[g]

        E = Fast_Hadamard(V) * nstates
        # Subtract the mean of E to prevent exponential overflow/underflow; 
        # a constant shift in E cancels out in (overlap_sq / norm_sq)
        E_shifted = E - np.mean(E)
        exp_minus_E[i] = np.exp(-E_shifted)
        exp_minus_2E[i] = np.exp(-2 * E_shifted)

    infs = np.ones(batch_dims + (ngroups,), dtype=float)
    order = np.ones(batch_dims + (ngroups,), dtype=int)

    active_mask = np.ones(ngroups, dtype=bool)
    global_idx_array_group = np.arange(0, ngroups, dtype=int)
    
    # Maintain the state directly in real space
    current_psi = psi.copy()

    # Sweep exactly ngroups times
    for kk in range(ngroups):
        # 1. A_real and C_real are calculated instantly in real space
        A_real = target_psi * current_psi
        C_real = current_psi ** 2

        # 2. SIMULTANEOUS EVALUATION: Evaluate all active groups instantly
        sum_a_0 = np.dot(exp_minus_E[active_mask], A_real)
        sum_c_0 = np.dot(exp_minus_2E[active_mask], C_real)

        overlap_sq = sum_a_0 ** 2
        norm_sq = sum_c_0

        # Since target_psi is normalized, 1 - overlap_sq / norm_sq is exact
        group_infs = 1.0 - (overlap_sq / norm_sq)

        # 3. Find the best candidate among active groups
        if not continuous:
            best_idx = np.argmin(group_infs, axis=-1)
        else:
            diff = np.abs(past_inf - group_infs)
            best_idx = np.argmin(diff, axis=-1)

        # Map local best_idx to global group index FIRST
        global_idx_group = global_idx_array_group[active_mask][best_idx]
        chosen_group = idx_groups[global_idx_group]
        best_inf = group_infs[best_idx]

        # 4. Apply the exact multi-state update using the GLOBAL index
        current_psi = current_psi * exp_minus_E[global_idx_group]
        # Renormalize current_psi at each step to prevent floating-point drift
        current_psi /= np.sqrt(np.sum(current_psi ** 2))

        active_mask[global_idx_group] = False

        infs[..., kk] = best_inf
        order[..., kk] = chosen_group[0]
        past_inf = best_inf

    return infs, order

# Initial parameters ....
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx", type=int, default=10)

    args, _ = parser.parse_known_args()
    params = vars(args)

    eps = 1e-16
    dt = 0.01
    Ndt = 100

    t_idx = params["t_idx"]
    L = params["L"]
    gi = params["gi"]
    gf = params["gf"]
    angle = params["angle"]
    t_start = params["t_start"]
    pbc = True
    
    hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)
    add = ""
    if pbc:
        add += "_pbc"
    if angle != 0.0:
        add += "_" + str(angle)
        initial_mask = np.ones(2**L, dtype=int)
    else:
        initial_mask = np.prod(hi.all_states(),axis=-1) > 0

    len_mask = np.sum(initial_mask)

    output_dir = "wavefunctions_data" + add
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename = os.path.join(output_dir, base_name)
    
    try:
        psi_history = np.lib.format.open_memmap(
                filename, 
                mode='r+', 
                dtype=np.complex128, 
                shape=(Ndt+1, 2**L)
        )
        psi_t_idx = psi_history[t_idx]
        print(f"Loaded target wavefunction shape: {psi_t_idx.shape}")
    except FileNotFoundError:
        # Dummy data for testing if file is missing
        print(f"Warning: {filename} not found. Generating dummy target wavefunction.")
        psi_t_idx = np.ones(2**L, dtype=np.complex128) / np.sqrt(2**L)

    continuous_list = [False, True]



    # In your main script, replace the group_equivalent_states() call with:
    local_filename = os.path.join(output_dir,f"groups_local_L{L}_a{angle}.npy")
    global_filename = os.path.join(output_dir,f"groups_global_L{L}_a{angle}.npy")

    if not os.path.exists(global_filename):
        raise FileNotFoundError(f"Missing {global_filename}. Run generate_groups.py first.")

    # Load the dense 2D arrays
    dense_groups_local = np.load(local_filename)
    dense_groups_global = np.load(global_filename)

    
    for continuous in continuous_list:

        ngroups = len(dense_groups_global)
        
        # Since your algorithm uses `idx_groups.pop()`, convert the dense array back to a list of lists 
        # while filtering out the -1 padding:
        idx_groups = [g[g != -1].tolist() for g in dense_groups_global]

        exact = np.abs(psi_t_idx)
        
        past_inf = 0.0
        inf_exact, order_exact = sweeps(exact, exact, 2**L, hi, past_inf, L, idx_groups,ngroups,continuous=continuous)
       
        print(f"Continuous={continuous}: Infs shape {inf_exact.shape}, Order shape {order_exact.shape}")
        
        base_name_inf = f"inf_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
        filename_inf = os.path.join(output_dir, base_name_inf)
        
        # Save lengths matching the number of groups, not len_mask
        ngroups = inf_exact.shape[-1]
        psi_inf = np.lib.format.open_memmap(
            filename_inf, 
            mode='w+', 
            dtype=np.complex128, 
            shape=(ngroups,)
        )
        psi_inf[:] = inf_exact[::-1]

        base_name_sort = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
        filename_sort = os.path.join(output_dir, base_name_sort)
        psi_sort = np.lib.format.open_memmap(
            filename_sort, 
            mode='w+', 
            dtype=int, 
            shape=(ngroups,)
        )
        psi_sort[:] = order_exact[::-1]

        psi_sort.flush()
        psi_inf.flush()
        del psi_sort
        del psi_inf

    if 'psi_history' in locals():
        psi_history.flush()
        del psi_history
