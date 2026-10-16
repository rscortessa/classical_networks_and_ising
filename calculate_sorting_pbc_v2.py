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


def group_equivalent_states(L, hi, states, pad_to_dense=False):
    """
    Groups Hilbert space states of a 1D spin-1/2 chain based on 
    translation (cyclic) and reflection symmetries.
    """
    batch_global_indices = hi.states_to_numbers(states)
    
    lookup = np.full(hi.n_states, -1, dtype=int)
    lookup[batch_global_indices] = np.arange(len(states))
    
    visited = np.zeros(len(states), dtype=bool)
    
    groups_local = []
    groups_global = []
    
    for idx, state in enumerate(states):
        if visited[idx]:
            continue
            
        symm_states = np.empty((2 * L, L))
        for shift in range(L):
            rolled = np.roll(state, shift)
            symm_states[shift] = rolled
            symm_states[L + shift] = rolled[::-1] 
            
        symm_global_all = np.unique(hi.states_to_numbers(symm_states))
        symm_local_all = lookup[symm_global_all]
        
        valid_mask = symm_local_all != -1
        symm_local = symm_local_all[valid_mask]
        symm_global = symm_global_all[valid_mask]
        
        groups_local.append(symm_local.tolist())
        groups_global.append(symm_global.tolist())
        
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
import numpy as np

def evaluate_candidate(A_real, C_real, wht_log_psi, idxs_group, nstates):
    """
    Evaluates the exact index 0 infidelity for a candidate group 
    using real-space diagonal exponential scaling.
    """
    # 1. Build the sparse vector of WHT coefficients to delete
    V = np.zeros(nstates, dtype=wht_log_psi.dtype)
    for g in idxs_group:
        V[g] = wht_log_psi[g]
        
    # 2. Transform to the real-space exponent.
    # Fast_Hadamard inherently applies a (1/nstates) scaling, 
    # so we multiply by nstates to get the exact unscaled exponent.
    E = Fast_Hadamard(V) * nstates
    
    # 3. Evaluate the exact unnormalized multi-state components for index 0
    sum_a_0 = np.sum(A_real * np.exp(-E))
    sum_c_0 = np.sum(C_real * np.exp(-2 * E))
    
    # The intermediate normalizations cancel out mathematically in this ratio
    overlap_sq = (sum_a_0 * nstates)**2
    norm_sq = sum_c_0 * nstates
    
    return 1.0 - (overlap_sq / norm_sq)


def deleting_group(a, c_square_psi, wht_log_psi, idxs_group, mask, nstates):
    # Create local copies to prevent overwriting during the sweep over candidates
    a = a.copy()
    c_square_psi = c_square_psi.copy()
    wht_log_psi = wht_log_psi.copy()
    mask = mask.copy()
    
    all_idx = np.arange(nstates)
    
    for idx in idxs_group:
        # Wrap index in an array to use with np.take_along_axis/put_along_axis
        idx_arr = np.array([idx])
        
        vals = np.take_along_axis(wht_log_psi, idx_arr, axis=-1)
    
        np.put_along_axis(wht_log_psi, idx_arr, 0.0, axis=-1)
        np.put_along_axis(mask, idx_arr, False, axis=-1)
        
        ch = np.cosh(vals) ; ch2 = np.cosh(2* vals)
        sh = np.sinh(vals) ; sh2 = np.sinh(2* vals)

        perm = all_idx ^ idx

        a = ch * a - sh * a[perm]
        c_square_psi = ch2 * c_square_psi - sh2 * c_square_psi[perm]

        Z_sq = c_square_psi[..., 0] * nstates
        c_square_psi /= Z_sq
        a /= np.sqrt(Z_sq)

    # Evaluate exact multi-state infidelity at the end of the group deletion
    overlap_sq = (a[..., 0] * nstates)**2
    norm_sq = c_square_psi[..., 0] * nstates
    infs = 1.0 - (overlap_sq / norm_sq)

    return infs, a, c_square_psi, wht_log_psi, mask





def group_sweep(a, c_square_psi, wht_log_psi, mask, idx_groups, past_inf, nstates, continuous):
    # Recover the real-space states ONCE at the start of the sweep
    A_real = Fast_Hadamard(a) * nstates
    C_real = Fast_Hadamard(c_square_psi) * nstates
    
    best_score = np.inf
    best_inf = np.inf
    best_idx_inf = -1
    
    # Evaluate all candidates purely by calculating index 0 in real space
    for i, idxs_group in enumerate(idx_groups):
        inf = evaluate_candidate(A_real, C_real, wht_log_psi, idxs_group, nstates)
        
        score = np.abs(past_inf - inf) if continuous else inf
        
        if score < best_score:
            best_score = score
            best_inf = inf
            best_idx_inf = i
            
    # Extract the winning group
    chosen_group = idx_groups.pop(best_idx_inf)
    
    # Apply the full O(2^L) sequential update only ONCE for the chosen winner
    # using your original deleting_group logic
    _, best_a, best_c, best_wht, best_mask = deleting_group(
        a, c_square_psi, wht_log_psi, chosen_group, mask, nstates
    )
    
    return best_inf, best_a, best_c, best_wht, best_mask, idx_groups, chosen_group


def sweeps(target_psi, psi, initial_mask, nstates, hi, past_inf, L, idx_groups, ngroups, continuous=False):
    batch_dims = target_psi.shape[:-1]

    a = Fast_Hadamard(target_psi * psi)
    c_square_psi = Fast_Hadamard(psi**2)
    wht_log_psi = Fast_Hadamard(np.log(psi))
    
    # =========================================================================
    # THE UPGRADE: Precompute the exponent matrices for ALL groups once.
    # This takes ~0.5 seconds and replaces millions of loops with BLAS matrix math.
    # =========================================================================
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

    mask = initial_mask.copy()
    
    # Use a boolean mask instead of .pop() so our indices align with the precomputed matrices
    active_mask = np.ones(ngroups, dtype=bool)
    
    # Sweep exactly ngroups times
    for kk in range(ngroups):
        # 1. Recover current real-space arrays once per sweep step
        A_real = Fast_Hadamard(a) * nstates
        C_real = Fast_Hadamard(c_square_psi) * nstates
        
        # 2. SIMULTANEOUS EVALUATION: Matrix-vector dot product evaluates all 686 groups instantly
        sum_a_0 = np.dot(exp_minus_E, A_real)
        sum_c_0 = np.dot(exp_minus_2E, C_real)
        
        overlap_sq = (sum_a_0 * nstates)**2
        norm_sq = sum_c_0 * nstates
        
        group_infs = 1.0 - (overlap_sq / norm_sq)
        
        # Mask out groups that have already been deleted
        group_infs = np.where(active_mask, group_infs, np.inf)
        
        # 3. Find the best candidate
        if not continuous:
            best_idx = np.argmin(group_infs, axis=-1)
        else:
            diff = np.abs(past_inf - group_infs)
            diff = np.where(active_mask, diff, np.inf)
            best_idx = np.argmin(diff, axis=-1)
            
        best_inf = group_infs[best_idx]
        chosen_group = idx_groups[best_idx]
        
        # Mark this group as deleted for future sweeps
        active_mask[best_idx] = False
        
        # 4. Apply the exact full O(2^L) update only for the winning group
        _, a, c_square_psi, wht_log_psi, mask = deleting_group(
            a, c_square_psi, wht_log_psi, chosen_group, mask, nstates
        )
        
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
        initial_mask = np.arange(2**L, dtype=int) % 2 == 0

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
        inf_exact, order_exact = sweeps(exact, exact, initial_mask, 2**L, hi, past_inf, L, idx_groups,ngroups,continuous=continuous)
       
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
