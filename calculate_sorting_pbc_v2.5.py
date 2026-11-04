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
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

import numpy as np


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


def get_real_space_orbits(L, dense_groups_global):
    """
    Extracts the representative indices and multiplicities (orbit sizes) 
    of all symmetry-equivalent state groups so real-space inner products 
    only sum over unique orbits M (~ 2^L / 2L) instead of 2^L states.
    """
    valid_mask = dense_groups_global != -1
    orbit_weights = np.sum(valid_mask, axis=1).astype(np.float64)
    orbit_reps = dense_groups_global[:, 0].astype(np.int64)
    
    # Check if dense_groups_global covers the full Hilbert space (e.g. angle != 0)
    # or only the even parity sector (angle == 0).
    total_covered = int(np.sum(orbit_weights))
    nstates = 1 << L
    
    if total_covered == nstates:
        return orbit_reps, orbit_weights
    else:
        # If dense_groups_global only has Z2-even groups (half of 2^L),
        # real-space states have both even and odd spin configurations, 
        # or we can use bitwise cyclic/reflection orbits on integers [0, 2^L).
        return compute_full_bitwise_orbits(L)


def compute_full_bitwise_orbits(L):
    """
    Fast bitwise computation of cyclic + reflection orbits over all 2^L basis states.
    Returns (orbit_reps, orbit_weights) of length M << 2^L.
    """
    nstates = 1 << L
    states = np.arange(nstates, dtype=np.uint32)
    min_rep = states.copy()

    # Track minimum representative under L cyclic bit-rotations and reflections
    curr = states.copy()
    
    # Bit-reversal (spatial reflection) via lookup table on bytes
    rev_curr = np.zeros_like(curr)
    temp = curr.copy()
    for _ in range(L):
        rev_curr = (rev_curr << 1) | (temp & 1)
        temp >>= 1

    mask_L = (1 << L) - 1
    for _ in range(L):
        min_rep = np.minimum(min_rep, curr)
        min_rep = np.minimum(min_rep, rev_curr)
        # Cyclic shift by 1 bit
        curr = ((curr << 1) & mask_L) | (curr >> (L - 1))
        rev_curr = ((rev_curr << 1) & mask_L) | (rev_curr >> (L - 1))

    orbit_reps, orbit_weights = np.unique(min_rep, return_counts=True)
    return orbit_reps.astype(np.int64), orbit_weights.astype(np.float64)


def precompute_compressed_exponents(wht_log_psi, dense_groups_global, orbit_reps, orbit_weights, nstates):
    """
    Computes exp(-E) and exp(-2E) evaluated ONLY on the M unique real-space 
    orbit representatives, using fast bitwise XOR popcount instead of 
    calling Fast_Hadamard(V) ngroups times on 2^L arrays.
    """
    ngroups = len(dense_groups_global)
    n_orbits = len(orbit_reps)

    exp_minus_E = np.empty((ngroups, n_orbits), dtype=np.float64)
    exp_minus_2E = np.empty((ngroups, n_orbits), dtype=np.float64)

    # Pre-build a 16-bit popcount lookup table for fast vectorized parity evaluation
    pop_lut = np.array([bin(i).count('1') & 1 for i in range(65536)], dtype=np.int8)

    def walsh_sign(g_arr, r_arr):
        # Computes (-1)^popcount(g & r) for g_arr[:, None] and r_arr[None, :]
        anded = np.bitwise_and(g_arr[:, None], r_arr[None, :])
        parity = pop_lut[anded & 0xFFFF] ^ pop_lut[(anded >> 16) & 0xFFFF]
        return 1.0 - 2.0 * parity

    # Normalize weights for exact mean calculation
    norm_weights = orbit_weights / nstates

    for i in range(ngroups):
        row = dense_groups_global[i]
        g_valid = row[(row != -1) & (row != 0)]
        
        if len(g_valid) == 0:
            exp_minus_E[i, :] = 1.0
            exp_minus_2E[i, :] = 1.0
            continue

        # All elements in a symmetry group have identical WHT coefficients,
        # or we can weight each g by wht_log_psi[g] directly:
        coeffs = wht_log_psi[g_valid]
        signs = walsh_sign(g_valid, orbit_reps)  # shape: (group_size, n_orbits)
        
        # Exact E evaluated on the M orbit representatives
        E_orbits = np.dot(coeffs, signs)
        
        # Shift by exact Hilbert-space mean to prevent overflow/underflow
        E_mean = np.dot(E_orbits, norm_weights)
        E_shifted = E_orbits - E_mean

        exp_E = np.exp(-E_shifted)
        exp_minus_E[i] = exp_E
        exp_minus_2E[i] = exp_E * exp_E

    return exp_minus_E, exp_minus_2E


def sweeps_compressed(target_psi_orbits, psi_orbits, orbit_weights, exp_minus_E, exp_minus_2E, reps, past_inf, continuous=False):
    """
    Performs the greedy group erasure sweep entirely in the compressed 
    real-space orbit basis of size M (~ 2^L / 2L).
    """
    ngroups = len(reps)

    # Normalize wavefunctions using orbit multiplicities: sum(w * psi^2) = 1
    target_norm = np.sqrt(np.dot(orbit_weights, target_psi_orbits ** 2))
    psi_norm = np.sqrt(np.dot(orbit_weights, psi_orbits ** 2))
    
    target_psi_orbits = target_psi_orbits / target_norm
    current_psi = psi_orbits / psi_norm

    infs = np.ones(ngroups, dtype=np.float64)
    order = np.ones(ngroups, dtype=np.int64)
    active_mask = np.ones(ngroups, dtype=bool)

    for kk in range(ngroups):
        # 1. Compute weighted real-space vectors of length M (not 2^L!)
        A_weighted = orbit_weights * target_psi_orbits * current_psi
        C_weighted = orbit_weights * (current_psi ** 2)

        # 2. Full contiguous BLAS matrix-vector multiply (NO boolean slicing copies!)
        sum_a_0 = exp_minus_E @ A_weighted
        sum_c_0 = exp_minus_2E @ C_weighted

        group_infs = 1.0 - (sum_a_0 ** 2) / sum_c_0

        # Mask out inactive groups in-place on the small 1D vector
        if not continuous:
            group_infs[~active_mask] = np.inf
            best_idx = int(np.argmin(group_infs))
        else:
            diff = np.abs(past_inf - group_infs)
            diff[~active_mask] = np.inf
            best_idx = int(np.argmin(diff))

        best_inf = group_infs[best_idx]

        # 3. Update state on the M orbits and renormalize
        current_psi = current_psi * exp_minus_E[best_idx]
        current_psi /= np.sqrt(np.dot(orbit_weights, current_psi ** 2))

        active_mask[best_idx] = False

        infs[kk] = best_inf
        order[kk] = reps[best_idx]
        past_inf = best_inf

    return infs, order


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

    dt = 0.01
    Ndt = 100

    t_idx = params["t_idx"]
    L = params["L"]
    gi = params["gi"]
    gf = params["gf"]
    angle = params["angle"]
    t_start = params["t_start"]
    nstates = 1 << L
    pbc = True
    
    add = "_pbc" if pbc else ""
    if angle != 0.0:
        add += "_" + str(angle)

    output_dir = "wavefunctions_data" + add
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename = os.path.join(output_dir, base_name)
    
    try:
        psi_history = np.lib.format.open_memmap(
            filename, 
            mode='r', 
            dtype=np.complex128, 
            shape=(Ndt + 1, nstates)
        )
        psi_t_idx = psi_history[t_idx]
        print(f"Loaded target wavefunction shape: {psi_t_idx.shape}")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Generating dummy target wavefunction.")
        psi_t_idx = np.ones(nstates, dtype=np.complex128) / np.sqrt(nstates)

    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")
    if not os.path.exists(global_filename):
        raise FileNotFoundError(f"Missing {global_filename}. Run generate_groups.py first.")

    dense_groups_global = np.load(global_filename)
    ngroups = len(dense_groups_global)
    reps = dense_groups_global[:, 0].astype(np.int64)

    # 1. Normalize exact wavefunction and compute WHT ONCE
    exact = np.abs(psi_t_idx)
    exact /= np.sqrt(np.sum(exact ** 2))
    wht_log_psi = Fast_Hadamard(np.log(exact))

    # 2. Extract compressed real-space symmetry orbits (size M << 2^L)
    print(f"Compressing real-space Hilbert space (2^{L} = {nstates}) into symmetry orbits...")
    orbit_reps, orbit_weights = get_real_space_orbits(L, dense_groups_global)
    print(f"Compressed {nstates} states -> {len(orbit_reps)} unique orbits ({nstates / len(orbit_reps):.1f}x reduction).")

    # 3. Precompute compressed exponent matrices ONCE for both continuous modes
    print("Precomputing compressed exponent matrices...")
    exp_minus_E, exp_minus_2E = precompute_compressed_exponents(
        wht_log_psi, dense_groups_global, orbit_reps, orbit_weights, nstates
    )

    # Slice wavefunction onto the M orbit representatives
    exact_orbits = exact[orbit_reps]

    for continuous in [False, True]:
        past_inf = 0.0
        inf_exact, order_exact = sweeps_compressed(
            exact_orbits, exact_orbits, orbit_weights, 
            exp_minus_E, exp_minus_2E, reps, past_inf, continuous=continuous
        )
       
        print(f"Continuous={continuous}: Infs shape {inf_exact.shape}, Order shape {order_exact.shape}")
        
        base_name_inf = f"inf_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
        filename_inf = os.path.join(output_dir, base_name_inf)
        
        psi_inf = np.lib.format.open_memmap(
            filename_inf, mode='w+', dtype=np.complex128, shape=(ngroups,)
        )
        psi_inf[:] = inf_exact[::-1]

        base_name_sort = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
        filename_sort = os.path.join(output_dir, base_name_sort)
        psi_sort = np.lib.format.open_memmap(
            filename_sort, mode='w+', dtype=int, shape=(ngroups,)
        )
        psi_sort[:] = order_exact[::-1]

        psi_sort.flush()
        psi_inf.flush()
        del psi_sort
        del psi_inf

    if 'psi_history' in locals():
        del psi_history
