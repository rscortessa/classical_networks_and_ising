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


def Fast_Hadamard(a, in_place=False):
    """
    In-place, zero-allocation Walsh-Hadamard Transform:
    f_hat(A) = (1 / 2^N) * sum_s f(s) X_A(s)
    Inverse is: f(s) = 2^N * Fast_Hadamard(f_hat)
    """
    if not in_place:
        a = a.copy()
    L = a.shape[-1]
    h = 1

    while h < L:
        a2 = a.reshape(a.shape[:-1] + (L // (2 * h), 2, h))
        x = a2[..., 0, :]
        y = a2[..., 1, :]

        # Zero-allocation in-place butterfly update (eliminates `t = x.copy()`)
        x += y
        y *= 2.0
        np.subtract(x, y, out=y)
        x *= 0.5
        y *= 0.5

        h *= 2

    return a


def precompute_KL_data(target_psi, psi, nstates, idx_groups, ngroups, batch_size=64, dtype=np.float64):
    """
    Precomputes `exp_minus_2E` and `group_kl_non_vac` ONCE for all sweeps.
    Uses sparse index gathering for `group_kl_non_vac` and batched 2D WHTs for `E`.
    """
    eps = 1e-12

    # Ensure strictly positive, consistent support for both log(psi) and psi**2
    target_psi = np.maximum(np.abs(target_psi), eps)
    target_psi /= np.sqrt(np.sum(target_psi ** 2))

    psi = np.maximum(np.abs(psi), eps)
    psi /= np.sqrt(np.sum(psi ** 2))

    P_target = target_psi ** 2
    wht_P_target = Fast_Hadamard(P_target, in_place=False)

    log_psi = np.log(psi)
    wht_log_psi = Fast_Hadamard(log_psi, in_place=False)

    print(f"Precomputing WHT KL contributions and exponent matrices for {ngroups} groups (batched)...")
    exp_minus_2E = np.empty((ngroups, nstates), dtype=dtype)
    group_kl_non_vac = np.zeros(ngroups, dtype=np.float64)

    # Process groups in batches of `batch_size` to vectorize Fast_Hadamard and fit in CPU cache
    V_batch = np.zeros((batch_size, nstates), dtype=np.float64)

    for start_i in range(0, ngroups, batch_size):
        end_i = min(start_i + batch_size, ngroups)
        cur_batch_len = end_i - start_i

        V_view = V_batch[:cur_batch_len]
        V_view.fill(0.0)

        for local_i, i in enumerate(range(start_i, end_i)):
            # Filter out vacuum state g == 0
            g_idx = np.asarray([g for g in idx_groups[i] if g != 0], dtype=np.intp)
            if g_idx.size > 0:
                v_vals = wht_log_psi[g_idx]
                V_view[local_i, g_idx] = v_vals
                # O(|g|) sparse dot product instead of O(2^L) full vector product!
                group_kl_non_vac[i] = (2.0 * nstates) * np.sum(wht_P_target[g_idx] * v_vals)

        # Single vectorized WHT call over the entire batch of groups
        E_batch = Fast_Hadamard(V_view, in_place=True)
        E_batch *= nstates

        # Subtract mean per group for numerical stability
        E_means = np.mean(E_batch, axis=-1, keepdims=True)
        E_batch -= E_means

        # Compute exp(-2 * E_shifted) in-place into E_batch and store in exp_minus_2E
        E_batch *= -2.0
        np.exp(E_batch, out=E_batch)
        exp_minus_2E[start_i:end_i] = E_batch

        group_kl_non_vac[start_i:end_i] -= 2.0 * E_means[:, 0]

    init_prob = (psi ** 2).copy()
    init_kl = float(np.sum(P_target * (np.log(P_target) - np.log(init_prob))))

    return exp_minus_2E, group_kl_non_vac, init_prob, init_kl


def sweeps_KL_fast(exp_minus_2E, group_kl_non_vac, init_prob, init_kl, past_kl, idx_groups, ngroups, continuous=False):
    """
    Executes the greedy group-deletion sweep without copying `exp_minus_2E`.
    Uses a single contiguous BLAS GEMV call (`exp_minus_2E @ current_prob`) per step.
    """
    kls = np.zeros(ngroups, dtype=np.float64)
    order = np.zeros(ngroups, dtype=int)

    # Penalty mask: 0.0 for active groups, +inf for deleted groups
    inactive_penalty = np.zeros(ngroups, dtype=np.float64)
    current_prob = init_prob.copy()
    accum_total_kl = init_kl

    for kk in range(ngroups):
        # 1. Contiguous BLAS matrix-vector product (NO boolean slicing / memory copying!)
        norm_sq = exp_minus_2E @ current_prob
        delta_vacuum_kl = np.log(norm_sq)

        # 2. Candidate KL for all groups
        raw_group_kls = accum_total_kl + group_kl_non_vac + delta_vacuum_kl

        # 3. Select the best active group using the additive penalty mask
        if not continuous:
            best_global_idx = int(np.argmin(raw_group_kls + inactive_penalty))
        else:
            diff = np.abs(past_kl - raw_group_kls) + inactive_penalty
            best_global_idx = int(np.argmin(diff))

        chosen_group = idx_groups[best_global_idx]
        exact_best_kl = float(raw_group_kls[best_global_idx])
        accum_total_kl = exact_best_kl

        # 4. Update real-space probability and renormalize
        current_prob *= exp_minus_2E[best_global_idx]
        current_prob /= np.sum(current_prob)

        # Mark group as inactive for future iterations
        inactive_penalty[best_global_idx] = np.inf

        clean_best_kl = max(0.0, exact_best_kl)
        kls[kk] = clean_best_kl
        order[kk] = chosen_group[0]
        past_kl = clean_best_kl

    return kls, order


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for WHT precomputation")

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
    batch_size = params["batch_size"]
    pbc = True
    nstates = 1 << L  # 2**L

    add = ""
    if pbc:
        add += "_pbc"
    if angle != 0.0:
        add += "_" + str(angle)

    output_dir = "wavefunctions_data" + add
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename = os.path.join(output_dir, base_name)

    try:
        # Use mode='r' (read-only) so we only page in the single time slice `t_idx`
        psi_history = np.lib.format.open_memmap(
            filename,
            mode='r',
            dtype=np.complex128,
            shape=(Ndt + 1, nstates)
        )
        psi_t_idx = np.array(psi_history[t_idx], copy=True)
        del psi_history  # Close memmap immediately to free virtual address space
        print(f"Loaded target wavefunction shape: {psi_t_idx.shape}")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Generating dummy target wavefunction.")
        psi_t_idx = np.ones(nstates, dtype=np.complex128) / np.sqrt(nstates)

    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")
    if not os.path.exists(global_filename):
        raise FileNotFoundError(f"Missing {global_filename}. Run generate_groups.py first.")

    dense_groups_global = np.load(global_filename, mmap_mode='r')
    ngroups = len(dense_groups_global)
    idx_groups = [g[g != -1].tolist() for g in dense_groups_global]
    del dense_groups_global

    exact = np.abs(psi_t_idx)
    del psi_t_idx

    # Precompute WHT and exponent matrices ONCE outside the `continuous` loop
    exp_minus_2E, group_kl_non_vac, init_prob, init_kl = precompute_KL_data(
        exact, exact, nstates, idx_groups, ngroups, batch_size=batch_size
    )

    continuous_list = [False, True]
    for continuous in continuous_list:
        past_kl = 0.0
        kl_exact, order_exact = sweeps_KL_fast(
            exp_minus_2E, group_kl_non_vac, init_prob, init_kl,
            past_kl, idx_groups, ngroups, continuous=continuous
        )

        print(f"Continuous={continuous}: KLs shape {kl_exact.shape}, Order shape {order_exact.shape}")

        base_name_kl = f"kl_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
        filename_kl = os.path.join(output_dir, base_name_kl)

        psi_kl = np.lib.format.open_memmap(
            filename_kl,
            mode='w+',
            dtype=np.complex128,
            shape=(ngroups,)
        )
        psi_kl[:] = kl_exact[::-1]

        base_name_sort = f"sort_KL_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
        filename_sort = os.path.join(output_dir, base_name_sort)
        psi_sort = np.lib.format.open_memmap(
            filename_sort,
            mode='w+',
            dtype=int,
            shape=(ngroups,)
        )
        psi_sort[:] = order_exact[::-1]
        print(order_exact)

        psi_sort.flush()
        psi_kl.flush()
        del psi_sort
        del psi_kl
