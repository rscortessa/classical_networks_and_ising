#!/usr/bin/env python
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
import netket as nk


def Fast_Hadamard(a):
    """
    Computes the Walsh-Hadamard Transform:
    f_hat(A) = (1 / 2^N) * sum_s f(s) X_A(s)
    Inverse is: f(s) = 2^N * Fast_Hadamard(f_hat)
    """
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


def sweeps_KL(target_psi, psi, nstates, hi, past_kl, L, idx_groups, ngroups, continuous=False):
    batch_dims = target_psi.shape[:-1]

    # Ensure initial wavefunctions are strictly positive and normalized in real space
    target_psi = np.abs(target_psi)
    target_psi = target_psi / np.sqrt(np.sum(target_psi ** 2))

    psi = np.abs(psi)
    psi = psi / np.sqrt(np.sum(psi ** 2))

    # Target probability P(s) = |target_psi(s)|^2 and its WHT: P_hat(A) = <X_A> / 2^N
    P_target = target_psi ** 2
    wht_P_target = Fast_Hadamard(P_target)

    # WHT of log(|psi(s)|): C_A^R = (1 / 2^N) * sum_s X_A(s) log|psi(s)|
    eps = 1e-16
    wht_log_psi = Fast_Hadamard(np.log(np.maximum(psi, eps)))

    print("Precomputing WHT KL contributions and exponent matrices for all groups...")
    exp_minus_2E = np.zeros((ngroups, nstates), dtype=float)
    group_kl_linear = np.zeros(ngroups, dtype=float)
    group_E_mean = np.zeros(ngroups, dtype=float)

    for i, idxs_group in enumerate(idx_groups):
        V = np.zeros(nstates, dtype=wht_log_psi.dtype)
        for g in idxs_group:
            # Exclude vacuum A = 0 from non-vacuum terms; A = 0 is handled by normalization
            if g != 0:
                V[g] = wht_log_psi[g]

        # Pure linear (non-vacuum) KL contribution for deleting group i:
        # sum_{A in group_i} 2^(2N+1) * P_hat(A) * C_A^psi
        group_kl_linear[i] = (2.0 * nstates) * np.sum(wht_P_target * V)

        # Real-space exponent E_i(s) = 2^N * sum_{A in group_i} C_A^psi * X_A(s)
        E = Fast_Hadamard(V) * nstates

        # Subtract mean(E) for numerical stability before exponentiating
        E_mean = float(np.mean(E))
        group_E_mean[i] = E_mean
        E_shifted = E - E_mean
        exp_minus_2E[i] = np.exp(-2.0 * E_shifted)

    kls = np.zeros(batch_dims + (ngroups,), dtype=float)
    kls_linear = np.zeros(batch_dims + (ngroups,), dtype=float)
    kls_vacuum = np.zeros(batch_dims + (ngroups,), dtype=float)
    order = np.zeros(batch_dims + (ngroups,), dtype=int)

    active_mask = np.ones(ngroups, dtype=bool)
    global_idx_array_group = np.arange(0, ngroups, dtype=int)

    # Maintain current normalized probability distribution |phi(s)|^2 in real space
    current_prob = (psi ** 2).copy()

    # If target_psi == psi initially, initial KL is 0.
    # Otherwise, compute exact initial KL offset:
    accum_total_kl = float(np.sum(P_target * np.log(np.maximum(P_target, eps) / np.maximum(current_prob, eps))))
    accum_linear_kl = 0.0
    accum_vacuum_kl = accum_total_kl

    # Sweep exactly ngroups times
    for kk in range(ngroups):
        # 1. Incremental vacuum (non-linear) contribution:
        # log( sum_s P_current(s) * exp(-2 E_g(s)) ) = log(norm_sq) - 2 * E_mean
        norm_sq = np.dot(exp_minus_2E[active_mask], current_prob)
        delta_vacuum_kl = np.log(norm_sq) - 2.0 * group_E_mean[active_mask]

        # 2. Exact candidate KL for each active group (linear + vacuum)
        delta_linear_kl = group_kl_linear[active_mask]
        raw_group_kls = accum_total_kl + delta_linear_kl + delta_vacuum_kl

        # 3. Find the best candidate among active groups
        if not continuous:
            best_idx = np.argmin(raw_group_kls, axis=-1)
        else:
            diff = np.abs(past_kl - raw_group_kls)
            best_idx = np.argmin(diff, axis=-1)

        # Map local best_idx to global group index
        global_idx_group = global_idx_array_group[active_mask][best_idx]
        chosen_group = idx_groups[global_idx_group]

        # Update running accumulations for total, linear, and vacuum KL
        exact_best_kl = float(raw_group_kls[best_idx])
        accum_total_kl = exact_best_kl
        accum_linear_kl += float(delta_linear_kl[best_idx])
        accum_vacuum_kl += float(delta_vacuum_kl[best_idx])

        # 4. Update the state probability in real space and renormalize
        current_prob = current_prob * exp_minus_2E[global_idx_group]
        current_prob /= np.sum(current_prob)

        active_mask[global_idx_group] = False

        # Clean tiny negative machine-precision noise on total KL only when storing
        # (Note: linear and vacuum parts can individually be negative, so do NOT clamp them)
        clean_best_kl = max(0.0, exact_best_kl)
        kls[..., kk] = clean_best_kl
        kls_linear[..., kk] = accum_linear_kl
        kls_vacuum[..., kk] = accum_vacuum_kl
        order[..., kk] = chosen_group[0]
        past_kl = clean_best_kl

    return kls, kls_linear, kls_vacuum, order


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
    pbc = True

    hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)
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
        psi_history = np.lib.format.open_memmap(
            filename,
            mode='r+',
            dtype=np.complex128,
            shape=(Ndt + 1, 2**L)
        )
        psi_t_idx = psi_history[t_idx]
        print(f"Loaded target wavefunction shape: {psi_t_idx.shape}")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Generating dummy target wavefunction.")
        psi_t_idx = np.ones(2**L, dtype=np.complex128) / np.sqrt(2**L)

    continuous_list = [False, True]

    local_filename = os.path.join(output_dir, f"groups_local_L{L}_a{angle}.npy")
    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")

    if not os.path.exists(global_filename):
        raise FileNotFoundError(f"Missing {global_filename}. Run generate_groups.py first.")

    dense_groups_local = np.load(local_filename)
    dense_groups_global = np.load(global_filename)

    for continuous in continuous_list:
        ngroups = len(dense_groups_global)
        idx_groups = [g[g != -1].tolist() for g in dense_groups_global]
        exact = np.abs(psi_t_idx)

        past_kl = 0.0
        kl_exact, kl_linear_exact, kl_vacuum_exact, order_exact = sweeps_KL(
            exact, exact, 2**L, hi, past_kl, L, idx_groups, ngroups, continuous=continuous
        )

        print(
            f"Continuous={continuous}: KLs shape {kl_exact.shape}, "
            f"Linear shape {kl_linear_exact.shape}, Vacuum shape {kl_vacuum_exact.shape}, "
            f"Order shape {order_exact.shape}"
        )

        suffix = f"L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
        ngroups = kl_exact.shape[-1]

        # 1. Save total KL
        filename_kl = os.path.join(output_dir, f"kl_{suffix}")
        psi_kl = np.lib.format.open_memmap(
            filename_kl,
            mode='w+',
            dtype=np.complex128,
            shape=(ngroups,)
        )
        psi_kl[:] = kl_exact[::-1]

        # 2. Save linear (non-vacuum) part of KL
        filename_kl_lin = os.path.join(output_dir, f"kl_linear_{suffix}")
        psi_kl_lin = np.lib.format.open_memmap(
            filename_kl_lin,
            mode='w+',
            dtype=np.complex128,
            shape=(ngroups,)
        )
        psi_kl_lin[:] = kl_linear_exact[::-1]

        # 3. Save vacuum (non-linear) part of KL
        filename_kl_vac = os.path.join(output_dir, f"kl_vacuum_{suffix}")
        psi_kl_vac = np.lib.format.open_memmap(
            filename_kl_vac,
            mode='w+',
            dtype=np.complex128,
            shape=(ngroups,)
        )
        psi_kl_vac[:] = kl_vacuum_exact[::-1]

        # 4. Save sorting order
        filename_sort = os.path.join(output_dir, f"sort_KL_{suffix}")
        psi_sort = np.lib.format.open_memmap(
            filename_sort,
            mode='w+',
            dtype=int,
            shape=(ngroups,)
        )
        psi_sort[:] = order_exact[::-1]
        print(order_exact)

        psi_kl.flush()
        psi_kl_lin.flush()
        psi_kl_vac.flush()
        psi_sort.flush()
        del psi_kl
        del psi_kl_lin
        del psi_kl_vac
        del psi_sort

    if 'psi_history' in locals():
        psi_history.flush()
        del psi_history
