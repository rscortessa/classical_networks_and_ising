#!/usr/bin/env python
# coding: utf-8
"""
calculate_sorting_KL_natural.py — Natural (linear-KL) coefficient sorting.

Selects the group ordering using ONLY the linear (non-vacuum WHT) part of the
KL divergence:
    group_kl_non_vac[i] = 2 * 2^L * sum_{c in group_i, c != 0} P_hat[c] * C_c^psi
while still tracking and saving the EXACT KL divergence (including the vacuum
log-partition normalization shift) along that trajectory.

Because candidate selection only depends on the 1D array `group_kl_non_vac`,
we NEVER need to precompute or store the (ngroups × 2^L) `exp_minus_2E` matrix!
Memory usage is strictly O(2^L) and candidate selection takes O(ngroups) per step.
"""

import os
import argparse

# Restrict threads BEFORE importing jax/numpy
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
)

import numpy as np
import netket as nk


# ---------------------------------------------------------------------------
# Walsh–Hadamard Transform (normalised: WHT(WHT(a)) * N = a)
# ---------------------------------------------------------------------------

def Fast_Hadamard(a):
    """
    Normalised Walsh-Hadamard Transform.
    f_hat(A) = (1 / 2^N) * sum_s f(s) X_A(s)
    Inverse: f(s) = 2^N * Fast_Hadamard(f_hat)(s)
    """
    a = a.copy()
    L = a.shape[-1]
    h = 1
    while h < L:
        a2 = a.reshape(a.shape[:-1] + (L // (2 * h), 2, h))
        x = a2[..., 0, :].copy()
        y = a2[..., 1, :]
        a2[..., 0, :] = 0.5 * (x + y)
        a2[..., 1, :] = 0.5 * (x - y)
        h *= 2
    return a


# ---------------------------------------------------------------------------
# Parity look-up table (Walsh characters via popcount parity)
# ---------------------------------------------------------------------------

_parity16 = np.array([bin(x).count("1") % 2 for x in range(65536)], dtype=np.int8)


def _walsh_chars(c, j_array):
    """χ_c(j) = (-1)^popcount(c & j) for all j, returned as ±1 float64."""
    masked = (j_array & c).astype(np.int32)
    parity = _parity16[masked & 0xFFFF] ^ _parity16[(masked >> 16) & 0xFFFF]
    return 1.0 - 2.0 * parity.astype(np.float64)


def _group_exponent(non_vac_group, wht_log_psi, j_array, V_buf, nstates, threshold):
    """
    Computes the real-space exponent E[j] = sum_{c in group, c!=0} wht_log_psi[c] * X_c(j)
    for a SINGLE chosen group on the fly.
    Uses sparse Walsh characters O(k*N) when k <= threshold, else full WHT O(N log N).
    """
    k = len(non_vac_group)
    if k == 0:
        return np.zeros(nstates, dtype=np.float64)

    if k <= threshold:
        E = np.zeros(nstates, dtype=np.float64)
        for c in non_vac_group:
            E += wht_log_psi[c] * _walsh_chars(c, j_array)
        return E
    else:
        V_buf[:] = 0.0
        for c in non_vac_group:
            V_buf[c] = wht_log_psi[c]
        return Fast_Hadamard(V_buf) * nstates


# ---------------------------------------------------------------------------
# Precomputation of the 1D Linear KL array (O(2^L) total time & memory)
# ---------------------------------------------------------------------------

def precompute_linear_kl(target_psi, psi, idx_groups, ngroups, nstates, eps=1e-16):
    """
    Precomputes ONLY the linear (non-vacuum WHT) KL contribution for each group:
      group_kl_non_vac[i] = 2 * nstates * sum_{c in group_i, c != 0} P_hat[c] * C_c^psi
    """
    wht_P_target = Fast_Hadamard(target_psi ** 2)
    wht_log_psi  = Fast_Hadamard(np.log(np.maximum(psi, eps)))

    group_kl_non_vac = np.zeros(ngroups, dtype=np.float64)
    for i, idxs_group in enumerate(idx_groups):
        non_vac = [c for c in idxs_group if c != 0]
        if non_vac:
            g_idx = np.asarray(non_vac, dtype=np.intp)
            group_kl_non_vac[i] = (2.0 * nstates) * float(
                np.sum(wht_P_target[g_idx] * wht_log_psi[g_idx])
            )

    return group_kl_non_vac, wht_log_psi


# ---------------------------------------------------------------------------
# Natural-Ordering Sweep (Linear KL selection, Exact KL evaluation)
# ---------------------------------------------------------------------------

def sweeps_KL_natural(
    target_psi,
    psi,
    nstates,
    hi,
    past_kl,
    L,
    idx_groups,
    ngroups,
    continuous=False,
    eps=1e-16,
     precomputed=None,
):
    """
    Orders groups using ONLY the linear WHT part of the KL divergence,
    while computing and returning the EXACT KL divergence at each step.

    Returns
    -------
    kls   : (ngroups,) float64  – Exact KL at each step along the natural ordering
    order : (ngroups,) int64    – first index of chosen group at each step
    """
    target_psi = np.abs(target_psi)
    target_psi = target_psi / np.sqrt(np.sum(target_psi ** 2))
    psi = np.abs(psi)
    psi = psi / np.sqrt(np.sum(psi ** 2))

    if precomputed is None:
        print("Precomputing linear WHT KL contributions (group_kl_non_vac) …")
        group_kl_non_vac, wht_log_psi = precompute_linear_kl(
            target_psi, psi, idx_groups, ngroups, nstates, eps=eps
        )
    else:
        group_kl_non_vac, wht_log_psi = precomputed

    log2N = int(round(np.log2(nstates)))
    threshold = max(1, log2N // 2)
    j_array = np.arange(nstates, dtype=np.int32)
    V_buf = np.zeros(nstates, dtype=np.float64)

    kls   = np.zeros(ngroups, dtype=np.float64)
    order = np.zeros(ngroups, dtype=np.int64)

    active_indices = list(range(ngroups))

    # Running probability and initial exact KL
    P_target     = target_psi ** 2
    current_prob = (psi ** 2).copy()
    accum_total_kl = float(
        np.sum(P_target * np.log(np.maximum(P_target, eps) / np.maximum(current_prob, eps)))
    )

    for kk in range(ngroups):
        # ------------------------------------------------------------------
        # 1. Select best group using ONLY the linear part of the KL
        # ------------------------------------------------------------------
        linear_candidates = accum_total_kl + group_kl_non_vac[active_indices]

        if not continuous:
            # Note: adding scalar `accum_total_kl` does not change argmin,
            # so this naturally picks the smallest `group_kl_non_vac`!
            local_best = int(np.argmin(linear_candidates))
        else:
            local_best = int(np.argmin(np.abs(past_kl - linear_candidates)))

        global_idx   = active_indices.pop(local_best)
        chosen_group = idx_groups[global_idx]

        # ------------------------------------------------------------------
        # 2. Compute the EXACT KL for the chosen group (including vacuum term)
        # ------------------------------------------------------------------
        non_vac_chosen = [c for c in chosen_group if c != 0]
        E_chosen = _group_exponent(
            non_vac_chosen, wht_log_psi, j_array, V_buf, nstates, threshold
        )
        E_mean = float(np.mean(E_chosen))
        E_shifted = E_chosen - E_mean
        ev2_chosen = np.exp(-2.0 * E_shifted)

        # Exact vacuum contribution: log( sum_s current_prob(s) * exp(-2 E_chosen(s)) )
        norm_sq = float(np.dot(ev2_chosen, current_prob))
        delta_vacuum_kl = float(np.log(max(norm_sq, 1e-300)))

        # Exact total KL update (compensating for E_mean shift)
        exact_kl = (
            accum_total_kl
            + group_kl_non_vac[global_idx]
            - 2.0 * E_mean
            + delta_vacuum_kl
        )
        accum_total_kl = float(exact_kl)

        # ------------------------------------------------------------------
        # 3. Update current_prob in real space and renormalise
        # ------------------------------------------------------------------
        current_prob *= ev2_chosen
        prob_sum = float(np.sum(current_prob))
        if prob_sum > 1e-300:
            current_prob /= prob_sum

        # Store exact KL (clamping tiny machine-precision negative noise)
        clean_kl  = max(0.0, accum_total_kl)
        kls[kk]   = clean_kl
        order[kk] = chosen_group[0]
        past_kl   = clean_kl

    return kls, order


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Natural (linear-KL) coefficient sorting with exact KL evaluation"
    )
    parser.add_argument("--L",       type=int,   default=14)
    parser.add_argument("--gi",      type=float, default=1.5)
    parser.add_argument("--gf",      type=float, default=0.5)
    parser.add_argument("--angle",   type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx",   type=int,   default=10)

    args, _ = parser.parse_known_args()
    params = vars(args)

    dt   = 0.01
    Ndt  = 100

    t_idx   = params["t_idx"]
    L       = params["L"]
    gi      = params["gi"]
    gf      = params["gf"]
    angle   = params["angle"]
    t_start = params["t_start"]
    pbc     = True
    nstates = 2 ** L

    hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)

    add = ""
    if pbc:
        add += "_pbc"
    if angle != 0.0:
        add += "_" + str(angle)

    output_dir = "wavefunctions_data" + add
    os.makedirs(output_dir, exist_ok=True)

    # ---- load target wavefunction ----
    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename  = os.path.join(output_dir, base_name)
    try:
        psi_history = np.lib.format.open_memmap(
            filename, mode="r", dtype=np.complex128, shape=(Ndt + 1, nstates)
        )
        psi_t_idx = np.array(psi_history[t_idx], copy=True)
        del psi_history
        print(f"Loaded target wavefunction shape: {psi_t_idx.shape}")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using uniform dummy wavefunction.")
        psi_t_idx = np.ones(nstates, dtype=np.complex128) / np.sqrt(nstates)

    # ---- load groups ----
    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")
    if not os.path.exists(global_filename):
        raise FileNotFoundError(
            f"Missing {global_filename}. Run generate_groups.py first."
        )
    dense_groups_global = np.load(global_filename)

    idx_groups = [g[g != -1].tolist() for g in dense_groups_global]
    ngroups    = len(idx_groups)
    exact      = np.abs(psi_t_idx)

    # Normalize once and precompute linear KL array once for both continuous modes
    exact_norm = exact / np.sqrt(np.sum(exact ** 2))
    precomputed = precompute_linear_kl(
        exact_norm, exact_norm, idx_groups, ngroups, nstates
    )

    # ---- run for both continuous modes ----
    for continuous in [False, True]:
        past_kl = 0.0
        kl_exact, order_exact = sweeps_KL_natural(
            exact_norm, exact_norm, nstates, hi, past_kl, L, idx_groups, ngroups,
            continuous=continuous,
            precomputed=precomputed,
        )

        print(
            f"[Natural] Continuous={continuous}: KLs shape {kl_exact.shape}, "
            f"Order shape {order_exact.shape}"
        )

        # ---- save KL values (distinguished by `kl_nat_` prefix) ----
        base_name_kl = (
            f"kl_nat_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}"
            f"C{continuous}.npy"
        )
        filename_kl = os.path.join(output_dir, base_name_kl)
        psi_kl = np.lib.format.open_memmap(
            filename_kl, mode="w+", dtype=np.float64, shape=(ngroups,)
        )
        psi_kl[:] = kl_exact[::-1]
        psi_kl.flush()
        del psi_kl

        # ---- save sort order (distinguished by `sort_KL_nat_` prefix) ----
        base_name_sort = (
            f"sort_KL_nat_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}"
            f"C{continuous}.npy"
        )
        filename_sort = os.path.join(output_dir, base_name_sort)
        psi_sort = np.lib.format.open_memmap(
            filename_sort, mode="w+", dtype=int, shape=(ngroups,)
        )
        psi_sort[:] = order_exact[::-1]
        print(order_exact)
        psi_sort.flush()
        del psi_sort
