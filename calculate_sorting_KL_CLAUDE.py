"""
calculate_sorting_KL_v2.py  —  KL-divergence greedy sorting, optimised for large systems.

Key improvements over v1
========================
1. Sparse O(k) group_kl_non_vac precomputation  (vs O(N) dense dot).
2. Sparse O(k·N) Walsh-character exp_minus_2E   (vs O(N log N) full WHT per group).
3. Single reused work-buffer V (no per-group allocation in hot loop).
4. float32 storage flag (--float32)  – halves exp_minus_2E memory.
5. Low-memory flag (--low-memory)    – O(N) extra memory, no precomputed exp table.
6. E_mean shift baked into group_kl_non_vac in normal mode; applied on-the-fly
   in low-memory mode.
7. Bug fixes: output dtype float64 (was complex128); division-by-zero guard in log.
8. Dead code removed: ordering=1.0 in Fast_Hadamard; unused batch_dims; unused
   dense_groups_local load.
9. List-based active-group tracking (avoids boolean-mask copies).

Memory (exp_minus_2E, one array):
  L=14 : ~153 MB (float64)  / ~76 MB (float32)
  L=16 : ~2.1  GB (float64) / ~1.05 GB (float32)
  L=18 : ~30   GB (float64) → use --low-memory
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


def _group_exponent_sparse(idxs_group, wht_log_psi, j_array):
    """
    E[j] = Σ_{c ∈ group, c≠0} wht_log_psi[c] · χ_c(j)   [O(k·N)]
    where k = |group| (excluding vacuum c=0).
    """
    E = np.zeros(len(j_array), dtype=np.float64)
    for c in idxs_group:
        if c == 0:
            continue
        E += wht_log_psi[c] * _walsh_chars(c, j_array)
    return E


# ---------------------------------------------------------------------------
# Precomputation (normal mode)
# ---------------------------------------------------------------------------

def precompute_kl_ops(
    target_psi, psi, idx_groups, ngroups, nstates, eps=1e-16, dtype=np.float64
):
    """
    Precompute:
      exp_minus_2E[i, j] = exp(-2 * (E_i(j) - mean(E_i)))          [ngroups × N]
      group_kl_non_vac[i] = 2N * Σ_c P_hat(c)·C_c^ψ - 2·mean(E_i)  [ngroups]

    E_mean shift is baked into both arrays so the main loop needs no correction.

    Uses sparse Walsh characters (O(k·N)) when k ≤ log2(N)/2,
    otherwise the full WHT route (O(N log N)).
    """
    log2N = int(round(np.log2(nstates)))

    wht_P_target = Fast_Hadamard(target_psi ** 2)       # P_hat(A)
    wht_log_psi  = Fast_Hadamard(np.log(np.maximum(psi, eps)))   # C_A^ψ

    j_array = np.arange(nstates, dtype=np.int32)
    V = np.zeros(nstates, dtype=np.float64)             # reused buffer

    exp_minus_2E   = np.empty((ngroups, nstates), dtype=dtype)
    group_kl_non_vac = np.zeros(ngroups, dtype=np.float64)

    threshold = max(1, log2N // 2)

    for i, idxs_group in enumerate(idx_groups):
        non_vac = [c for c in idxs_group if c != 0]
        k = len(non_vac)

        # --- O(k) sparse KL non-vacuum term ---
        kl_nv = 0.0
        for c in non_vac:
            kl_nv += wht_P_target[c] * wht_log_psi[c]
        group_kl_non_vac[i] = 2.0 * nstates * kl_nv

        # --- exponent E(j) ---
        if k == 0:
            exp_minus_2E[i] = 1.0
            continue

        if k <= threshold:
            E = _group_exponent_sparse(non_vac, wht_log_psi, j_array)
        else:
            V[:] = 0.0
            for c in non_vac:
                V[c] = wht_log_psi[c]
            E = Fast_Hadamard(V) * nstates

        E_mean = float(np.mean(E))
        E_shifted = E - E_mean
        exp_minus_2E[i] = np.exp(-2.0 * E_shifted).astype(dtype)

        # Compensate E_mean in the KL non-vacuum term
        group_kl_non_vac[i] -= 2.0 * E_mean

    return exp_minus_2E, group_kl_non_vac, wht_log_psi


def precompute_kl_non_vac_only(target_psi, psi, idx_groups, ngroups, nstates, eps=1e-16):
    """
    Precompute only group_kl_non_vac (without E_mean correction) for low-memory mode.
    E_mean correction is applied on-the-fly in the main loop.

    Returns (group_kl_non_vac_raw, wht_P_target, wht_log_psi).
    """
    wht_P_target = Fast_Hadamard(target_psi ** 2)
    wht_log_psi  = Fast_Hadamard(np.log(np.maximum(psi, eps)))

    group_kl_non_vac_raw = np.zeros(ngroups, dtype=np.float64)
    for i, idxs_group in enumerate(idx_groups):
        kl_nv = 0.0
        for c in idxs_group:
            if c != 0:
                kl_nv += wht_P_target[c] * wht_log_psi[c]
        group_kl_non_vac_raw[i] = 2.0 * nstates * kl_nv

    return group_kl_non_vac_raw, wht_P_target, wht_log_psi


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweeps_KL(
    target_psi,
    psi,
    nstates,
    hi,
    past_kl,
    L,
    idx_groups,
    ngroups,
    continuous=False,
    use_float32=False,
    low_memory=False,
    eps=1e-16,
):
    """
    Greedy KL-divergence minimising sweep.

    Returns
    -------
    kls   : (ngroups,) float64  – KL at each step
    order : (ngroups,) int64    – first index of chosen group at each step
    """
    # Normalise (real-space, positive)
    target_psi = np.abs(target_psi)
    target_psi = target_psi / np.sqrt(np.sum(target_psi ** 2))
    psi = np.abs(psi)
    psi = psi / np.sqrt(np.sum(psi ** 2))

    dtype = np.float32 if use_float32 else np.float64
    j_array = np.arange(nstates, dtype=np.int32)

    # ------------------------------------------------------------------
    if not low_memory:
        print("Precomputing exp_minus_2E and group_kl_non_vac …")
        exp_minus_2E, group_kl_non_vac, wht_log_psi = precompute_kl_ops(
            target_psi, psi, idx_groups, ngroups, nstates, eps=eps, dtype=dtype
        )
    else:
        print("Low-memory mode: precomputing group_kl_non_vac only …")
        group_kl_non_vac_raw, _wht_P_target, wht_log_psi = precompute_kl_non_vac_only(
            target_psi, psi, idx_groups, ngroups, nstates, eps=eps
        )

    # ------------------------------------------------------------------
    kls   = np.zeros(ngroups, dtype=np.float64)
    order = np.zeros(ngroups, dtype=np.int64)

    active_indices = list(range(ngroups))

    # Running probability and initial KL
    P_target     = target_psi ** 2
    current_prob = psi ** 2
    accum_total_kl = float(
        np.sum(P_target * np.log(np.maximum(P_target, eps) / np.maximum(current_prob, eps)))
    )

    for kk in range(ngroups):
        n_active = len(active_indices)

        # ------------------------------------------------------------------
        # Compute candidate KLs for all active groups
        # ------------------------------------------------------------------
        if not low_memory:
            rows_2E = exp_minus_2E[active_indices]          # (n_active, N)
            norm_sq  = rows_2E @ current_prob               # (n_active,)  incremental vacuum
            delta_vacuum_kl = np.log(np.maximum(norm_sq, 1e-300))

            raw_group_kls = (
                accum_total_kl
                + group_kl_non_vac[active_indices]
                + delta_vacuum_kl
            )
        else:
            # On-the-fly: compute exponent for each active group
            norm_sq_arr         = np.empty(n_active, dtype=np.float64)
            e_mean_corrections  = np.empty(n_active, dtype=np.float64)

            for loc_i, glob_i in enumerate(active_indices):
                E     = _group_exponent_sparse(idx_groups[glob_i], wht_log_psi, j_array)
                E_mean = float(np.mean(E))
                E_sh  = E - E_mean
                ev2   = np.exp(-2.0 * E_sh)
                norm_sq_arr[loc_i]        = float(np.dot(ev2, current_prob))
                e_mean_corrections[loc_i] = E_mean

            delta_vacuum_kl = np.log(np.maximum(norm_sq_arr, 1e-300))

            # Apply E_mean correction on-the-fly
            kl_nv_corrected = (
                group_kl_non_vac_raw[active_indices] - 2.0 * e_mean_corrections
            )
            raw_group_kls = accum_total_kl + kl_nv_corrected + delta_vacuum_kl

        # ------------------------------------------------------------------
        # Select best group
        # ------------------------------------------------------------------
        if not continuous:
            local_best = int(np.argmin(raw_group_kls))
        else:
            local_best = int(np.argmin(np.abs(past_kl - raw_group_kls)))

        global_idx   = active_indices[local_best]
        chosen_group = idx_groups[global_idx]
        exact_best_kl = float(raw_group_kls[local_best])
        accum_total_kl = exact_best_kl

        # ------------------------------------------------------------------
        # Update current_prob incrementally
        # ------------------------------------------------------------------
        if not low_memory:
            ev2_chosen = exp_minus_2E[global_idx].astype(np.float64)
        else:
            E_ch   = _group_exponent_sparse(idx_groups[global_idx], wht_log_psi, j_array)
            E_sh   = E_ch - float(np.mean(E_ch))
            ev2_chosen = np.exp(-2.0 * E_sh)

        current_prob *= ev2_chosen
        prob_sum = float(np.sum(current_prob))
        if prob_sum > 1e-300:
            current_prob /= prob_sum

        active_indices.pop(local_best)

        # Store (clamp tiny negative machine-precision noise)
        clean_kl  = max(0.0, exact_best_kl)
        kls[kk]   = clean_kl
        order[kk] = chosen_group[0]
        past_kl   = clean_kl

    return kls, order


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KL-divergence greedy coefficient sorting (v2, large-system optimised)"
    )
    parser.add_argument("--L",        type=int,   default=14)
    parser.add_argument("--gi",       type=float, default=1.5)
    parser.add_argument("--gf",       type=float, default=0.5)
    parser.add_argument("--angle",    type=float, default=0.0)
    parser.add_argument("--t_start",  type=float, default=0.0)
    parser.add_argument("--t_idx",    type=int,   default=10)
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Store exp_minus_2E in float32 (halves memory; minor precision loss)",
    )
    parser.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        help="Skip precomputing exp table; O(N) extra memory, L× slower per sweep step",
    )
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

    use_float32 = params["float32"]
    low_memory  = params["low_memory"]

    hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)

    add = ""
    if pbc:
        add += "_pbc"
    if angle != 0.0:
        add += "_" + str(angle)

    output_dir = "wavefunctions_data" + add
    os.makedirs(output_dir, exist_ok=True)

    # ---- memory table ----
    N = 2 ** L
    mem_f64 = nk.hilbert.Spin(s=1/2, N=L).n_states * 8 / 1e9 if False else N * 8 / 1e9
    print(f"\nMemory estimate for exp_minus_2E (ngroups ~ N = {N}):")
    print(f"  float64 : {N * N * 8 / 1e9:.2f} GB")
    print(f"  float32 : {N * N * 4 / 1e9:.2f} GB")
    print(f"  (ngroups may be << N for PBC; actual usage is ngroups × N × bytes)\n")

    # ---- load target wavefunction ----
    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename  = os.path.join(output_dir, base_name)
    try:
        psi_history = np.lib.format.open_memmap(
            filename, mode="r+", dtype=np.complex128, shape=(Ndt + 1, 2 ** L)
        )
        psi_t_idx = psi_history[t_idx]
        print(f"Loaded target wavefunction shape: {psi_t_idx.shape}")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using uniform dummy wavefunction.")
        psi_t_idx = np.ones(2 ** L, dtype=np.complex128) / np.sqrt(2 ** L)

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

    # ---- run for both continuous modes ----
    for continuous in [False, True]:
        past_kl = 0.0
        kl_exact, order_exact = sweeps_KL(
            exact, exact, 2 ** L, hi, past_kl, L, idx_groups, ngroups,
            continuous=continuous,
            use_float32=use_float32,
            low_memory=low_memory,
        )

        print(
            f"Continuous={continuous}: KLs shape {kl_exact.shape}, "
            f"Order shape {order_exact.shape}"
        )

        # ---- save KL values ----
        base_name_kl = (
            f"kl_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}"
            f"C{continuous}.npy"
        )
        filename_kl = os.path.join(output_dir, base_name_kl)
        psi_kl = np.lib.format.open_memmap(
            filename_kl, mode="w+", dtype=np.float64, shape=(ngroups,)   # fixed: was complex128
        )
        psi_kl[:] = kl_exact[::-1]
        psi_kl.flush()
        del psi_kl

        # ---- save sort order ----
        base_name_sort = (
            f"sort_KL_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}"
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

    if "psi_history" in locals():
        psi_history.flush()
        del psi_history
