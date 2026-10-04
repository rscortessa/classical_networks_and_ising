#!/usr/bin/env python
# coding: utf-8
"""
Optimized infidelity-minimizing sorting of WHT coefficients.

Key algorithmic improvement over the original:
  The original sweeps()/single_sweep() with candidates=1 simply follows the
  pre-sorted |log_WHT| order — it does no real greedy search.

  Here we replace that with a TRUE greedy algorithm that at each step:
    1. Computes exact infidelities for ALL remaining coefficients simultaneously
       using two Walsh-Hadamard Transforms (WHT) — O(N log N) per step.
    2. Picks the coefficient that gives the minimum infidelity.

  Mathematical insight (see derivation below):
    After applying a subset S of WHT coefficients, psi_j is known.
    For candidate coefficient c, the new (un-normalized) overlap with exact is:

        overlap_c = cosh(hwt[c])·Σ_a  −  N·WHT(a)[c]·sinh(hwt[c])
        norm_c²   = cosh(2hwt[c])·Σ_b −  N·WHT(b)[c]·sinh(2hwt[c])

    where a_j = exact_j·psi_j,  b_j = psi_j²,  Σ_a = Σ_j a_j,  Σ_b = 1.

    WHT(a) and WHT(b) give us every coefficient simultaneously in O(N log N),
    so we evaluate ALL remaining candidates in one vectorized pass.

  Complexity: O((N/2) · N log N) total,  fully vectorized.
  With N = 2^L the original was effectively O(N) (candidates=1, no real search);
  the TRUE greedy here always finds the globally optimal next coefficient.
"""

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

import numpy as np


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (same as original)
# ---------------------------------------------------------------------------

def Fast_Hadamard(a):
    """Normalized Walsh-Hadamard Transform.  WHT(e_c)_j = chi_c(j)/N
    where chi_c(j) = (-1)^popcount(c & j) and N = len(a)."""
    a = a.copy()
    L = a.shape[-1]
    h = 1
    while h < L:
        a2 = a.reshape(a.shape[:-1] + (L // (2 * h), 2, h))
        x = a2[..., 0, :]
        y = a2[..., 1, :]
        t = x.copy()
        a2[..., 0, :] = 0.5 * (t + y)
        a2[..., 1, :] = 0.5 * (t - y)
        h *= 2
    return a


# ---------------------------------------------------------------------------
# Infidelity (same as original)
# ---------------------------------------------------------------------------

def inf(psi, phi):
    return 1.0 - (np.einsum('...m,...m->...', psi, phi, optimize=True)) ** 2


# ---------------------------------------------------------------------------
# Core update (same as original — kept for reference / comparison)
# ---------------------------------------------------------------------------

def update(id_coeff, hwt, nstates):
    aux = np.zeros_like(hwt)
    id_coeff_expanded = id_coeff[..., np.newaxis]
    vals = np.take_along_axis(hwt, id_coeff_expanded, axis=-1)
    np.put_along_axis(aux, id_coeff_expanded, -vals, axis=-1)
    return np.exp(Fast_Hadamard(aux) * nstates)


# ---------------------------------------------------------------------------
# NEW: efficient greedy sorting algorithm
# ---------------------------------------------------------------------------

def greedy_wht_sorting(exact, hwt, candidate_indices):
    """
    True greedy sorting that minimizes the infidelity curve.

    At step k it selects the coefficient  c*  from the remaining candidates
    that yields the lowest infidelity when applied to the current psi.
    All candidates are evaluated in a single vectorized computation
    that exploits the WHT structure — no inner loop over states.

    Parameters
    ----------
    exact : ndarray, shape (N,)
        Exact (target) wavefunction, real and normalized.
    hwt : ndarray, shape (N,)
        Real part of WHT( log(exact) ).  Same as log_WHT.real in the notebook.
    candidate_indices : ndarray, shape (M,)
        Indices into hwt to consider (e.g. bare_sorting from original code).

    Returns
    -------
    order : ndarray, shape (M,), dtype int
        Optimal greedy ordering of the candidate coefficient indices.
    infs  : ndarray, shape (M,), float
        Infidelity after each coefficient is applied (monotone).
    """
    N = len(exact)
    M = len(candidate_indices)

    psi = exact.copy()                        # start from exact  (inf = 0)
    available = np.ones(M, dtype=bool)        # which candidates remain

    order = np.empty(M, dtype=int)
    infs  = np.empty(M, dtype=float)

    # hwt values for the candidates, precomputed
    hwt_cands = hwt[candidate_indices]        # shape (M,)

    for step in range(M):
        # ------------------------------------------------------------------
        # Compute a_j = exact_j · psi_j  and  b_j = psi_j²
        # ------------------------------------------------------------------
        a = exact * psi                        # shape (N,)
        b = psi * psi                          # shape (N,)  (psi is normalized → Σb = 1)

        sum_a = np.dot(exact, psi)             # current overlap
        # sum_b = 1.0  (psi normalized)

        # ------------------------------------------------------------------
        # Walsh-Hadamard transforms — O(N log N) each
        # ------------------------------------------------------------------
        wht_a = Fast_Hadamard(a)               # WHT(a)[c] = (1/N)·Σ_j chi_c(j)·a_j
        wht_b = Fast_Hadamard(b)

        # ------------------------------------------------------------------
        # For each available candidate c:
        #
        #   update(c) multiplies psi_j by exp(-chi_c(j) · hwt[c])
        #   (see derivation in module docstring for sign convention)
        #
        #   A_c = Σ_{chi_c(j)=+1} a_j = (sum_a + N·wht_a[c]) / 2
        #   B_c = Σ_{chi_c(j)=-1} a_j = (sum_a − N·wht_a[c]) / 2
        #
        #   new_overlap  = exp(-hwt[c])·A_c + exp(+hwt[c])·B_c
        #   new_norm²    = exp(-2hwt[c])·A'_c + exp(+2hwt[c])·B'_c
        #                  (A', B' same formula with b instead of a)
        #
        #   infidelity_c = 1 − new_overlap² / new_norm²
        # ------------------------------------------------------------------
        avail_mask = available                             # boolean shape (M,)
        hwt_c = hwt_cands[avail_mask]                     # shape (n_avail,)

        cand_global = candidate_indices[avail_mask]       # global indices
        wht_a_c = wht_a[cand_global]                      # shape (n_avail,)
        wht_b_c = wht_b[cand_global]

        A_c = (sum_a + N * wht_a_c) / 2.0
        B_c = (sum_a - N * wht_a_c) / 2.0
        Ap_c = (1.0  + N * wht_b_c) / 2.0    # sum_b = 1
        Bp_c = (1.0  - N * wht_b_c) / 2.0

        e_neg = np.exp(-hwt_c)
        e_pos = np.exp( hwt_c)

        new_overlap = e_neg * A_c + e_pos * B_c
        new_norm_sq = e_neg**2 * Ap_c + e_pos**2 * Bp_c

        infidelity_c = 1.0 - new_overlap**2 / np.maximum(new_norm_sq, 1e-300)

        # ------------------------------------------------------------------
        # Select best candidate (minimum infidelity)
        # ------------------------------------------------------------------
        best_local = int(np.argmin(infidelity_c))

        # Map back to full-array indices
        avail_positions = np.where(avail_mask)[0]         # positions in [0,M)
        best_M_idx      = avail_positions[best_local]     # index inside M
        best_global_idx = candidate_indices[best_M_idx]   # true coeff index

        # ------------------------------------------------------------------
        # Apply chosen coefficient to psi
        # ------------------------------------------------------------------
        delta = update(np.array([best_global_idx]), hwt, N)   # shape (1, N)
        psi = psi * delta[0]
        norm = np.linalg.norm(psi)
        psi /= norm

        # ------------------------------------------------------------------
        # Record and mark used
        # ------------------------------------------------------------------
        order[step]  = best_global_idx
        infs[step]   = infidelity_c[best_local]
        available[best_M_idx] = False

    return order, infs


# ---------------------------------------------------------------------------
# Convenience: build the same bare_sorting as the original notebook
# ---------------------------------------------------------------------------

def initial_candidate_indices(log_wht_real, half_N):
    """Sort upper half of spectrum by |log_WHT| — matches original bare_sorting."""
    return np.argsort(np.abs(log_wht_real))[half_N:]


# ---------------------------------------------------------------------------
# Main — same setup as original notebook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import netket as nk
    import sys
    sys.path.append("../netket_with_NQS")

    L          = 10
    hi         = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)
    gi         = 1.5
    gf         = 0.5
    angle      = 0.0

    t_start = 0.0
    dt      = 0.01
    Ndt     = 100
    t_idx   = 60

    output_dir = "wavefunctions_data"
    os.makedirs(output_dir, exist_ok=True)

    # Load wavefunction
    base_name    = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename     = os.path.join(output_dir, base_name)
    psi_history  = np.lib.format.open_memmap(filename, mode='r+',
                                              dtype=np.complex128, shape=(Ndt+1, 2**L))
    psi_t_idx    = psi_history[t_idx]

    log_WHT      = Fast_Hadamard(np.log(psi_t_idx))
    exact        = np.abs(psi_t_idx)
    hwt          = log_WHT.real
    nstates      = 2**L

    # Initial candidate set (upper half by |log_WHT|), same as original
    bare_sorting = initial_candidate_indices(hwt, nstates // 2)

    print("Running greedy WHT sorting …")
    order_new, infs_new = greedy_wht_sorting(exact, hwt, bare_sorting)
    print(f"  infidelity range: {infs_new.min():.4e} → {infs_new.max():.4e}")
    print(f"  order shape: {order_new.shape}")

    # Save — same filename convention as original
    base_inf  = (f"inf_L{L}gi{gi:.2f}gf{gf:.2f}"
                 f"ti{t_start}dt{dt}t{t_idx}greedyWHT_Nt{Ndt}.npy")
    base_sort = (f"sort_L{L}gi{gi:.2f}gf{gf:.2f}"
                 f"ti{t_start}dt{dt}t{t_idx}greedyWHT_Nt{Ndt}.npy")

    psi_inf = np.lib.format.open_memmap(
        os.path.join(output_dir, base_inf), mode='w+',
        dtype=np.complex128, shape=(nstates // 2,))
    psi_sort = np.lib.format.open_memmap(
        os.path.join(output_dir, base_sort), mode='w+',
        dtype=int, shape=(nstates // 2,))

    psi_inf[:]  = infs_new[::-1]
    psi_sort[:] = order_new[::-1]

    psi_inf.flush()
    psi_sort.flush()
    psi_history.flush()
    print("Done.")
