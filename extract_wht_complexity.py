#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


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


def extract_metrics_for_time(wht_log_psi, sort_order, rep_to_size, eps_l1=1e-2, eps_l2=1e-4):
    """
    Extracts scalar complexity metrics from the sorted WHT expansion at a single time step:
      1. Integrated Tail Area (A_L1, A_L2)
      2. Epsilon-Truncation Rank (N_eps_L1, N_eps_L2)
      3. Bulk Exponential Decay Scale xi_L1 (from linear fit on [15%, 75%] of spectrum)
      4. WHT Spectral Entropy S_WHT and Effective Number of Modes exp(S_WHT)
    """
    valid_reps = np.array([r for r in sort_order if r != 0], dtype=int)
    abs_c = np.abs(wht_log_psi[valid_reps].real)
    n_terms = np.array([rep_to_size[r] for r in valid_reps], dtype=np.float64)

    cum_coeffs = np.cumsum(n_terms)
    n_total = cum_coeffs[-1]

    w_l1 = n_terms * abs_c
    w_l2 = n_terms * (abs_c ** 2)

    total_l1 = np.sum(w_l1)
    total_l2 = np.sum(w_l2)

    R_l1 = (np.cumsum(w_l1[::-1])[::-1] - w_l1) / total_l1
    R_l2 = (np.cumsum(w_l2[::-1])[::-1] - w_l2) / total_l2

    # Prepend (0, 1.0)
    x_full = np.insert(cum_coeffs, 0, 0.0)
    R_l1_full = np.insert(R_l1, 0, 1.0)
    R_l2_full = np.insert(R_l2, 0, 1.0)

    # 1. Normalized Area Under the Tail Curve (Trapezoidal integration in [0, 1])
    area_l1 = np.trapz(R_l1_full, x_full) / n_total
    area_l2 = np.trapz(R_l2_full, x_full) / n_total

    # 2. Epsilon-Truncation Rank: N_kept / N_total needed to reach R <= eps
    idx_l1 = np.searchsorted(-R_l1_full, -eps_l1)
    n_eps_l1 = x_full[min(idx_l1, len(x_full) - 1)] / n_total

    idx_l2 = np.searchsorted(-R_l2_full, -eps_l2)
    n_eps_l2 = x_full[min(idx_l2, len(x_full) - 1)] / n_total

    # 3. Bulk Exponential Decay Rate: fit log10(R_L1) = -gamma * (x / N_total) + const
    # Fit over the bulk window x / N_total in [0.15, 0.75]
    x_norm = cum_coeffs[:-1] / n_total
    bulk_mask = (x_norm >= 0.15) & (x_norm <= 0.75) & (R_l1[:-1] > 1e-15)
    if np.sum(bulk_mask) > 5:
        slope_l1, _ = np.polyfit(x_norm[bulk_mask], np.log10(R_l1[:-1][bulk_mask]), 1)
        decay_rate_l1 = -slope_l1  # Decades of decay across normalized spectrum
    else:
        decay_rate_l1 = np.nan

    # 4. WHT Spectral Shannon Entropy (using L1 and L2 distributions over individual coefficients)
    # Each group j has `n_terms[j]` coefficients, each carrying individual probability p_ind = (w_j / total) / n_terms[j]
    p_group_l1 = w_l1 / total_l1
    valid_p1 = p_group_l1 > 1e-30
    S_l1 = -np.sum(p_group_l1[valid_p1] * np.log(p_group_l1[valid_p1] / n_terms[valid_p1])) / np.log(n_total)

    p_group_l2 = w_l2 / total_l2
    valid_p2 = p_group_l2 > 1e-30
    S_l2 = -np.sum(p_group_l2[valid_p2] * np.log(p_group_l2[valid_p2] / n_terms[valid_p2])) / np.log(n_total)

    return {
        "area_l1": area_l1,
        "area_l2": area_l2,
        "n_eps_l1": n_eps_l1,
        "n_eps_l2": n_eps_l2,
        "decay_rate_l1": decay_rate_l1,
        "S_l1": S_l1,
        "S_l2": S_l2,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and plot WHT complexity metrics vs time.")
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--eps-l1", type=float, default=1e-2, help="Threshold for L1 truncation rank.")
    parser.add_argument("--eps-l2", type=float, default=1e-4, help="Threshold for L2 truncation rank.")

    args = parser.parse_args()

    L = args.L
    gi = args.gi
    gf = args.gf
    angle = args.angle
    t_start = args.t_start
    dt = args.dt
    Ndt = args.Ndt
    nstates = 1 << L

    add = "_pbc" + (f"_{angle}" if angle != 0.0 else "")
    output_dir = "wavefunctions_data" + add

    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")
    dense_groups_global = np.load(global_filename)
    rep_to_size = {}
    for g in dense_groups_global:
        valid_g = g[g != -1]
        if len(valid_g) > 0:
            rep_to_size[int(valid_g[0])] = len(valid_g)

    wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    wf_file = os.path.join(output_dir, wf_base)
    psi_history = np.lib.format.open_memmap(
        wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
    )

    t_idx_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    for continuous, ls, marker in [(False, '-', 'o'), (True, '--', 's')]:
        times = []
        metrics_list = []

        for t_idx in t_idx_list:
            sort_base = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
            sort_file = os.path.join(output_dir, sort_base)
            if not os.path.exists(sort_file):
                continue

            sort_order = np.array(np.lib.format.open_memmap(sort_file, mode='r', dtype=int))
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            wht_log_psi = Fast_Hadamard(np.log(psi_t))

            m = extract_metrics_for_time(
                wht_log_psi, sort_order, rep_to_size, eps_l1=args.eps_l1, eps_l2=args.eps_l2
            )
            times.append(t_idx * dt)
            metrics_list.append(m)

        if not times:
            continue

        # Panel (0,0): Integrated Tail Area (Log scale)
        axes[0, 0].plot(times, [m["area_l1"] for m in metrics_list], ls=ls, marker=marker,
                        label=rf"$L_1$ Area (C={continuous})")
        axes[0, 0].plot(times, [m["area_l2"] for m in metrics_list], ls=ls, marker=marker,
                        label=rf"$L_2$ Area (C={continuous})")

        # Panel (0,1): Fraction of Coefficients Needed for Error <= eps
        axes[0, 1].plot(times, [m["n_eps_l1"] for m in metrics_list], ls=ls, marker=marker,
                        label=rf"$R_{{L_1}} \leq {args.eps_l1:g}$ (C={continuous})")
        axes[0, 1].plot(times, [m["n_eps_l2"] for m in metrics_list], ls=ls, marker=marker,
                        label=rf"$R_{{L_2}} \leq {args.eps_l2:g}$ (C={continuous})")

        # Panel (1,0): Normalized WHT Spectral Entropy (Order-independent if C=False/True)
        if not continuous:
            axes[1, 0].plot(times, [m["S_l1"] for m in metrics_list], '-o', color='tab:purple',
                            label=r"$S_{\mathrm{WHT}}^{(L_1)} / \ln N_{\mathrm{tot}}$")
            axes[1, 0].plot(times, [m["S_l2"] for m in metrics_list], '-s', color='tab:red',
                            label=r"$S_{\mathrm{WHT}}^{(L_2)} / \ln N_{\mathrm{tot}}$")

        # Panel (1,1): Bulk Exponential Decay Slope (Decades across spectrum)
        axes[1, 1].plot(times, [m["decay_rate_l1"] for m in metrics_list], ls=ls, marker=marker,
                        label=f"Bulk Decay Slope $\\gamma_{{L_1}}$ (C={continuous})")

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"Normalized Integrated Tail Area $\mathcal{A}$")
    axes[0, 0].set_title("1. Integrated Tail Remainder Area")
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_ylabel(r"Fraction of Basis $N_\epsilon / N_{\mathrm{tot}}$")
    axes[0, 1].set_title(r"2. $\epsilon$-Truncation Rank (Compressibility)")
    axes[0, 1].grid(True, which="both", ls="--", alpha=0.4)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel("Time $t$")
    axes[1, 0].set_ylabel(r"Normalized WHT Entropy $S_{\mathrm{WHT}} / \ln N_{\mathrm{tot}}$")
    axes[1, 0].set_title("3. WHT Spectral Entropy (Order-Invariant)")
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.4)
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].set_xlabel("Time $t$")
    axes[1, 1].set_ylabel(r"Bulk Decay Rate $-\Delta \log_{10}(R_{L_1}) / \Delta(N/N_{\mathrm{tot}})$")
    axes[1, 1].set_title("4. Bulk Exponential Decay Rate (Higher = Faster Convergence)")
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.4)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"WHT Wavefunction Complexity vs. Time (L={L}, gi={gi:.2f} -> gf={gf:.2f})", fontsize=14)
    plt.tight_layout()

    out_plot = os.path.join(output_dir, f"wht_complexity_metrics_L{L}_gi{gi:.2f}_gf{gf:.2f}.png")
    plt.savefig(out_plot, dpi=300)
    print(f"Saved complexity metrics plot to {out_plot}")
    plt.show()
