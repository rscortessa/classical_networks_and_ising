#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def Fast_Hadamard(a):
    ordering = 1.0    
    a = a.copy()
    nstates = a.shape[-1]
    h = 1
    
    while h < nstates:
        a2 = a.reshape(a.shape[:-1] + (nstates // (2 * h), 2, h))
        x = a2[..., 0, :]
        y = a2[..., 1, :]

        t = x.copy()
        a2[..., 0, :] = 0.5 * (t + y)
        a2[..., 1, :] = 0.5 * (t - y) * ordering

        h *= 2

    return a


def get_hamming_weights(L):
    """
    Fast, memory-efficient computation of the order (number of +1 spins / set bits)
    for all 2^L states without allocating a (2^L, L) array via NetKet.
    """
    weights = np.zeros(1 << L, dtype=np.uint8)
    for i in range(L):
        step = 1 << i
        weights[step:2 * step] = weights[:step] + 1
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Average and Total WHT Participation Ratio (Effective Parameters) vs System Size L."
    )
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)

    args = parser.parse_args()

    L_list = np.array([8, 10, 12, 14, 16, 18, 20, 22])
    num_L = len(L_list)

    gi = args.gi
    gf = args.gf
    angle = args.angle
    t_start = args.t_start
    dt = args.dt
    Ndt = args.Ndt

    pbc = True
    add = "_pbc" if pbc else ""
    if angle != 0.0:
        add += f"_{angle}"

    output_dir = "wavefunctions_data" + add

    t_idx_list = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    times = t_idx_list * dt
    num_times = len(t_idx_list)

    # 2D arrays of shape (num_L, num_times) for the effective parameter metrics
    pr_weighted_avg = np.full((num_L, num_times), np.nan)   # sum_k P(k) * PR_k
    pr_global_total = np.full((num_L, num_times), np.nan)   # (sum_A c_A^2)^2 / sum_A c_A^4
    pr_unweighted_avg = np.full((num_L, num_times), np.nan) # (1 / N_k) * sum_k PR_k
    pr_sector_sum = np.full((num_L, num_times), np.nan)     # sum_k PR_k

    # 1. Load wavefunctions for each L once and compute PR quantities
    for l_idx, L in enumerate(L_list):
        nstates = 1 << L
        wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
        wf_file = os.path.join(output_dir, wf_base)

        if not os.path.exists(wf_file):
            print(f"[Warning] Missing wavefunction file for L={L}: {wf_file}. Skipping...")
            continue

        print(f"Processing L = {L}...")
        psi_history = np.lib.format.open_memmap(
            wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
        )

        order = get_hamming_weights(L)
        list_orders = np.arange(2, L + 1, 2)

        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            wht_log_psi = Fast_Hadamard(np.log(psi_t)).real
            wht_log_psi[0] = 0

            # Compute squared and fourth-power WHT coefficients
            c2 = wht_log_psi ** 2
            c4 = c2 ** 2

            all_weight = np.sum(c2)
            total_c4 = np.sum(c4)

            # Global Participation Ratio across all non-zero WHT coefficients
            if total_c4 > 0:
                pr_global_total[l_idx, c_idx] = (all_weight ** 2) / total_c4

            # Sector-wise sums via vectorized bincount
            sum_c2_by_k = np.bincount(order, weights=c2, minlength=L + 1)[list_orders]
            sum_c4_by_k = np.bincount(order, weights=c4, minlength=L + 1)[list_orders]

            # Relative spectral weight P(k) = W_k / all_weight
            w_orders = sum_c2_by_k / all_weight

            # Sector Participation Ratio PR_k = W_k^2 / sum_{|A|=k} c_A^4
            valid = sum_c4_by_k > 0
            pr_k = np.zeros_like(sum_c2_by_k)
            pr_k[valid] = (sum_c2_by_k[valid] ** 2) / sum_c4_by_k[valid]

            # Store summary effective-parameter metrics
            pr_weighted_avg[l_idx, c_idx] = np.sum(w_orders * pr_k)
            pr_unweighted_avg[l_idx, c_idx] = np.mean(pr_k)
            pr_sector_sum[l_idx, c_idx] = np.sum(pr_k)

    # 2. Plotting Setup (2x2 grid comparing the effective parameter definitions)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
    })

    cmap = cm.magma
    colors = cmap(np.linspace(0.12, 0.88, num_times))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    fig.suptitle(
        rf"Effective Number of WHT Parameters vs System Size $L$ ($g_i={gi:.2f},\; g_f={gf:.2f}$)",
        fontsize=13, weight="bold", y=0.98
    )

    for c_idx, t_val in enumerate(times):
        curve_kw = dict(
            ls="-", marker="o", color=colors[c_idx],
            lw=1.6, markersize=4.5, label=rf"$t = {t_val:.2f}$", alpha=0.9
        )
        axes[0, 0].plot(L_list, pr_weighted_avg[:, c_idx], **curve_kw)
        axes[0, 1].plot(L_list, pr_global_total[:, c_idx], **curve_kw)
        axes[1, 0].plot(L_list, pr_unweighted_avg[:, c_idx], **curve_kw)
        axes[1, 1].plot(L_list, pr_sector_sum[:, c_idx], **curve_kw)

    # Reference scaling lines (linear L vs exponential 2^(L-1))
    for ax in axes.flat:
        ax.plot(L_list, L_list, ls="--", color="gray", lw=1.2, alpha=0.7, label=r"$\propto L$ (linear)")

    axes[1, 0].plot(
        L_list, (2.0 ** (L_list - 1)) / (L_list / 2.0),
        ls=":", color="black", lw=1.2, alpha=0.6, label=r"$2^{L-1}/(L/2)$ (max)"
    )
    axes[1, 1].plot(
        L_list, 2.0 ** (L_list - 1) - 1,
        ls=":", color="black", lw=1.2, alpha=0.6, label=r"$2^{L-1}-1$ (max)"
    )

    # Subplot Titles and Axis Labels
    axes[0, 0].set_title(r"Weighted Average Sector PR: $\sum_k P(k)\,\mathrm{PR}_k$", pad=8)
    axes[0, 0].set_ylabel(r"$\langle \mathrm{PR}_k \rangle_{P(k)}$")

    axes[0, 1].set_title(r"Global WHT Participation Ratio: $(\sum_A c_A^2)^2 / \sum_A c_A^4$", pad=8)
    axes[0, 1].set_ylabel(r"$\mathrm{PR}_{\mathrm{total}}$")

    axes[1, 0].set_title(r"Unweighted Mean Sector PR: $\frac{2}{L}\sum_k \mathrm{PR}_k$", pad=8)
    axes[1, 0].set_ylabel(r"$\overline{\mathrm{PR}_k}$")
    axes[1, 0].set_xlabel(r"System Size ($L$)")

    axes[1, 1].set_title(r"Cumulative Sector PR: $\sum_k \mathrm{PR}_k$", pad=8)
    axes[1, 1].set_ylabel(r"$\sum_k \mathrm{PR}_k$")
    axes[1, 1].set_xlabel(r"System Size ($L$)")

    for ax in axes.flat:
        ax.set_yscale("log")
        ax.set_xticks(L_list)
        ax.grid(True, which="major", ls=":", lw=0.8, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize=7.5, ncol=2, loc="best", frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir, f"wht_effective_params_PR_vs_L_gi{gi:.2f}_gf{gf:.2f}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved effective parameter PR plot to {plot_path}")
    plt.show()
