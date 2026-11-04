#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.special as sc


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
        description="Plot the Participation Ratio PR_k vs System Size L for different times."
    )
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument(
        "--all_orders", action="store_true",
        help="If set, plots a 3x4 grid of all even orders k=2..22 instead of just k=2,4,6,8."
    )

    args = parser.parse_args()

    L_list = np.array([8, 10, 12, 14, 16, 18, 20, 22])
    num_L = len(L_list)

    if args.all_orders:
        target_orders = np.arange(2, np.max(L_list) + 1, 2)  # [2, 4, ..., 22]
    else:
        target_orders = np.array([2, 4, 6, 8])               # First 4 even orders
    num_orders = len(target_orders)

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

    # Storage array: shape (num_L, num_times, num_orders)
    pr_k_all = np.full((num_L, num_times, num_orders), np.nan)

    # 1. Load wavefunctions for each L and compute PR_k via vectorized bincount
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
        valid_mask = target_orders <= L
        valid_orders = target_orders[valid_mask]

        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            wht_log_psi = Fast_Hadamard(np.log(psi_t)).real
            wht_log_psi[0] = 0

            # Compute c_A^2 and c_A^4
            c2 = wht_log_psi ** 2
            c4 = c2 ** 2

            # Sum c_A^2 and c_A^4 within each order k using np.bincount
            sum_c2_by_k = np.bincount(order, weights=c2, minlength=L + 1)
            sum_c4_by_k = np.bincount(order, weights=c4, minlength=L + 1)

            num = sum_c2_by_k[valid_orders] ** 2
            den = sum_c4_by_k[valid_orders]

            # Avoid division by zero if a sector has zero weight
            pr_vals = np.where(den > 0, num / den, np.nan)
            pr_k_all[l_idx, c_idx, valid_mask] = pr_vals

    # 2. Plotting Setup
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

    if args.all_orders:
        fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    fig.suptitle(
        rf"Sector Participation Ratio $\mathrm{{PR}}_k$ vs System Size $L$ ($g_i={gi:.2f},\; g_f={gf:.2f}$)",
        fontsize=13, weight="bold", y=0.98
    )

    axes_flat = axes.flat

    for k_idx, k_val in enumerate(target_orders):
        ax = axes_flat[k_idx]

        # Plot time curves
        for c_idx, t_val in enumerate(times):
            ax.plot(
                L_list,
                pr_k_all[:, c_idx, k_idx],
                ls="-",
                marker="o",
                color=colors[c_idx],
                label=rf"$t = {t_val:.2f}$",
                alpha=0.9
            )

        # Plot combinatorial upper bound binom(L, k) for reference
        valid_L = L_list[L_list >= k_val]
        ax.plot(
            valid_L,
            sc.binom(valid_L, k_val),
            ls="--",
            color="gray",
            lw=1.2,
            alpha=0.7,
            label=rf"$\binom{{L}}{{{k_val}}}$ (max)"
        )

        ax.set_title(
            rf"Order $k = {k_val}$: $\mathrm{{PR}}_{{{k_val}}} = \frac{{(\sum_{{|A|={k_val}}} c_A^2)^2}}{{\sum_{{|A|={k_val}}} c_A^4}}$",
            pad=8
        )
        ax.set_yscale("log")
        ax.set_ylabel(rf"$\mathrm{{PR}}_{{{k_val}}}$")
        ax.set_xticks(L_list)
        ax.grid(True, which="major", ls=":", lw=0.8, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)

        if not args.all_orders:
            if k_idx >= 2:
                ax.set_xlabel(r"System Size ($L$)")
            ax.legend(fontsize=7.5, ncol=2, loc="best", frameon=True)
        else:
            if k_idx >= 7:
                ax.set_xlabel(r"System Size ($L$)")
                ax.tick_params(labelbottom=True)

    if args.all_orders:
        legend_ax = axes_flat[11]
        legend_ax.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        legend_ax.legend(
            handles, labels, loc="center", ncol=2, frameon=True,
            title=r"$\bf{Time\ (t)}$", title_fontsize=10, fontsize=8.5
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    suffix = "all_orders" if args.all_orders else "k2_4_6_8"
    plot_path = os.path.join(
        output_dir, f"wht_PR_k_{suffix}_vs_L_gi{gi:.2f}_gf{gf:.2f}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved PR_k vs L plot to {plot_path}")
    plt.show()
