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
        description="Plot sorted WHT coefficients in descending order for k = 2, 4, 6, 8, 10."
    )
    parser.add_argument("--L", type=int, default=14, help="System size L (must be >= 10)")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument(
        "--mode", type=str, default="abs", choices=["abs", "sq", "norm_sq"],
        help="'abs': |c_A|, 'sq': c_A^2, 'norm_sq': c_A^2 / sum(c^2)"
    )

    args = parser.parse_args()

    L = args.L
    gi = args.gi
    gf = args.gf
    angle = args.angle
    t_start = args.t_start
    dt = args.dt
    Ndt = args.Ndt
    nstates = 1 << L

    target_orders = [2, 4, 6, 8, 10]
    if L < max(target_orders):
        raise ValueError(f"System size L={L} must be at least {max(target_orders)} to include all target orders.")

    pbc = True
    add = "_pbc" if pbc else ""
    if angle != 0.0:
        add += f"_{angle}"

    output_dir = "wavefunctions_data" + add

    wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    wf_file = os.path.join(output_dir, wf_base)
    if not os.path.exists(wf_file):
        raise FileNotFoundError(f"Missing wavefunction file: {wf_file}")

    psi_history = np.lib.format.open_memmap(
        wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
    )

    t_idx_list = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    times = t_idx_list * dt
    num_times = len(t_idx_list)

    # Precompute index masks for each order k in [2, 4, 6, 8, 10]
    order = get_hamming_weights(L)
    order_indices = {k: np.flatnonzero(order == k) for k in target_orders}

    # Storage dictionary: sorted_coeffs[k] has shape (num_times, binom(L, k))
    sorted_coeffs = {
        k: np.zeros((num_times, len(order_indices[k])), dtype=np.float64)
        for k in target_orders
    }

    # 1. Compute WHT for each time step and sort coefficients per order k in descending order
    print(f"Processing L = {L} for orders {target_orders}...")
    for c_idx, t_idx in enumerate(t_idx_list):
        psi_t = np.abs(psi_history[t_idx])
        psi_t /= np.sqrt(np.sum(psi_t ** 2))
        wht_log_psi = Fast_Hadamard(np.log(psi_t)).real
        wht_log_psi[0] = 0

        all_weight = np.sum(wht_log_psi ** 2)

        for k in target_orders:
            coeffs_k = np.abs(wht_log_psi[order_indices[k]])
            if args.mode == "sq":
                coeffs_k = coeffs_k ** 2
            elif args.mode == "norm_sq":
                coeffs_k = (coeffs_k ** 2) / all_weight

            # Sort in descending order
            sorted_coeffs[k][c_idx] = np.sort(coeffs_k)[::-1]

    # 2. Plotting Setup (2x3 grid: 5 subplots for k=2,4,6,8,10 + 1 panel for legend)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
    })

    cmap = cm.magma
    colors = cmap(np.linspace(0.12, 0.88, num_times))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    fig.suptitle(
        rf"Sorted WHT Coefficients by Order $k$ ($L={L},\; g_i={gi:.2f},\; g_f={gf:.2f}$)",
        fontsize=13, weight="bold", y=0.98
    )

    axes_flat = axes.flat

    if args.mode == "abs":
        ylabel_str = r"$|\hat{F}(A)|$"
    elif args.mode == "sq":
        ylabel_str = r"$\hat{F}(A)^2$"
    else:
        ylabel_str = r"$\hat{F}(A)^2 / \omega_{\mathrm{co}}$"

    for k_idx, k_val in enumerate(target_orders):
        ax = axes_flat[k_idx]
        num_terms_k = len(order_indices[k_val])
        ranks = np.arange(1, num_terms_k + 1)

        for c_idx, t_val in enumerate(times):
            ax.plot(
                ranks,
                sorted_coeffs[k_val][c_idx],
                ls="-",
                color=colors[c_idx],
                label=rf"$t = {t_val:.2f}$",
                alpha=0.9
            )

        ax.set_title(
            rf"Order $k = {k_val}$ ($\binom{{{L}}}{{{k_val}}} = {num_terms_k}$ coeffs)",
            pad=8
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Coefficient Rank (descending)")
        if k_idx % 3 == 0:
            ax.set_ylabel(ylabel_str)

        ax.grid(True, which="major", ls=":", lw=0.8, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)

    # 3. Use the 6th panel (axes_flat[5]) for the global time legend
    legend_ax = axes_flat[5]
    legend_ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_ax.legend(
        handles, labels,
        loc="center",
        ncol=2,
        frameon=True,
        title=r"$\bf{Time\ Evolution\ (t)}$",
        title_fontsize=10.5,
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir, f"wht_sorted_coeffs_k2_to_10_L{L}_{args.mode}_gi{gi:.2f}_gf{gf:.2f}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved sorted coefficients plot to {plot_path}")
    plt.show()
