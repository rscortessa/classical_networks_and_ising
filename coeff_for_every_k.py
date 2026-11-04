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


def get_first_coeff_info(k):
    """
    Returns the array index and monomial site indices for the first state
    matching `order == k` (the lowest integer with exactly k set bits).
    """
    # The smallest integer with k bits set to 1 is (2^k - 1)
    first_idx = (1 << k) - 1
    # Active bit positions (0-indexed from the fastest-varying spin)
    site_indices = [i for i in range(k) if (first_idx >> i) & 1]
    monomial_str = "".join([f"s_{{{i}}}" for i in site_indices])
    indices_str = ", ".join(map(str, site_indices))
    return first_idx, indices_str, monomial_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the first WHT coefficient of the first 4 orders vs L for different times."
    )
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument(
        "--mode", type=str, default="abs", choices=["abs", "raw", "norm_sq"],
        help="'abs': |c_A|, 'raw': signed c_A, 'norm_sq': c_A^2 / sum(c^2)"
    )

    args = parser.parse_args()

    L_list = np.array([8, 10, 12, 14, 16, 18, 20, 22])
    num_L = len(L_list)

    # First 4 non-zero even orders
    target_orders = [2, 4, 6, 8]
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

    # Precompute the first mask index for each of the 4 orders: idx = 2^k - 1
    first_indices = [(1 << k) - 1 for k in target_orders]

    # Storage array: shape (num_L, num_times, num_orders)
    first_coeffs = np.full((num_L, num_times, num_orders), np.nan)

    # 1. Load wavefunctions for each L and extract the first coefficient per order
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

        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            wht_log_psi = Fast_Hadamard(np.log(psi_t)).real
            wht_log_psi[0] = 0

            all_weight = np.sum(wht_log_psi ** 2)

            for k_idx, mask_first_idx in enumerate(first_indices):
                val = wht_log_psi[mask_first_idx]
                if args.mode == "abs":
                    first_coeffs[l_idx, c_idx, k_idx] = np.abs(val)
                elif args.mode == "raw":
                    first_coeffs[l_idx, c_idx, k_idx] = val
                elif args.mode == "norm_sq":
                    first_coeffs[l_idx, c_idx, k_idx] = (val ** 2) /all_weight

    # 2. Plotting Setup (2x2 grid for the first 4 orders: k = 2, 4, 6, 8)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10.5,
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
        rf"First WHT Monomial Coefficient vs System Size $L$ ($g_i={gi:.2f},\; g_f={gf:.2f}$)",
        fontsize=13, weight="bold", y=0.98
    )

    axes_flat = axes.flat

    for k_idx, k_val in enumerate(target_orders):
        ax = axes_flat[k_idx]
        first_idx, indices_str, monomial_str = get_first_coeff_info(k_val)

        for c_idx, t_val in enumerate(times):
            ax.plot(
                L_list,
                first_coeffs[:, c_idx, k_idx],
                ls="-",
                marker="o",
                color=colors[c_idx],
                label=rf"$t = {t_val:.2f}$",
                alpha=0.9
            )

        # Title showing Order k, WHT array index, active spin indices A, and monomial
        ax.set_title(
            rf"Order $k={k_val}$ (WHT idx $={first_idx}$): "
            rf"$A = \{{{indices_str}\}}$ ($\chi_A = {monomial_str}$)",
            pad=8
        )

        if args.mode in ["abs", "norm_sq"]:
            ax.set_yscale("log")

        if args.mode == "abs":
            ax.set_ylabel(rf"$|\hat{{F}}(A_{{{k_val}}})|$")
        elif args.mode == "raw":
            ax.set_ylabel(rf"$\hat{{F}}(A_{{{k_val}}})$")
        else:
            ax.set_ylabel(rf"$\hat{{F}}(A_{{{k_val}}})^2 / \omega_{{\mathrm{{co}}}}$")

        if k_idx >= 2:
            ax.set_xlabel(r"System Size ($L$)")

        ax.set_xticks(L_list)
        ax.grid(True, which="major", ls=":", lw=0.8, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize=7.5, ncol=2, loc="best", frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir, f"wht_first_coeff_k2_4_6_8_vs_L_{args.mode}_gi{gi:.2f}_gf{gf:.2f}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved first-coefficient plot to {plot_path}")
    plt.show()
