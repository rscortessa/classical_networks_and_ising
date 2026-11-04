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
    parser = argparse.ArgumentParser(description="Plot WHT Order Quantities vs System Size L for various times t.")
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

    # Colormap for different times t
    cmap = cm.magma
    colors = cmap(np.linspace(0.12, 0.88, num_times))

    # Publication styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.autolayout": False,
    })

    # 2D arrays of shape (num_L, num_times) initialized with NaNs in case any L file is missing
    w_average = np.full((num_L, num_times), np.nan)
    w_exp = np.full((num_L, num_times), np.nan)
    w_binom = np.full((num_L, num_times), np.nan)
    w_co = np.full((num_L, num_times), np.nan)

    # 1. Outer loop over L so each large memmap file and order array is loaded only once
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
        binom_coeffs = sc.binom(L, list_orders)
        exp_coeffs = np.exp(list_orders)

        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            wht_log_psi = Fast_Hadamard(np.log(psi_t)).real
            wht_log_psi[0] = 0

            #all_weight = np.sum(wht_log_psi ** 2)
            
            # Bin squared WHT coefficients by order k
            weights_by_k = np.bincount(order, weights=wht_log_psi ** 2, minlength=L + 1)

            w_orders = weights_by_k[list_orders]/binom_coeffs
            all_weight = np.sum(w_orders)
            w_orders /= all_weight
            

            w_average[l_idx, c_idx] = np.sum(w_orders * list_orders * binom_coeffs)/2**L
            w_exp[l_idx, c_idx] = np.log(np.sum(w_orders * exp_coeffs * binom_coeffs))/2**L
            w_binom[l_idx, c_idx] = np.sum(binom_coeffs * w_orders)/2**L
            w_co[l_idx, c_idx] = all_weight

    # 2. Plot quantities vs System Size L for each time t
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True)
    fig.suptitle(rf"WHT Order Scaling vs System Size $L$ ($g_i={gi:.2f},\; g_f={gf:.2f}$)", 
                 fontsize=13, weight="bold", y=0.98)

    for c_idx, t_val in enumerate(times):
        curve_kw = dict(ls="-", marker="o", color=colors[c_idx], lw=1.6, 
                        markersize=4.5, label=f"$t = {t_val:.2f}$", alpha=0.9)
        
        axes[0, 0].plot(L_list, w_average[:, c_idx], **curve_kw)
        axes[0, 1].plot(L_list, w_exp[:, c_idx], **curve_kw)
        axes[1, 0].plot(L_list, w_binom[:, c_idx], **curve_kw)
        axes[1, 1].plot(L_list, w_co[:, c_idx], **curve_kw)

    # Titles and Labels
    axes[0, 0].set_title(r"Average Order $\langle k \rangle$", pad=8)
    axes[0, 0].set_ylabel(r"$\langle k \rangle$")

    axes[0, 1].set_title(r"Exponential Moment $\log \langle e^k \rangle$", pad=8)
    axes[0, 1].set_ylabel(r"$\log \langle e^k \rangle$")

    axes[1, 0].set_title(r"Binomial Moment $\langle \binom{L}{k} \rangle$", pad=8)
    axes[1, 0].set_ylabel(r"$\langle \binom{L}{k} \rangle$")
    axes[1, 0].set_xlabel(r"System Size ($L$)")
    axes[1, 0].set_yscale("log")  # Optional: helpful since <binom(L,k)> grows rapidly with L

    axes[1, 1].set_title(r"Total Weight $\omega_{\mathrm{co}}$", pad=8)
    axes[1, 1].set_ylabel(r"$\omega_{\mathrm{co}}$")
    axes[1, 1].set_xlabel(r"System Size ($L$)")

    # Grid, Ticks, and Legend aesthetics
    for ax in axes.flat:
        ax.set_xticks(L_list)
        ax.grid(True, which="major", ls=":", lw=0.8, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize=7.5, ncol=2, loc="best", frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"wht_order_scaling_vs_L_gi{gi:.2f}_gf{gf:.2f}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved scaling vs L plot to {plot_path}")
    plt.show()
