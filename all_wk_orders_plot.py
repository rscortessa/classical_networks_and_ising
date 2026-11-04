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
    parser = argparse.ArgumentParser(description="Plot w_orders vs System Size L across subplots for each time t.")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)

    args = parser.parse_args()

    L_list = np.array([8, 10, 12, 14, 16, 18, 20, 22])
    num_L = len(L_list)
    max_L = np.max(L_list)
    all_list_orders = np.arange(2, max_L + 1, 2)  # [2, 4, 6, ..., 22]
    num_orders = len(all_list_orders)

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

    # 3D array: (num_L, num_times, num_orders) initialized with NaN for orders k > L
    w_orders_all = np.full((num_L, num_times, num_orders), np.nan)

    # 1. Load each L once and populate w_orders_all
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
        num_valid_orders = len(list_orders)

        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            wht_log_psi = Fast_Hadamard(np.log(psi_t)).real
            wht_log_psi[0] = 0

            all_weight = np.sum(wht_log_psi ** 2)
            
            # Bin squared WHT coefficients by order k
            weights_by_k = np.bincount(order, weights=wht_log_psi ** 2, minlength=L + 1)
            w_orders = weights_by_k[list_orders] 

            w_orders_all[l_idx, c_idx, :num_valid_orders] = w_orders

    # 2. Styling and Subplot Setup (3x4 grid for 11 time steps + 1 legend panel)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
    })

    cmap = cm.viridis
    colors = cmap(np.linspace(0.05, 0.95, num_orders))

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey=True)
    fig.suptitle(rf"WHT Order Weights $P(k)$ vs System Size $L$ ($g_i={gi:.2f},\; g_f={gf:.2f}$)", 
                 fontsize=14, weight="bold", y=0.98)

    axes_flat = axes.flat

    # 3. Plot each time step in its own subplot
    for c_idx, t_val in enumerate(times):
        ax = axes_flat[c_idx]
        
        for k_idx, k_val in enumerate(all_list_orders):
            ax.plot(
                L_list, 
                w_orders_all[:, c_idx, k_idx], 
                ls="-", 
                marker="o", 
                color=colors[k_idx], 
                label=rf"$k = {k_val}$", 
                alpha=0.9
            )

        ax.set_title(rf"$t = {t_val:.2f}$", pad=6)
        ax.set_yscale("log")
        ax.set_xticks(L_list)
        ax.set_ylim([1e-10, 2.0])
        ax.grid(True, which="major", ls=":", lw=0.8, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)

        # Labels on outer edges
        if c_idx % 4 == 0:
            ax.set_ylabel(r"$P(k)$")
        if c_idx >= 7:
            ax.set_xlabel(r"System Size ($L$)")
            ax.tick_params(labelbottom=True)

    # 4. Use the 12th panel (axes_flat[11]) for the global k-order legend
    legend_ax = axes_flat[11]
    legend_ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_ax.legend(
        handles, labels, 
        loc="center", 
        ncol=2, 
        frameon=True,
        title=r"$\bf{Order\ (k)}$", 
        title_fontsize=10.5, 
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"wht_w_orders_vs_L_by_time_gi{gi:.2f}_gf{gf:.2f}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved w_orders vs L subplot grid to {plot_path}")
    plt.show()
