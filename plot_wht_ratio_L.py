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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WHT L1 sum vs system size L for different times.")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)

    args = parser.parse_args()

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

    Ls = [8, 10, 12, 14, 16, 18, 20, 22]
    t_idx_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # weights[idx_L, c_idx] stores the quantity for system size Ls[idx_L] at time t_idx_list[c_idx]
    weights = np.zeros((len(Ls), len(t_idx_list)))

    for idx_L, L in enumerate(Ls):
        nstates = 1 << L  # Update Hilbert space dimension 2^L for each system size

        wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
        wf_file = os.path.join(output_dir, wf_base)
        if not os.path.exists(wf_file):
            raise FileNotFoundError(f"Missing wavefunction file: {wf_file}")

        psi_history = np.lib.format.open_memmap(
            wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
        )

        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            prob_t = psi_t ** 2
            WHT = Fast_Hadamard(prob_t)
            weights[idx_L, c_idx] = np.sum(np.abs(WHT[1:])) * 2**L

    # Single plot: Quantity vs System Size L for different times t
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.viridis(np.linspace(0, 1, len(t_idx_list)))

    for c_idx, t_idx in enumerate(t_idx_list):
        t_val = t_start + t_idx * dt
        ax.plot(
            Ls,
            weights[:, c_idx],
            marker="o",
             markersize=4,
            color=colors[c_idx],
            lw=1.6,
            label=rf"$t = {t_val:.2f}$",
        )

    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\sum_{A \neq \emptyset} |\hat{P}(A)|$")
    ax.set_title(rf"System-Size Scaling ($g_i={gi},\; g_f={gf}$)")
    ax.legend(fontsize=8, ncol=2, title="Time")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"wht_ratio_vs_L_gi{gi:.2f}_gf{gf:.2f}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved system-size dependence plot to {plot_path}")
    plt.show()
