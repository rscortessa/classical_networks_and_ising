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


def compute_group_ratio(wht_log_psi,rep_to_size):
    """
    Computes the ratio between two successive groups in `sort_order`:
        new_array[i] = (WHT_log_psi[rep_{i+1}].real / WHT_log_psi[rep_i].real) * (n_{i+1} / n_i)
    """
    eps=1e-8
    # Extract the WHT coefficient for each group in the sorted sequence
    
    # Extract the number of terms (multiplicity) in each group
    print(rep_to_size.keys())
    reps = np.array(list(rep_to_size.keys()),dtype=int)
    n_terms = np.array(list(rep_to_size.values()))
    wht_sorted = np.abs(wht_log_psi[reps])

    # Compute the weighted group contribution: WHT_coeff * n_terms
    idx_terms =  np.argsort(wht_sorted * n_terms)[::-1]
    weighted_terms = wht_sorted[idx_terms] * n_terms[idx_terms]
    n_terms = n_terms[idx_terms]

    mask = weighted_terms > 1e-8
    
    num = weighted_terms[mask][1:]
    den = weighted_terms[mask][:-1]    
    ratio_array = num/ den
    
    return ratio_array, n_terms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WHT convergence ratio across t_idx.")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--erasure-order", action="store_true",
                        help="If set, reverses sort file to go from least-important (first erased) to most-important.")
    parser.add_argument("--abs-ratio", action="store_true",
                        help="If set, plots |new_array[i]| on a log scale to inspect decay rates.")

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
    Ls = [8,10,12,14,16,18,20,22]
    t_idx_list = [0]

    # Create a 2-panel plot (one for Continuous=False, one for Continuous=True)
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
    colors = cm.viridis(np.arange(0,len(Ls)))
    ax = axes
    continuous= True
    
    for idx_L,L in enumerate(Ls):
        nstates = 1 << L
        
        # 1. Load symmetry groups and build lookup: representative -> number of terms in group
        global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")
        if not os.path.exists(global_filename):
            raise FileNotFoundError(f"Missing {global_filename}. Run generate_pbc_groups.py first.")

        dense_groups_global = np.load(global_filename)
        rep_to_size = {}
        for g in dense_groups_global:
            valid_g = g[g != -1]
            if len(valid_g) > 0:
                rep_to_size[int(valid_g[0])] = len(valid_g)

            
        # 2. Open wavefunction history memmap
        wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
        wf_file = os.path.join(output_dir, wf_base)
        if not os.path.exists(wf_file):
            raise FileNotFoundError(f"Missing wavefunction file: {wf_file}")

        psi_history = np.lib.format.open_memmap(
            wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
        )

    

        for c_idx, t_idx in enumerate(t_idx_list):
        
            # Compute WHT of log(|psi(t_idx)|)
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            wht_log_psi = Fast_Hadamard(np.log(psi_t))

            # Compute successive group ratio array
            new_array, _ = compute_group_ratio(wht_log_psi,rep_to_size)
            x_group_idx = np.arange(len(new_array))

            y_plot = np.abs(new_array) if args.abs_ratio else new_array
            
            ax.plot(
                x_group_idx, y_plot,
                color=colors[idx_L], marker='.', markersize=3, linewidth=1.0, alpha=0.85,
                label=rf"$L={L}$"
            )

    # Reference horizontal lines at ratio = +1 (and -1 if signed)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7, label="Ratio = 1")
    if not args.abs_ratio:
        ax.axhline(-1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.4)

    ax.set_xlabel("Group Index $i$")
    if args.abs_ratio:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(-0.2,10)
    
    fig.suptitle(rf"Ratio "+r"$|\frac{C_{i+1}}{C_{i}}|\cdot \frac{n_{i+1}}{n_{i}}$"+"\n"+rf" $( g_i={gi:.2f}, g_f={gf:.2f},t={t_idx*0.01:.2f})$", fontsize=13)
    plt.tight_layout()

    suffix = "_abs" if args.abs_ratio else "_signed"
    order_tag = "_erasure" if args.erasure_order else "_importance"
    plot_path = os.path.join(output_dir, f"wht_ratio_different_L_gi{gi:.2f}_gf{gf:.2f}{order_tag}{suffix}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved convergence ratio plot to {plot_path}")
    plt.show()
