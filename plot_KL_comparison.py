#!/usr/bin/env python
# coding: utf-8
"""
plot_KL_comparison.py — Compares KL divergence trajectories between the usual 
greedy ordering and the natural (linear-KL) ordering across multiple time slices.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def main():
    parser = argparse.ArgumentParser(description="Plot KL vs Number of Removed Coefficients (Usual vs Natural)")
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    
    args = parser.parse_args()
    
    pbc = True
    add = "_pbc" if pbc else ""
    if args.angle != 0.0:
        add += f"_{args.angle}"
        
    output_dir = "wavefunctions_data" + add
    t = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    L = args.L
    Ntimes = 100
    continuous_list = [True, False]

    n_values = np.arange(0, Ntimes + 1, 10)
    cmap = plt.get_cmap('viridis', len(n_values))
    norm = mcolors.BoundaryNorm(np.arange(-5, Ntimes + 6, 10) / Ntimes, cmap.N)

    # 1. Load the groups to figure out the size of each group
    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{args.angle}.npy")
    if not os.path.exists(global_filename):
        raise FileNotFoundError(f"Cannot find {global_filename}. Make sure groups are generated.")
    
    dense_groups_global = np.load(global_filename)
    rep_to_size = {}
    for g in dense_groups_global:
        valid_elements = g[g != -1]
        if len(valid_elements) > 0:
            rep = valid_elements[0]
            rep_to_size[rep] = len(valid_elements)

    for continuous in continuous_list:
        # Create a figure with 2 subplots side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Datasets to plot: (subplot_index, file_prefix, title_label)
        configs = [
            (0, "kl", "Usual Greedy Ordering"),
            (1, "kl_nat", "Natural (Linear-KL) Ordering")
        ]

        for ax_idx, prefix, title_text in configs:
            ax = axes[ax_idx]
            
            for t_idx in t:
                if prefix == "kl":
                    kl_file = os.path.join(output_dir, f"kl_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
                    sort_file = os.path.join(output_dir, f"sort_KL_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
                else:
                    kl_file = os.path.join(output_dir, f"kl_nat_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
                    sort_file = os.path.join(output_dir, f"sort_KL_nat_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")

                if not os.path.exists(kl_file) or not os.path.exists(sort_file):
                    print(f"Warning: Missing data files for {prefix} at t_idx={t_idx}, Continuous={continuous}. Skipping...")
                    continue
                
                kls = np.load(kl_file)
                order = np.load(sort_file)
                
                group_sizes = [1] + [rep_to_size[rep] for rep in order]
                x_cumulative_removed = np.cumsum(group_sizes)[:-1]
                
                ax.plot(
                    x_cumulative_removed, kls, 
                    color=cmap(norm(t_idx / Ntimes)), 
                    marker='.', markersize=4, 
                    label=f'{t_idx / Ntimes:.2f}' if ax_idx == 0 else None
                )

            ax.set_xlabel("$n_c$ (Coefficients Removed)", fontsize=12)
            if ax_idx == 0:
                ax.set_ylabel("KL Divergence", fontsize=12)
            ax.set_title(title_text, fontsize=14)
            ax.set_yscale("log")
            ax.set_ylim([1e-6, max(10.0, ax.get_ylim()[1])])
            ax.grid(True, which="both", ls="--", alpha=0.5)

        # Shared colorbar for time
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=n_values / Ntimes)
        cbar.set_label("Time $t$", fontsize=12)

        fig.suptitle(f"KL Divergence Comparison ($L={L}, g_i={args.gi}, g_f={args.gf}, \text{{Continuous}}={continuous}$)", fontsize=16)
        
        # Save the comparative plot
        plot_path = os.path.join(output_dir, f"kl_comparison_L{args.L}_gi{args.gi}_gf_{args.gf}_continuous{continuous}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved successfully to {plot_path}")

if __name__ == "__main__":
    main()
