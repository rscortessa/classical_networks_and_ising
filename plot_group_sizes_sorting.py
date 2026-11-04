#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Restrict thread usage
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx", type=int, default=10)
    args = parser.parse_args()

    L, gi, gf, angle, t_start, t_idx = args.L, args.gi, args.gf, args.angle, args.t_start, args.t_idx
    nstates = 1 << L
    pbc = True
    add = "_pbc" + (f"_{angle}" if angle != 0.0 else "")
    output_dir = "wavefunctions_data" + add

    # Load symmetry groups and map group representative to its size
    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")
    dense_groups_global = np.load(global_filename)

    rep_to_size = {}
    for g in dense_groups_global:
        valid_g = g[g != -1]
        if len(valid_g) > 0:
            rep_to_size[int(valid_g[0])] = len(valid_g)

    continuous_list = [False, True]

    for continuous in continuous_list:
        # Load the saved sort file directly from the output directory
        sort_filename = os.path.join(
            output_dir, 
            f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt0.01t{t_idx}Nt100C{continuous}.npy"
        )
        
        if not os.path.exists(sort_filename):
            print(f"Sorting file not found: {sort_filename}")
            continue

        # Load sorting order (saved as order_exact[::-1], so reverse back to chronological selection order)
        order_exact = np.load(sort_filename)[::-1]

        # Get group sizes in the order they were selected
        group_sizes = [rep_to_size[int(rep)] for rep in order_exact]
        
        #group_sizes = np.cumsum(np.array(group_sizes))
        
        group_ranks = np.arange(1, len(group_sizes) + 1)

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(group_ranks, group_sizes, marker='.', linestyle='-', color='tab:blue')
        plt.title(f"Coefficients per Group by Selection Order\n(L={L}, t={t_idx*0.01:.2f}, Continuous={continuous})")
        plt.xlabel("Group Selection Rank (1st, 2nd, 3rd, ... group chosen)")
        plt.ylabel("Number of Coefficients in Group")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plot_filename = f"group_sizes_sorting_L{L}_t{t_idx}_C{continuous}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path}")
