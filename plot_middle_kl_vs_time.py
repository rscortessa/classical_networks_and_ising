#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot KL at the middle point vs time for different system sizes")
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
    Ls = [10, 12, 14, 16, 18]
    t_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    continuous_list = [True, False]
    
    for continuous in continuous_list:
        plt.figure(figsize=(10, 6))
        
        for L in Ls:
            # 1. Load global groups for system size L
            global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{args.angle}.npy")
            if not os.path.exists(global_filename):
                print(f"Warning: Cannot find {global_filename}. Skipping L={L}...")
                continue
        
            dense_groups_global = np.load(global_filename)
            rep_to_size = {}
            for g in dense_groups_global:
                valid_elements = g[g != -1]
                if len(valid_elements) > 0:
                    rep = valid_elements[0]
                    rep_to_size[rep] = len(valid_elements)
            
            times_val = []
            mid_kl_vals = []
            
            for t_idx in t_list:
                inf_file = os.path.join(output_dir, f"kl_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
                sort_file = os.path.join(output_dir, f"sort_KL_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")

                if not os.path.exists(inf_file) or not os.path.exists(sort_file):
                    continue
                
                infs = np.real(np.load(inf_file))
                order = np.load(sort_file)
                
                group_sizes = [1] + [rep_to_size.get(rep, 1) for rep in order]
                x_cumulative_removed = np.cumsum(group_sizes)[:-1]
                fraction = x_cumulative_removed / (2**L)
                
                # Find the index closest to the middle point (fraction = 0.5)
                mid_idx = np.argmin(np.abs(fraction - 0.25))
                mid_kl_vals.append(infs[mid_idx])
                times_val.append(t_idx * args.dt)
            
            if len(times_val) > 0:
                plt.plot(times_val, mid_kl_vals, marker='o', markersize=4, label=f'L = {L}')
        
        plt.xlabel(r"Time $t$")
        plt.ylabel(r"KL at $n_c / n_{\text{states}} \approx 0.5$")
        plt.yscale("log")
        plt.title(r"Middle-Point KL vs Time ($g_i$=" + f"{args.gi}, $g_f$={args.gf}, Continuous={continuous})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"middle_kl_vs_time_gi{args.gi}_gf{args.gf}_continuous{continuous}.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved successfully to {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
