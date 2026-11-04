#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Fidelity F[i] vs System Size L for different times")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)

    N_fs=5
    
    args = parser.parse_args()
    
    pbc = True
    add = "_pbc" if pbc else ""
    if args.angle != 0.0:
        add += f"_{args.angle}"
        
    output_dir = "wavefunctions_data" + add
    Ls = [8, 10, 12, 14, 16, 18]
    t_idxs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    continuous_list = [True, False]

    for continuous in continuous_list:
        for t_idx in t_idxs:
            t_val = t_idx * args.dt
            
            Ls_used = []
            F0_vals = []
            F1_vals = []
            
            for L in Ls:
                inf_file = os.path.join(output_dir, f"sort_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
                
                if not os.path.exists(inf_file):
                    continue
                
                infs = np.load(inf_file)
                if len(infs) < 2:
                    continue
                
                # Compute Fidelity F = 1 - infs
                F = infs
                Ls_used.append(L)
                F0_vals.append(np.sum(F[0]))
                F1_vals.append(F[1])
            
            if len(Ls_used) == 0:
                continue
                
            # Create a plot of F[0] and F[1] vs L for this specific time
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(Ls_used, F0_vals, marker='o', linestyle='-', color='teal', label=r'$F[0]$')
            ax.plot(Ls_used, F1_vals, marker='s', linestyle='--', color='darkorange', label=r'$F[1]$')
            
            ax.set_xlabel(r"System Size $L$", fontsize=12)
            ax.set_ylabel(r"Fidelity $F[i]$", fontsize=12)
            ax.set_title(f"Fidelity vs System Size ($t={t_val:.2f}, g_i={args.gi}, g_f={args.gf}$, C={continuous})", fontsize=13)
            #ax.set_yscale("log")  # Often helpful since fidelity/infidelity spans orders of magnitude
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(fontsize=11)
            plt.tight_layout()
            
            plot_filename = os.path.join(output_dir, f"sorting_vs_t{t_idx}_continuous{continuous}.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close(fig)
            
        print(f"Completed Fidelity vs L plots for Continuous={continuous}")

if __name__ == "__main__":
    main()
