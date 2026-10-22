#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def main():
    parser = argparse.ArgumentParser(description="Plot Infidelity vs Number of Removed Coefficients")
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    
    args = parser.parse_args()
    
    pbc = True
    add = "_pbc" if pbc else ""
    if args.angle != 0.0:
        add += f"_{args.angle}"
        
    output_dir = "wavefunctions_data" + add
    t = [0,10,20,30,40,50,60,70,80,90,100]
    L = args.L
    Ntimes=100
    continuous_list = [True,False]

    #colors = cm.viridis(np.linspace(0, 1, len(t_idx_list)))
    n_values = np.arange(0, Ntimes + 1, 10)
    # Create a discrete colormap and boundary normalization for each step of n
    cmap = plt.get_cmap('viridis', len(n_values))
    norm = mcolors.BoundaryNorm(np.arange(-5, Ntimes + 6, 10)/Ntimes, cmap.N)

    
    for continuous in continuous_list:
        plt.figure(figsize=(10, 6))
        for t_idx in t:
            
           
            # 1. Load the groups to figure out the size of each group
            global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{args.angle}.npy")
            if not os.path.exists(global_filename):
                raise FileNotFoundError(f"Cannot find {global_filename}. Make sure groups are generated.")
        
            dense_groups_global = np.load(global_filename)
            
        
            # Map each group's representative index (the first element) to the size of that group
            # We ignore the -1 padding when counting the size
            rep_to_size = {}
            for g in dense_groups_global:
                valid_elements = g[g != -1]
                if len(valid_elements) > 0:
                    rep = valid_elements[0]
                    rep_to_size[rep] = len(valid_elements)
        
    
            
            
            inf_file = os.path.join(output_dir, f"inf_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
            sort_file = os.path.join(output_dir, f"sort_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
            
            if not os.path.exists(inf_file) or not os.path.exists(sort_file):
                print(f"Warning: Missing data files for Continuous={continuous}. Skipping...")
                continue
            
            # 2. Load the arrays and reverse them back to chronological order
            # (They were saved as inf_exact[::-1] and order_exact[::-1] in the main script)
            infs = np.load(inf_file)
            order = np.load(sort_file)
            # 3. Reconstruct the x-axis (cumulative sum of removed coefficients)
            # order array contains the representative index of each removed group
            group_sizes = [1]+[rep_to_size[rep] for rep in order]
            #print(group_sizes)
            x_cumulative_removed = np.cumsum(group_sizes)[:-1]
            
            # 4. Plot
            plt.plot(x_cumulative_removed, infs, color=cmap(norm(t_idx/Ntimes)),marker='.', markersize=4, label=f'{t_idx/Ntimes:.2f}')
            print(infs)
            
        plt.xlabel("$n_c$")
        plt.ylabel("Infidelity")
        plt.ylim([1e-10,1])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), ticks=n_values/Ntimes)
        cbar.set_label(" time $t$")
        plt.yscale("log")
        plt.title(f"Infidelity vs $n_c$ (L={L} t={t_idx*0.01} gi={args.gi}, gf={args.gf})")
        plt.yscale("log") # Infidelity usually drops logarithmically, so a log scale is helpful
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
    
        # Save the plot
        plot_path = os.path.join(output_dir, f"inf_plot_L{args.L}_gi{args.gi}_gf_{args.gf}continuous{continuous}.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved successfully to {plot_path}")
    

if __name__ == "__main__":
    main()
