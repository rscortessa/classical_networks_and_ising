#!/usr/init/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def exp_decay(L, A, B):
    return A+L*B

def main():
    parser = argparse.ArgumentParser(description="Size scaling analysis of Fidelity (F[0] and F[1]) vs L")
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
    Ls = [ 10,12,14,16,18]
    t_idxs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    times = np.array(t_idxs) * args.dt
    continuous_list = [True, False]

    for continuous in continuous_list:
        # Storage for fit parameters and errors across times for F[0] and F[1]
        results = {
            'F0': {'A': [], 'A_err': [], 'B': [], 'B_err': []},
            'F1': {'A': [], 'A_err': [], 'B': [], 'B_err': []}
        }
        
        valid_times = []

        
        
        for t_idx in t_idxs:
            F0_vals = []
            F1_vals = []
            Ls_used = []
            
            
            for L in Ls:
                inf_file = os.path.join(output_dir, f"kl_L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy")
                
                if not os.path.exists(inf_file):
                    continue
                
                infs = np.load(inf_file)
                if len(infs) < 2:
                    continue
                
                # Compute Fidelity F = 1 - infs
                F =infs
                F0_vals.append(F[10])
                F1_vals.append(F[20])
                Ls_used.append(L)
                
            # Require at least 3 points for a meaningful exponential fit
            if len(Ls_used) < 3:
                print(f"Warning: Not enough system sizes for t_idx={t_idx}, Continuous={continuous}. Skipping time step.")
                continue
                
            valid_times.append(t_idx * args.dt)

            Ls_arr = np.array(Ls_used, dtype=float)




            
            # Fit F[0]
            try:
                popt_0, pcov_0 = curve_fit(exp_decay, Ls_arr, np.array(F0_vals), p0=(1.0, 0.1), maxfev=50000)

                perr_0 = np.sqrt(np.diag(pcov_0))
                
                results['F0']['A'].append(popt_0[0])
                results['F0']['A_err'].append(perr_0[0])
                results['F0']['B'].append(popt_0[1])
                results['F0']['B_err'].append(perr_0[1])
                    
            except Exception as e:
                results['F0']['A'].append(np.nan); results['F0']['A_err'].append(np.nan)
                results['F0']['B'].append(np.nan); results['F0']['B_err'].append(np.nan)

            # Fit F[1]
            try:
                popt_1, pcov_1 = curve_fit(exp_decay, Ls_arr, np.array(F1_vals), p0=(1.0, 0.1), maxfev=50000)
                perr_1 = np.sqrt(np.diag(pcov_1))
    
        
                results['F1']['A'].append(popt_1[0])
                results['F1']['A_err'].append(perr_1[0])
                results['F1']['B'].append(popt_1[1])
                results['F1']['B_err'].append(perr_1[1])
            except Exception as e:
                results['F1']['A'].append(np.nan); results['F1']['A_err'].append(np.nan)
                results['F1']['B'].append(np.nan); results['F1']['B_err'].append(np.nan)
                
                
            plt.figure()
            for idx_L,L in enumerate(Ls):
                plt.scatter(L,exp_decay(L,results['F0']['A'][-1], results['F0']['B'][-1]),color="black")
                plt.scatter(L,F0_vals[idx_L])
        
            plt.savefig("0_"+str(t_idx)+".png")

            plt.figure()
            for idx_L,L in enumerate(Ls):
                plt.scatter(L,exp_decay(L,results['F1']['A'][-1], results['F1']['B'][-1]),color="black")
                plt.scatter(L,F1_vals[idx_L])
        
            plt.savefig("1_"+str(t_idx)+".png")

        valid_times = np.array(valid_times)

            
        # Plotting A and B parameters vs Time
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        
        # --- F[0] : A vs Time ---
        axes[0].errorbar(valid_times, results['F0']['A'], yerr=results['F0']['A_err'], fmt='-o', color='blue',label=r"$n_c=0$",capsize=3)
        axes[0].errorbar(valid_times, results['F1']['A'], yerr=results['F1']['A_err'], fmt='-s', color='red',label=r"$n_c=1$", capsize=3)
   
        axes[0].set_title(r"$A(n_c,t)$")
        axes[0].set_ylabel("$A$")
        axes[0].set_xlabel("Time $t$")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()
        # --- F[0] : B vs Time ---
        axes[1].errorbar(valid_times, results['F0']['B'], yerr=results['F0']['B_err'], fmt='-o', color='blue',label=r"$n_c=0$", capsize=3)
        axes[1].errorbar(valid_times, results['F1']['B'], yerr=results['F1']['B_err'], fmt='-o', color='red',label=r"$n_c=1$", capsize=3)
        axes[1].set_title(r"$B(n_c,t)$")
        axes[1].set_ylabel("$B$")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Time $t$")
        axes[1].legend()
        # --- F[1] : A vs Time ---
        
   
        fig.suptitle(r"Fit Parameters $KL(n_c,t)=A(n_c,t)e^{-B(n_c,t)L}$"+"\n"+f"($g_i={args.gi}, g_f={args.gf}$", fontsize=14)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"size_scaling_fit_params_continuous{continuous}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved size scaling plot successfully to {plot_path}")

if __name__ == "__main__":
    main()
