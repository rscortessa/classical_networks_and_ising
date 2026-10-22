#!/usr/bin/env python                       
import os

# 1. Restrict threads BEFORE importing netket/jax/numpy                                                                                                                                                                                     
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

import numpy as np
import netket as nk
import matplotlib.pyplot as plt
import sys
sys.path.append("../netket_with_NQS")

Ls = [8, 10, 12, 14, 16, 18, 20]
gi = 1.5
gf = 0.5
angle = 0.0
i = 2

t_start = 0.0
dt = 0.01
Ndt = 100
eps=1e-14
pbc = True
add = "_pbc" if pbc else ""
output_dir = "wavefunctions_data" + add
os.makedirs(output_dir, exist_ok=True)

singularities_filename = os.path.join(output_dir, f"singularities_gi{gi:.2f}_gf{gf:.2f}eps{eps=}.npy")
N_singularities = np.lib.format.open_memmap(
     singularities_filename,
     mode='w+',
     dtype=np.float64, 
     shape=(len(Ls),Ndt+1,)
    )

for idx_L, L in enumerate(Ls):
    hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)

    flip_idx = np.array(hi.all_states())
    flip_idx[:, i] = flip_idx[:, i] * (-1)
    single_flip_numbers = hi.states_to_numbers(flip_idx)

    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename = os.path.join(output_dir, base_name)

    if not os.path.exists(filename):
        print(f"File not found: {filename}, skipping L={L}")
        continue

    psi_history = np.lib.format.open_memmap(
        filename,
        mode='r',
        dtype=np.complex128,
        shape=(Ndt + 1, 2**L)
    )

    log_flip_filename = os.path.join(output_dir, f"logflip_L{L}_gi{gi:.2f}_gf{gf:.2f}.npy")
    
    
    log_flip_ratio_file = np.lib.format.open_memmap(
        log_flip_filename,
        mode='w+',
        dtype=np.float64, 
        shape=(Ndt+1, 2**L)
    )
    
   

    for idx_t in range(Ndt + 1):
        numerator = np.abs(psi_history[idx_t, single_flip_numbers])
        denominator = np.abs(psi_history[idx_t])
        
        #N_singularities[idx_L] = np.sum(denominator < eps)/2**L
        
        # Ignore division-by-zero warnings, allowing np.inf and np.nan to be created naturally
        with np.errstate(divide='ignore', invalid='ignore'):
            flip_ratio_i = numerator / denominator
            log_flip_ratio = np.log10(flip_ratio_i)
            
        mask = np.isfinite(log_flip_ratio)

        # Save the full array (including infs and NaNs) to the memory map
        log_flip_ratio_file[idx_t] = log_flip_ratio
        
        # Save the fraction of singular (non-finite) states

        # Filter for histograms
        log_flip_ratio_finite = log_flip_ratio[mask]
        
        if len(log_flip_ratio_finite) > 0:
            counts, bin_edges = np.histogram(log_flip_ratio_finite, bins="auto", density=True)
        else:
            counts, bin_edges = np.array([]), np.array([])

        hist_data_filename = os.path.join(
            output_dir, 
            f"hist_data_L{L}_t{idx_t}_gi{gi:.2f}_gf{gf:.2f}.npz"
        )
        np.savez(hist_data_filename, counts=counts, bin_edges=bin_edges)

        log_psi = np.sort(log_flip_ratio_file[idx_t])
        cum_log_psi = np.arange(1,2**L+1,1)/2**L
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Subplot 1: Histogram ---
        if len(log_flip_ratio_finite) > 0:
            ax1.hist(log_flip_ratio_finite, bins=100, density=True, alpha=0.75, color='teal', edgecolor='black')
        ax1.set_title(r"Histogram $\log_{10}(|\psi(s')/ \psi(s)|)$")
        ax1.set_xlabel(r"$\log_{10}(|\psi(s')/ \psi(s)|)$")
        ax1.set_ylabel("Density")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # --- Subplot 2: Cumulative Plot ---
        ax2.plot(log_psi, cum_log_psi, linestyle="dashed", color='darkorange')
        ax2.set_title(r"Cumulative $\log_{10}(|\psi(s')/ \psi(s)|)$")
        ax2.set_xlabel(r"$\log_{10}(|\psi(s')/ \psi(s)|)$")
        ax2.set_ylabel("Cumulative Density")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.set_yscale("log")
        # Overall title for parameters
        fig.suptitle(rf"$L={L}, t={idx_t*dt:.2f}, gi={gi}, gf={gf}$", fontsize=14)
        plt.tight_layout()

        plot_filename = os.path.join(
            output_dir,
            f"hist_plot_L{L}_t{idx_t}_gi{gi:.2f}_gf{gf:.2f}.png"
        )
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)
        
        #plt.figure(figsize=(8, 5))
        #if len(log_flip_ratio_finite) > 0:

        #    plt.hist(log_flip_ratio_finite, bins=100, density=True, alpha=0.75, color='teal', edgecolor='black')
        #    plt.plot(log_psi,cum_log_psi,linestyle="dashed")
            
        #plt.title(r"Histogram $\log_{10}"+rf"(|\psi(s^')/ \psi(s)|)\;[L={L}, t={idx_t*dt:.2f}, gi={gi}, gf={gf}]$")
        #plt.xlabel(r"$\log_{10}(|\psi(s^')/ \psi(s)|)$")
        #plt.ylabel("Density")
        #plt.grid(True, linestyle="--", alpha=0.5)
        #
        #plot_filename = os.path.join(
        #    output_dir, 
        #    f"hist_plot_L{L}_t{idx_t}_gi{gi:.2f}_gf{gf:.2f}.png"
        #)
        #plt.savefig(plot_filename, dpi=300)
        #plt.close()

    #N_singularities.flush()
    log_flip_ratio_file.flush()
    del psi_history
    del log_flip_ratio_file
    #del N_singularities
    print("Completed histograms for L =", L)
