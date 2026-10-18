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

Ls = [8, 10, 12, 14, 16, 18, 20, 22]
gi = 1.5
gf = 0.5
angle = 0.0
i = 2
eps = 1e-10

t_start = 0.0
dt = 0.01
Ndt = 100

pbc = True
add = "_pbc" if pbc else ""
output_dir = "wavefunctions_data" + add
os.makedirs(output_dir, exist_ok=True)

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
    singularities_filename = os.path.join(output_dir, f"singularities_L{L}_gi{gi:.2f}_gf{gf:.2f}.npy")
    
    # FIX 1: Pass log_flip_filename instead of filename
    log_flip_ratio_file = np.lib.format.open_memmap(
        log_flip_filename,
        mode='w+',
        dtype=np.float64, # explicitly using numpy float64 
        shape=(Ndt+1, 2**L)
    )
    N_singularities = np.lib.format.open_memmap(
        singularities_filename,
        mode='w+',
        dtype=np.float64, # explicitly using numpy float64 
        shape=(Ndt+1,)
    )

    for idx_t in range(Ndt + 1):
        # Calculate flip ratio        
        numerator = np.abs(psi_history[idx_t, single_flip_numbers])
        denominator = np.abs(psi_history[idx_t])

        flip_ratio_i = numerator/denominator
        # FIX 2: Add eps to flip_ratio_i to avoid RuntimeWarnings & -inf when storing back to mmap
        log_flip_ratio = np.log(flip_ratio_i)
        mask = np.isfinite(log_flip_ratio)

        log_flip_ratio_file[idx_t] = log_flip_ratio
        N_singularities[idx_t] = np.sum(mask != True)/2**L
        
        log_flip_ratio = log_flip_ratio[mask]
        
        # Compute histogram of log(flip_ratio_i)
        counts, bin_edges = np.histogram(log_flip_ratio, bins=100, density=True)

        # Save histogram data arrays (.npz) incorporating parameter names
        hist_data_filename = os.path.join(
            output_dir, 
            f"hist_data_L{L}_t{idx_t}_gi{gi:.2f}_gf{gf:.2f}.npz"
        )
        np.savez(hist_data_filename, counts=counts, bin_edges=bin_edges)

        # Save histogram plot figure (.png)
        plt.figure(figsize=(8, 5))
        plt.hist(log_flip_ratio, bins=100, density=True, alpha=0.75, color='teal', edgecolor='black')
        plt.title(f"Histogram of log(flip_ratio) [L={L}, t={idx_t*dt:.2f}, gi={gi}, gf={gf}]")
        plt.xlabel("log(flip_ratio)")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plot_filename = os.path.join(
            output_dir, 
            f"hist_plot_L{L}_t{idx_t}_gi{gi:.2f}_gf{gf:.2f}.png"
        )
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    N_singularities.flush()
    log_flip_ratio_file.flush()
    # No need to flush psi_history since it's loaded in 'r' (read-only) mode
    del psi_history
    del log_flip_ratio_file
    del N_singularities
    print("Completed histograms for L =", L)
