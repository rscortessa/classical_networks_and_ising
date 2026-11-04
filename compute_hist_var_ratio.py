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

fig = plt.figure()

times = np.arange(0,Ndt+1)*0.01
for idx_L, L in enumerate(Ls):
    
    log_flip_filename = os.path.join(output_dir, f"logflip_L{L}_gi{gi:.2f}_gf{gf:.2f}.npy")
    log_flip_ratio_file = np.lib.format.open_memmap(
        log_flip_filename,
        mode='r+',
        dtype=np.float64, 
        shape=(Ndt+1, 2**L)
    )
    log_ratio = np.abs(log_flip_ratio_file)
    sec_moment = np.mean(log_ratio**3,axis=-1)
   
    plt.plot(times,sec_moment,label=f"L={L}")
        
    
    log_flip_ratio_file.flush()
    del log_flip_ratio_file
    
    #del N_singularities
    print("Completed histograms for L =", L)



plot_filename = os.path.join(
            output_dir,
            f"square_mean_plot_L{L}_gi{gi:.2f}_gf{gf:.2f}.png"
    )
plt.savefig(plot_filename, dpi=300)
plt.close(fig)
print("Finished",plot_filename)
