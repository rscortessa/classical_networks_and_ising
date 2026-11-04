#!/usr/bin/env python                                                                                                                                                                                                                                                                                                                                                                        
# coding: utf-8                                                                                                                                                                                                                                                                                                                                                                              

# In[5]:                                                                                                                                                                                                                                                                                                                                                                                     



import os
import matplotlib.pyplot as plt
# 1. Restrict threads BEFORE importing netket/jax/numpy                                                                                                                                                                                                                                                                                                                                      
os.environ["OMP_NUM_THREADS"] = "2" # Example: Restrict to 2 threads per node                                                                                                                                                                                                                                                                                                                
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
# Tell JAX to use exactly 2 threads for its CPU operations                                                                                                                                                                                                                                                                                                                                   
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

import numpy as np
import netket as nk
import sys
sys.path.append("../netket_with_NQS")



Ls = [8,10,12,14,16,18,20,22]
gi = 1.5
gf = 0.5
angle=0.0
i=1
eps = 1e-10

t_start = 0.0
dt = 0.01
Ndt = 100
fontsize=15

pbc=True
add=""
if pbc:
    add+="_pbc"
time = t_start+np.arange(Ndt+1)*dt
output_dir = "wavefunctions_data"+add
os.makedirs(output_dir, exist_ok=True)

mean_filename = os.path.join(output_dir,"order_psi_diff_array.npy")

mean_psi_diff= np.load(mean_filename)


n_L, n_t = mean_psi_diff.shape

# Define system sizes and time axis (adjust parameters as needed)
# If Ls is not already defined in your scope:
# Ls = [12, 14, 16, 18, 20]
t_max = 1.0
times = np.linspace(0.0, t_max, n_t)

# Create figure with two subplots side by side
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5), sharex=True)

# Color palette across system sizes
colors = plt.cm.viridis(np.linspace(0, 0.9, n_L))

for idx in range(n_L):
    label = f"L = {Ls[idx]}" if "Ls" in locals() else f"System {idx + 1}"

    # Left plot: Mean
    ax1.plot(times, mean_psi_diff[idx], label=label, color=colors[idx], lw=1.8)

# Formatting Left Subplot (Mean)
plt.suptitle(r"$f(s)=1/ \Psi(s)$", fontsize=13)
ax1.set_title(r"$\langle |A|/L \rangle_{f}$", fontsize=13)
ax1.set_xlabel(r"Time $t$", fontsize=12)
ax1.set_yscale("log")
#ax1.set_ylabel(
#    r"$\mathbb{E}\left[\log_{10} |\psi(s^\prime)/\psi(s)|\right]$", fontsize=12
#)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(frameon=True)

plt.tight_layout()

sing_plot = os.path.join(output_dir,"avg_psi_diff_gi"+str(gi)+"_gf"+str(gf)+".png")
plt.savefig(sing_plot)






