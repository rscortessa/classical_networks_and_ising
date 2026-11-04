#!/usr/bin/env python                                                                                                                                                                                                                       
# coding: utf-8                                                                                                                                                                                                                             

# In[5]:                                                                                                                                                                                                                                    



import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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



Ls = [8,10,12,14,16]
gi = 1.5
gf = 0.5
angle=0.0
i=2
eps = 1e-10

t_start = 0.0
dt = 0.01
Ndt = 100


pbc=True
add=""
if pbc:
    add+="_pbc"
time = t_start+np.arange(Ndt+1)*dt
output_dir = "wavefunctions_data"+add
os.makedirs(output_dir, exist_ok=True)
singularities_filename = os.path.join(output_dir, f"singularities_gi{gi:.2f}_gf{gf:.2f}eps{eps=}.npy")
N_singularities = np.lib.format.open_memmap(
     singularities_filename,
     mode='w+',
     dtype=np.float64, 
     shape=(len(Ls),Ndt+1,)
    )



prob_spin_flip_i = np.ones((len(Ls),Ndt+1))


for idx_L,L in enumerate(Ls):
    hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)

    flip_idx = np.array(hi.all_states())
    flip_idx[:,i] = flip_idx[:,i]*(-1)
    single_flip_numbers = hi.states_to_numbers(flip_idx)

    flip_ratio_i = np.zeros(2**L)

    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"

    filename = os.path.join(output_dir, base_name)
    psi_history = np.lib.format.open_memmap(
        filename,
        mode='r+',
        dtype=np.complex128,
        shape=(Ndt+1,2**L)
    )
    
    plt.figure()
    log_psi = np.log(psi_history).real - np.mean(np.log(psi_history).real,axis=-1)[:,None]
    S_t_sum_square = np.sum( log_psi**2,axis=-1)
    sorted_S_square = np.sort(log_psi**2,axis=-1)[:,::-1]
    cum_sum = np.cumsum(sorted_S_square,axis=-1)/S_t_sum_square[:,None]
    ncoeffs = np.arange(0,2**L,1)

    cmap = cm.viridis  # Or 'plasma', 'coolwarm', 'inferno', etc.
    norm = mcolors.Normalize(vmin=0, vmax=Ndt)

    for idx_t in [0,10,20,30,40,60,80,90]:
        plt.scatter(ncoeffs, cum_sum[idx_t], s=4,color=cmap(norm(idx_t)))

    # Create a ScalarMappable and add the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.ylim([0,1])
    plt.xlim([0,0.2*2**L])
    plt.colorbar(sm, ax=plt.gca(), label="idx_t")
    plt.savefig(f"L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.png")
    psi_history.flush()
    del psi_history
    print("done",L)
flip_filename = os.path.join(output_dir,"flip_array.npy")
np.save(flip_filename,prob_spin_flip_i)






