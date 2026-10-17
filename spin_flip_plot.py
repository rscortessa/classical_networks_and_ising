#!/usr/bin/env python                                                                                                                                                                                                                       
# coding: utf-8                                                                                                                                                                                                                             

# In[5]:                                                                                                                                                                                                                                    



import os

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
i=2
eps = 1e-16

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

    for idx_t in range(Ndt+1):
        flip_ratio_i = np.abs(psi_history[idx_t,single_flip_numbers]/(psi_history[idx_t]+eps))
        prob_spin_flip_i[idx_L,idx_t] = np.mean(flip_ratio_i)

    psi_history.flush()
    del psi_history
    print("done",L)
flip_filename = os.path.join(output_dir,"flip_array.npy")
np.save(flip_filename,prob_spin_flip_i)






