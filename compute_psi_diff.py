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

def Fast_Hadamard(a):
    ordering = 1.0    
    a = a.copy()
    L = a.shape[-1]
    h = 1
    
    while h < L:
        a2 = a.reshape(a.shape[:-1] + (L // (2 * h), 2, h))
        x = a2[..., 0, :]
        y = a2[..., 1, :]

        t = x.copy()
        a2[..., 0, :] = 0.5 * (t + y)
        a2[..., 1, :] = 0.5 * (t - y) * ordering

        h *= 2

    return a



Ls = [8,10,12,14,16,18,20,22]
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


mean_psi_diff = np.zeros((len(Ls),Ndt+1))
std_psi_diff = np.zeros((len(Ls),Ndt+1))

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
        #WHT =Fast_Hadamard(np.log(psi_history[idx_t]))
        #WHT[0] = 0
        new_psi = psi_history[idx_t] #np.exp(2**Ls[idx_L] * Fast_Hadamard(WHT))
        
        flip_ratio_i = np.abs(1/new_psi[single_flip_numbers]-1/(new_psi))
        mean_psi_diff[idx_L,idx_t] = np.mean(flip_ratio_i)
        std_psi_diff[idx_L,idx_t]  = np.std(flip_ratio_i)
        
    psi_history.flush()
    del psi_history
    print("done",L)
    
mean_filename = os.path.join(output_dir,"mean_psi_diff_array.npy")
std_filename = os.path.join(output_dir,"std_psi_diff_array.npy")
np.save(mean_filename,mean_psi_diff)
np.save(std_filename,std_psi_diff)






