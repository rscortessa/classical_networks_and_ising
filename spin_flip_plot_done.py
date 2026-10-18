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
i=2
eps = 1e-16

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
flip_filename = os.path.join(output_dir,"flip_array.npy")
flip_array = np.load(flip_filename)

singularities_array = np.zeros((len(Ls),Ndt+1))
for idx_L,L in enumerate(Ls):
    singularities_filename = os.path.join(output_dir, f"singularities_L{L}_gi{gi:.2f}_gf{gf:.2f}.npy")
    singularities_array[idx_L] = np.load(singularities_filename)

plt.figure()
plt.title(r"$\langle |\Psi(\vec{\sigma}^{(1))},t)/\Psi(\vec{\sigma}^{(0))},t)| \rangle \;,\;\sigma_i^{(1)}=-\sigma_i^{(0)}$",fontsize=fontsize)
plt.xlabel("time(t)",fontsize=fontsize)
for idx_L,L in enumerate(Ls):
    plt.plot(time,flip_array[idx_L],label=str(L))
    plt.plot(time,singularities_array[idx_L],label=str(L))
plt.legend(fontsize=fontsize)
flip_plot = os.path.join(output_dir,"spin_flip_gi"+str(gi)+"_gf"+str(gf)+".png")
plt.savefig(flip_plot)






