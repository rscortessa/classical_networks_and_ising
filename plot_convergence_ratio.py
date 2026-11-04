#!/usr/bin/env python
# coding: utf-8

# In[5]:



import os
import argparse
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
import os
sys.path.append("../netket_with_NQS")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

    
# Initial parameters ....
parser = argparse.ArgumentParser()

parser.add_argument("--L", type=int, default=16)
parser.add_argument("--gi", type=float, default=1.5)
parser.add_argument("--gf", type=float, default=0.5)
parser.add_argument("--angle", type=float, default=0.0)
parser.add_argument("--t_start", type=float, default=0.0)
args_list=None
args, _ = parser.parse_known_args(args_list)
params = vars(args)



eps = 1e-16
dt = 0.01
Ndt = 100

L = params["L"]
gi = params["gi"]
gf = params["gf"]
angle= params["angle"]
t_start = params["t_start"]

pbc=True
hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)
add=""
if pbc:
    add+="_pbc"
if angle != 0.0:
    add+="_"+str(angle)
    

output_dir = "wavefunctions_data"+add
os.makedirs(output_dir, exist_ok=True)



base_name_ratio = f"ratio_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
filename = os.path.join(output_dir, base_name_ratio)
psi_ratio = np.lib.format.open_memmap(
        filename, 
        mode='w+', 
        dtype=float, 
        shape=(Ndt+1,2**(L-1)-1,2)
)

t=[40,41,42,43,44,46,47,48,49,51,52,53,54,56,57,58,59,61,62,63,64,66,67,68,69,71,72,73,74]#range(Ndt+1):
plt.figure()
plt.title("plot")
plt.yscale("log")
for t_idx in t:
    print(psi_ratio[t_idx,:,0])
    #plt.plot(psi_ratio[t_idx,:,1])
    plt.plot(psi_ratio[t_idx,:,0])
plt.savefig("ratio_series_L"+str(L)+"gi"+str(gi)+"gf"+str(gf)+"angle"+str(angle)+".png")
