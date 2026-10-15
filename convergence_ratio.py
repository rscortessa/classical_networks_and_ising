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
parser.add_argument("--t_idx", type=int, default=50)
args_list=None
args, _ = parser.parse_known_args(args_list)
params = vars(args)



eps = 1e-10
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
    initial_mask  = np.ones(2**L,dtype=int)
else:
    initial_mask = np.arange(2**L,dtype=int) % 2 == 0

len_mask = np.sum(initial_mask)
    

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

plt.figure()

for t_idx in [40,41,42,43,44,46,47,48,49,51,52,53,54,56,57,58,59,61,62,63,64,66,67,68,69,71,72,73,74]:#range(Ndt+1):
    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename = os.path.join(output_dir, base_name)

    psi_history = np.lib.format.open_memmap(
        filename, 
        mode='r+', 
        dtype=np.complex128, 
        shape=(Ndt+1,2**L)
    )
    
    psi_t_idx = psi_history[t_idx]
    print(psi_t_idx.shape)


    continuous = True

    log_WHT = Fast_Hadamard(np.log(psi_t_idx))   
    log_sorting = np.sort(np.abs(log_WHT.real))[::-1]
    log_sorting = log_sorting[:2**(L-1)]

    base_name_sort = f"sort_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy"
    filename_sort = os.path.join(output_dir, base_name_sort)
    psi_sort = np.lib.format.open_memmap(
        filename_sort, 
        mode='r+', 
        dtype=int, 
        shape=(len_mask,)
        )

    print(psi_sort)
    
    #print(np.abs(log_WHT[psi_sort].real)[:-1]/np.abs(log_WHT[psi_sort].real)[1:])
    safety_number = np.sum(np.abs(log_WHT[psi_sort].real) > eps)
    
    
    psi_ratio[t_idx,:safety_number-1,0] = np.abs(log_WHT[psi_sort].real)[:safety_number-1]/np.abs(log_WHT[psi_sort].real)[1:safety_number]
    psi_ratio[t_idx,safety_number-1:,0] = 0
    
    safety_number = np.sum(log_sorting > eps)
    
    psi_ratio[t_idx,:safety_number-1,1] = np.abs(log_sorting[:safety_number-1].real/log_sorting[1:safety_number].real)
    psi_ratio[t_idx,safety_number-1:,1] = 0.0
    
    print("ratio")
    print(psi_ratio[t_idx,:safety_number-1,0])
        


    plt.plot(1/psi_ratio[t_idx,:safety_number-1,1])

plt.yscale("log")
plt.ylim([1e-4,10])
plt.savefig("ratio_series_L"+str(L)+"gi"+str(gi)+"gf"+str(gf)+"angle"+str(angle)+".png")

psi_sort.flush()
del psi_sort

psi_ratio.flush()
psi_history.flush()

del psi_history
del psi_ratio
