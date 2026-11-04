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
parser.add_argument("--t_idx", type=int, default=40)
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
t_idx = params["t_idx"]

ii=int(L/2)
pbc=True
hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)
states = hi.all_states()
add=""

if pbc:
    add+="_pbc"
if angle != 0.0:
    add+="_"+str(angle)
    initial_mask  = np.ones(2**L,dtype=int)
else:
    initial_mask = np.arange(2**L,dtype=int) % 2 == 0

states = states[initial_mask]

proxy= -1 * np.ones(ii)
proxy[-1]= 1
clusters_ii = states[(states[:,:ii] == proxy).all(axis=1)]
clusters_idx = hi.states_to_numbers(clusters_ii)

local_idx_respect_ii = []
for x in range(ii):
    
    proxy = -1 * np.ones(ii-x)
    proxy[0] = 1
    
    local_idx_respect_ii.append(hi.states_to_numbers(clusters_ii[ (clusters_ii[:,ii+x:] == proxy).all(axis=1)]))

print(local_idx_respect_ii)
print(clusters_ii)



len_mask = np.sum(initial_mask)


output_dir = "wavefunctions_data"+add
os.makedirs(output_dir, exist_ok=True)


plt.figure()


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
log_WHT[0] = 0
plt.figure()
plt.xlim([0,200])
plt.yscale("log")
plt.plot(np.abs(log_WHT[psi_sort].real))
plt.savefig("sorting_trial.png")

mask = np.isin(psi_sort,clusters_idx)

sort_local_ii = psi_sort[mask]
print("---",sort_local_ii)

n_clusters_ii = len(sort_local_ii)
print(n_clusters_ii)
spectra_local_ii = []
emp_distribution = np.zeros((ii,n_clusters_ii))

plt.figure()
for x in range(ii):

    aux = np.isin(sort_local_ii,local_idx_respect_ii[x][:])
    print(aux)
    #aux = sort_local_ii in local_idx_respect_ii[x]
    
    emp_distribution[x] = np.array([np.sum(aux[:j+1]) for j in range(n_clusters_ii)])
    emp_distribution[x] = emp_distribution[x]/(emp_distribution[x,-1]+1e-14)


    plt.plot(emp_distribution[x],label=str(x+1))

#print(emp_distribution)
#plt.xlim([0,50])
#plt.yscale("log")
plt.legend()
plt.savefig("proposal.png")

psi_sort.flush()
del psi_sort

psi_history.flush()

del psi_history











