import os
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
from class_WF import rotated_IsingModel


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
singularities_filename = os.path.join(output_dir, f"average_order_gi{gi:.2f}_gf{gf:.2f}.npy")
average_order = np.zeros((len(Ls),Ndt+1))

for idx_L,L in enumerate(Ls):
    hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)
    S_array = np.sum(hi.all_states() == 1,axis=-1)
    print(S_array)
    H_out = rotated_IsingModel(angle*np.pi/180,gf,L,hi,pbc=pbc)
    H_out = H_out.to_sparse()
    
    base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    filename = os.path.join(output_dir, base_name)

    psi_history = np.lib.format.open_memmap(
        filename,
        mode='r+',
        dtype=np.complex128,
        shape=(Ndt+1,2**L)
    )
    
    for idx_t in range(Ndt+1):
        e_local = Fast_Hadamard((H_out@psi_history[idx_t])/psi_history[idx_t])
        e_local_square =(e_local*np.conjugate(e_local)).real
        print(np.mean(e_local_square * S_array)/np.mean(e_local_square))
        average_order[idx_L,idx_t] = np.mean(e_local_square * S_array)/np.mean(e_local_square)
    print("finished L=",str(L))
av_filename = os.path.join(output_dir,"average_order_array.npy")
np.save(av_filename,average_order)






