#!/usr/bin/env python
import os
import argparse
import matplotlib.pyplot as plt
# 1. Restrict threads BEFORE importing netket/jax/numpy
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

import numpy as np
import netket as nk


def Fast_Hadamard(a):
    """
    Computes the Walsh-Hadamard Transform:
    f_hat(A) = (1 / 2^N) * sum_s f(s) X_A(s)
    Inverse is: f(s) = 2^N * Fast_Hadamard(f_hat)
    """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx", type=int, default=10)

    args, _ = parser.parse_known_args()
    params = vars(args)

    dt = 0.01
    Ndt = 100

   
    t_idx = params["t_idx"]
    gi = params["gi"]
    gf = params["gf"]
    angle = params["angle"]
    t_start = params["t_start"]
    pbc = True

    

    add = ""
    if pbc:
        add += "_pbc"
    if angle != 0.0:
        add += "_" + str(angle)

    output_dir = "wavefunctions_data" + add
    os.makedirs(output_dir, exist_ok=True)

    Ls = [8,10,12,14]
    KL_L = []
    for idx_L, L in enumerate(Ls):
        hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)
        states = hi.all_states()
        mask = np.sum(states == 1, axis=-1)
        orders = [i for i in range(0,L+1,2)]
        KL = np.ones((Ndt+1,len(orders)))
    
        base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
        filename = os.path.join(output_dir, base_name)

        try:
            psi_history = np.lib.format.open_memmap(
                filename,
                mode='r+',
                dtype=np.complex128,
                shape=(Ndt + 1, 2**L)
            )
            psi_t_idx = psi_history
            print(f"Loaded target wavefunction shape: {psi_t_idx.shape}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Generating dummy target wavefunction.")
            psi_t_idx = np.ones(2**L, dtype=np.complex128) / np.sqrt(2**L)

        av_sz = Fast_Hadamard(np.abs(psi_t_idx)**2)
        exact = Fast_Hadamard(np.log(np.abs(psi_t_idx)**2))
        logN = -exact[:,0]
        exact[:,0] = 0.0

    
        for idx_order,order in enumerate(orders):
            coeffs_order = np.zeros((Ndt+1,2**L))
            mask_order = mask <= order
            
            coeffs_order[:,mask_order] = exact[:,mask_order]
            logNtrunc = -np.log(np.linalg.norm( Fast_Hadamard(coeffs_order) * 2**L,axis=-1))

            KL[:,idx_order] = np.sum(exact[:,~mask_order]*av_sz[:,~mask_order],axis=-1) 
            KL[:,idx_order] -= (logN-logNtrunc)/2**L
            
        KL_L.append(KL)    


    for idx_L,L in enumerate(Ls):
        plt.figure()
        orders = [i for i in range(0,L+1,2)]
        for tt in range(Ndt+1):
            plt.plot(orders,KL_L[idx_L][tt])
        plt.savefig(f"KL_L{L}.png")
