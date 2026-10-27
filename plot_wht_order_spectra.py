#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netket as nk
import scipy.special as sc

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


def compute_wht_periodic_spectra(wht_log_psi,rep_to_size):
    """
    Computes the L1 and L2 Relative Tail Remainders along the sorted group sequence.
    `sort_order` is ordered from most important (index 0) to least important (index -1).
    """
    # Filter out index 0 (global normalization offset) if present
    reps = np.array(list(rep_to_size.keys()),dtype=int)
    n_terms = np.array(list(rep_to_size.values()))
    wht_sorted = np.abs(wht_log_psi[reps])

    # Compute the weighted group contribution: WHT_coeff * n_terms
    idx_terms =  np.argsort(wht_sorted * n_terms)[::-1]
    w_l1 = wht_sorted[idx_terms] * n_terms[idx_terms]

    n_terms = n_terms[idx_terms] 
    cum_coeffs = np.cumsum(n_terms)

    #cum_coeffs = np.insert(cum_coeffs, 0, 0.0)
    
    # Drop the final point where Remainder == 0 so log-scale plotting stays clean
    return cum_coeffs,w_l1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Relative Tail Remainder of WHT expansion.")
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--norm", type=str, default="both", choices=["L1", "L2", "both"],
                        help="Choose whether to plot L1, L2, or both tail remainders.")

    args = parser.parse_args()

    L = args.L
    gi = args.gi
    gf = args.gf
    angle = args.angle
    t_start = args.t_start
    dt = args.dt
    Ndt = args.Ndt
    nstates = 1 << L

    pbc = True
    add = "_pbc" if pbc else ""
    if angle != 0.0:
        add += f"_{angle}"

    output_dir = "wavefunctions_data" + add

    # 2. Load target wavefunctions
    wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
    wf_file = os.path.join(output_dir, wf_base)
    if not os.path.exists(wf_file):
        raise FileNotFoundError(f"Missing wavefunction file: {wf_file}")

    psi_history = np.lib.format.open_memmap(
        wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
    )

    t_idx_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    colors = cm.viridis(np.linspace(0, 1, len(t_idx_list)))

    
    fig, axes = plt.subplots(3, 2, figsize=(10, 5),squeeze=False)
    plt.suptitle(rf"$L={L},\;g_i={gi},\;g_f={gf}$",fontsize=10)

    hi = nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)
    states = hi.all_states()
    order = np.sum(states == 1,axis=-1)

    w_average = np.zeros(len(t_idx_list))
    w_exp = np.zeros(len(t_idx_list))
    w_binom = np.zeros(len(t_idx_list))
    w_co = np.zeros(len(t_idx_list))
    
    for c_idx, t_idx in enumerate(t_idx_list):

        psi_t = np.abs(psi_history[t_idx])
        psi_t /= np.sqrt(np.sum(psi_t ** 2))
        wht_log_psi = Fast_Hadamard(np.log(psi_t)).real
        wht_log_psi[0] = 0

        all_weight = np.sum(wht_log_psi*wht_log_psi)
        list_orders = np.arange(2,L+1,2)
        w_orders = np.zeros(len(list_orders))
        for idx,idx_order in enumerate(list_orders):
            mask = order == idx_order
            coeff = np.sum(wht_log_psi.real[mask]**2)/all_weight
            w_orders[idx] = coeff

        w_average[c_idx] = np.sum(w_orders * list_orders)
        w_exp[c_idx] = np.log(np.sum(w_orders * np.exp(list_orders)))
        w_binom[c_idx] = np.log(np.sum(sc.binom(L,list_orders)*w_orders))
        w_co[c_idx] =  all_weight
        
        axes[0,0].plot(list_orders,w_orders,linestyle="dashed",color=colors[c_idx],label=f"{t_idx*0.01:.1f}", lw=1.6)

    axes[0,1].plot( np.array(t_idx_list)*0.01,w_average,linestyle="solid",marker="o",lw=1.0)
    axes[1,0].plot( np.array(t_idx_list)*0.01,w_exp,linestyle="solid",marker="o",lw=1.0)
    axes[1,1].plot( np.array(t_idx_list)*0.01,w_binom,linestyle="solid",marker="o",lw=1.0)
    axes[2,0].plot( np.array(t_idx_list)*0.01,w_co,linestyle="solid",marker="o",lw=1.0)
    
    axes[0,0].set_yscale("log")
    axes[0,0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0,0].set_xlabel("order")

    axes[0,1].grid(True, which="both", ls="--", alpha=0.4)
    axes[0,1].set_xlabel(r"time$(t)$")
    
    axes[1,0].grid(True, which="both", ls="--", alpha=0.4)
    axes[1,0].set_xlabel(r"time$(t)$")

    
    axes[2,0].grid(True, which="both", ls="--", alpha=0.4)
    axes[2,0].set_xlabel(r"time$(t)$")


    
        
    axes[0,0].set_title(r"$P(k),of\;the\;CN$")
    axes[0,0].set_ylabel(r"$P(k)$")
    axes[0,0].legend(fontsize=8, ncol=2)
    axes[0,0].set_ylim([1e-9,10])

    axes[0,1].set_ylabel(r"$\langle k \rangle$")
    axes[1,0].set_ylabel(r"$\log \langle e^{k} \rangle$")
    axes[1,1].set_ylabel(r"$\log \langle \binom{n}{k} \rangle$")
    axes[2,0].set_ylabel(r"$\omega_{co}$")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"wht_order_spectra_L{L}_gi{gi:.2f}_gf{gf:.2f}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Relative Tail Remainder plot to {plot_path}")
    plt.show()
