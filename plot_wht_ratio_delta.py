#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def Fast_Hadamard(a):
    ordering = 1.0
    a = a.copy()
    L_dim = a.shape[-1]
    h = 1

    while h < L_dim:
        a2 = a.reshape(a.shape[:-1] + (L_dim // (2 * h), 2, h))
        x = a2[..., 0, :]
        y = a2[..., 1, :]

        t = x.copy()
        a2[..., 0, :] = 0.5 * (t + y)
        a2[..., 1, :] = 0.5 * (t - y) * ordering

        h *= 2

    return a


def compute_connected_ratio_delta(corr_delta, L, mode="2body"):
    """
    Computes the intensive connected correlation deviation sum anchored at site 0:
    \Delta \tilde{\gamma}(t) = \sum_{r=1}^{L-1} |\Delta G_{0,r}^c(t)|
    """
    # Single-site magnetization deviation \Delta m1
    m1_delta = corr_delta[1]

    # 1. Connected 2-body correlation deviations \Delta G_{0,r}^c
    g2_delta = np.zeros(L, dtype=np.float64)
    for r in range(1, L):
        mask_2 = 1 | (1 << r)
        # Note: since <\sigma_i^z> = 0 by Z2 symmetry throughout the quench, 
        # \Delta (m_i m_j) = \Delta G_{ij}. We subtract base corrections properly.
        g2_delta[r] = corr_delta[mask_2]

    gamma_2_delta = np.sum(np.abs(g2_delta[1:]))

    if mode == "2body":
        return gamma_2_delta

    # 2. Connected 4-body correlation deviations (Ursell functions)
    gamma_4_delta = 0.0
    for i in range(1, L - 2):
        for j in range(i + 1, L - 1):
            for k in range(j + 1, L):
                mask_4 = 1 | (1 << i) | (1 << j) | (1 << k)
                g4_delta_raw = corr_delta[mask_4]

                # Pairwise connected deviations using translational invariance
                wick_1 = g2_delta[i] * g2_delta[k - j]
                wick_2 = g2_delta[j] * g2_delta[k - i]
                wick_3 = g2_delta[k] * g2_delta[j - i]

                g4_delta_conn = g4_delta_raw - (wick_1 + wick_2 + wick_3)
                gamma_4_delta += np.abs(g4_delta_conn) / 4.0

    return gamma_2_delta,gamma_4_delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Delta-Connected Correlation Convergence Ratio from t=0.")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--mode", type=str, default="2body", choices=["2body", "2and4"],
                        help="Compute 2-body ('2body') or 2+4 body ('2and4') deviation sum.")

    args = parser.parse_args()

    gi = args.gi
    gf = args.gf
    angle = args.angle
    t_start = args.t_start
    dt = args.dt
    Ndt = args.Ndt
    mode = args.mode

    pbc = True
    add = "_pbc" if pbc else ""
    if angle != 0.0:
        add += f"_{angle}"

    output_dir = "wavefunctions_data" + add

    Ls = [8, 10, 12, 14, 16, 18, 20, 22]
    t_idx_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    times = t_start + np.array(t_idx_list) * dt

    C2 = np.zeros((len(Ls), len(t_idx_list)))
    C4 = np.zeros((len(Ls), len(t_idx_list)))
    colors = cm.viridis(np.linspace(0, 1, len(Ls)))

    fig, axes = plt.subplots(1, 1, figsize=(10, 5), squeeze=False)
    plt.suptitle(rf"Delta-Connected Ratio from $t=0$ ($g_i={gi},\;g_f={gf},\;\mathrm{{mode}}={mode}$)", fontsize=11)

    for idx_L, L in enumerate(Ls):
        nstates = 1 << L

        wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
        wf_file = os.path.join(output_dir, wf_base)
        if not os.path.exists(wf_file):
            raise FileNotFoundError(f"Missing wavefunction file: {wf_file}")

        psi_history = np.lib.format.open_memmap(
            wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
        )

        # 1. Base wavefunction and WHT at t = 0
        psi_0 = np.abs(psi_history[0])
        psi_0 /= np.sqrt(np.sum(psi_0 ** 2))
        prob_0 = psi_0 ** 2
        WHT_0 = Fast_Hadamard(prob_0)
        corr_0 = WHT_0 * nstates

        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            prob_t = psi_t ** 2

            WHT_t = Fast_Hadamard(prob_t)
            corr_t = WHT_t * nstates

            # Delta correlations relative to t = 0
            corr_delta = corr_t - corr_0

            
            aux= compute_connected_ratio_delta(corr_delta, L, mode=mode)
            C2[idx_L, c_idx] = aux[0]
            C4[idx_L, c_idx] = aux[1]
        axes[0, 0].plot(times, C2[idx_L], marker="markersize" if False else "o", markersize=4,
                        color=colors[idx_L], lw=1.6, label=f"C2,L={L}")
        
        axes[0, 0].plot(times, C4[idx_L]/L**(1.7), marker="markersize" if False else "o", markersize=4,linestyle="dashed",
                        color=colors[idx_L], lw=1.6, label=f"C4/L,L{L}")

    # Perturbative threshold line at \Delta\tilde{\gamma} = 1
    axes[0, 0].axhline(1.0, color="crimson", ls="--", lw=1.2, label=r"$\Delta\tilde{\gamma} = 1$ (Perturbative limit)")

    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0, 0].set_xlabel("time (t)")

    if mode == "2body":
        axes[0, 0].set_title(r"2-Body Deviation Ratio $\Delta\tilde{\gamma}_2(t) = \sum_{r=1}^{L-1} |\Delta G_{0,r}^c(t)|$")
        axes[0, 0].set_ylabel(r"$\Delta\tilde{\gamma}_2(t)$")
    else:
        axes[0, 0].set_title(r"2+4 Body Deviation Ratio $\Delta\tilde{\gamma}_{2+4}(t)$")
        axes[0, 0].set_ylabel(r"$\Delta\tilde{\gamma}_2(t) + \Delta\tilde{\gamma}_4(t)$")

    axes[0, 0].legend(fontsize=8, ncol=2)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"wht_conn_ratio_delta_{mode}_gi{gi:.2f}_gf{gf:.2f}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Delta-Connected Ratio plot to {plot_path}")
    plt.show()
