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


def compute_connected_ratio(corr, L, mode="2body"):
    """
    Computes the intensive connected correlation sum anchored at site 0
    using translational invariance on a periodic chain of length L.
    
    Parameters
    ----------
    corr : np.ndarray of shape (2^L,)
        Unnormalized expectation values <\chi_A> = 2^L * WHT(P).
    L : int
        System size.
    mode : str
        - "2body": \tilde{\gamma}_2(t) = \sum_{r=1}^{L-1} |G_{0,r}^c(t)|
        - "2and4": \tilde{\gamma}_2(t) + \tilde{\gamma}_4(t) including 4-point connected Ursell functions
    """
    # Single-site magnetization <\sigma_i^z> (vanishes for Z2-symmetric states, kept for generality)
    m1 = corr[1]

    # 1. Connected 2-body correlations G_{0,r}^c = <\sigma_0^z \sigma_r^z> - <\sigma_0^z><\sigma_r^z>
    g2_conn = np.zeros(L, dtype=np.float64)
    for r in range(1, L):
        mask_2 = 1 | (1 << r)
        g2_conn[r] = corr[mask_2] - (m1 ** 2)

    gamma_2 = np.sum(np.abs(g2_conn[1:]))

    if mode == "2body":
        return gamma_2

    # 2. Connected 4-body correlations (Ursell functions) anchored at site 0:
    # G_{0,i,j,k}^c = <\sigma_0 \sigma_i \sigma_j \sigma_k>
    #                 - G_{0,i}^c G_{j,k}^c - G_{0,j}^c G_{i,k}^c - G_{0,k}^c G_{i,j}^c
    # (Divided by 4 so each translationally distinct 4-spin cluster is counted once per site)
    gamma_4 = 0.0
    for i in range(1, L - 2):
        for j in range(i + 1, L - 1):
            for k in range(j + 1, L):
                mask_4 = 1 | (1 << i) | (1 << j) | (1 << k)
                g4_raw = corr[mask_4]

                # Pairwise connected products using translational invariance G_{a,b}^c = g2_conn[|b-a|]
                wick_1 = g2_conn[i] * g2_conn[k - j]
                wick_2 = g2_conn[j] * g2_conn[k - i]
                wick_3 = g2_conn[k] * g2_conn[j - i]

                g4_conn = g4_raw - (wick_1 + wick_2 + wick_3)
                gamma_4 += np.abs(g4_conn) / 4.0

    return gamma_2 + gamma_4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Connected Correlation Convergence Ratio of WHT expansion.")
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--mode", type=str, default="2body", choices=["2body", "2and4"],
                        help="Compute 2-body connected sum ('2body') or 2-body + 4-body connected cumulant sum ('2and4').")

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

    weights = np.zeros((len(Ls), len(t_idx_list)))
    colors = cm.viridis(np.linspace(0, 1, len(Ls)))

    fig, axes = plt.subplots(1, 1, figsize=(10, 5), squeeze=False)
    plt.suptitle(rf"Connected Correlation Ratio ($g_i={gi},\;g_f={gf},\;\mathrm{{mode}}={mode}$)", fontsize=11)

    for idx_L, L in enumerate(Ls):
        nstates = 1 << L  # Properly update 2^L for each chain length L

        wf_base = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
        wf_file = os.path.join(output_dir, wf_base)
        if not os.path.exists(wf_file):
            raise FileNotFoundError(f"Missing wavefunction file: {wf_file}")

        psi_history = np.lib.format.open_memmap(
            wf_file, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
        )
        nom = psi_history[0]**2
        for c_idx, t_idx in enumerate(t_idx_list):
            psi_t = np.abs(psi_history[t_idx])
            psi_t /= np.sqrt(np.sum(psi_t ** 2))
            prob_t = psi_t ** 2

            # Fast_Hadamard returns \hat{P}(A) = 2^{-L} <\chi_A>
            WHT = Fast_Hadamard(prob_t)
            WHT_0 = Fast_Hadamard(nom)
            # Multiply by 2^L to obtain physical correlations <\chi_A> \in [-1, 1]
            corr = WHT * nstates-WHT_0 * nstates

            weights[idx_L, c_idx] = compute_connected_ratio(corr, L, mode=mode)

        axes[0, 0].plot(times, weights[idx_L], marker="o", markersize=4,
                        color=colors[idx_L], lw=1.6, label=f"L={L}")

    # Reference line at \tilde{\gamma} = 1 (perturbative convergence threshold)
    axes[0, 0].axhline(1.0, color="crimson", ls="--", lw=1.2, label=r"$\tilde{\gamma} = 1$")

    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0, 0].set_xlabel("time (t)")

    if mode == "2body":
        axes[0, 0].set_title(r"Local 2-Body Connected Convergence Ratio $\tilde{\gamma}_2(t)$")
        axes[0, 0].set_ylabel(r"$\tilde{\gamma}_2(t) = \sum_{r=1}^{L-1} |\langle \sigma_0^z \sigma_r^z \rangle_c|$")
    else:
        axes[0, 0].set_title(r"Connected 2-Body + 4-Body Cumulant Ratio $\tilde{\gamma}_{2+4}(t)$")
        axes[0, 0].set_ylabel(r"$\tilde{\gamma}_2(t) + \tilde{\gamma}_4(t)$")

    axes[0, 0].legend(fontsize=8, ncol=2)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"wht_conn_ratio_{mode}_gi{gi:.2f}_gf{gf:.2f}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Connected Correlation Ratio plot to {plot_path}")
    plt.show()
