#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import netket as nk

# 1. Restrict threads BEFORE multi-threaded backends spin up
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"


def Fast_Hadamard(a):
    """
    Computes Walsh-Hadamard Transform:
    f_hat(A) = (1 / 2^L) * sum_s f(s) \chi_A(s)
    Inverse is: f(s) = 2^L * Fast_Hadamard(f_hat)
    """
    ordering = 1.0
    a = a.copy()
    dim = a.shape[-1]
    h = 1

    while h < dim:
        a2 = a.reshape(a.shape[:-1] + (dim // (2 * h), 2, h))
        x = a2[..., 0, :]
        y = a2[..., 1, :]

        t = x.copy()
        a2[..., 0, :] = 0.5 * (t + y)
        a2[..., 1, :] = 0.5 * (t - y) * ordering

        h *= 2

    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot KL divergence vs truncation order.")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--t_idx", type=int, default=50, help="Fixed time slice index for all-L comparison plot.")
    parser.add_argument("--Ls", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])

    args, _ = parser.parse_known_args()
    gi = args.gi
    gf = args.gf
    angle = args.angle
    t_start = args.t_start
    dt = args.dt
    Ndt = args.Ndt
    t_idx_target = args.t_idx
    Ls = args.Ls

    pbc = True
    add = "_pbc" if pbc else ""
    if angle != 0.0:
        add += f"_{angle}"

    output_dir = "wavefunctions_data" + add
    os.makedirs(output_dir, exist_ok=True)

    times = t_start + np.arange(Ndt + 1) * dt
    KL_dict = {}

    for L in Ls:
        nstates = 1 << L
        hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)
        states = hi.all_states()  # Shape: (2^L, L)
        # Cluster orders: count number of down spins (spin == -1 or spin == 1 depending on convention)
        mask_weights = np.sum(states == 1, axis=-1)
        orders = np.array([i for i in range(0, L + 1, 2)])

        base_name = f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
        filename = os.path.join(output_dir, base_name)

        if os.path.exists(filename):
            psi_history = np.lib.format.open_memmap(
                filename, mode='r', dtype=np.complex128, shape=(Ndt + 1, nstates)
            )
            print(f"[L={L}] Loaded wavefunction memmap: shape {psi_history.shape}")
        else:
            print(f"[L={L}] Warning: {filename} not found. Using normalized dummy state.")
            psi_history = np.ones((Ndt + 1, nstates), dtype=np.complex128) / np.sqrt(nstates)

        # Probabilities P(s, t) = |\psi(s, t)|^2
        prob_t = np.abs(psi_history)**2
        # Ensure row-wise normalization
        prob_t /= np.sum(prob_t, axis=-1, keepdims=True)

        # Expectation values \hat{P}(A) = <\chi_A> / 2^L
        av_sz = Fast_Hadamard(prob_t)

        # Exact effective couplings: C_A = WHT(ln P)
        # Avoid log(0) by clipping to numerical floor
        log_prob = np.log(np.clip(prob_t, 1e-15, 1.0))
        exact_couplings = Fast_Hadamard(log_prob)

        # Storage for KL(t, order)
        KL = np.zeros((Ndt + 1, len(orders)))

        for idx_order, order in enumerate(orders):
            mask_order = (mask_weights <= order)

            coeffs_order = np.zeros_like(exact_couplings)
            coeffs_order[:, mask_order] = exact_couplings[:, mask_order]

            # Reconstruct unnormalized log-probabilities: ln P_trunc(s) = 2^L * WHT(C_trunc)
            unnorm_log_P_trunc = Fast_Hadamard(coeffs_order) * nstates

            # Partition function: ln Z_trunc = logsumexp(unnorm_log_P_trunc)
            max_log = np.max(unnorm_log_P_trunc, axis=-1, keepdims=True)
            log_Z_trunc = max_log.squeeze(-1) + np.log(
                np.sum(np.exp(unnorm_log_P_trunc - max_log), axis=-1)
            )

            # D_KL(P || P_trunc) = \sum_s P(s) [ln P(s) - ln P_trunc(s) + ln Z_trunc]
            # In WHT space: \sum_s P(s) ln P_trunc(s) = nstates * \sum_A av_sz(A) * coeffs_order(A)
            cross_term = nstates * np.sum(av_sz * coeffs_order, axis=-1)
            entropy_term = np.sum(prob_t * log_prob, axis=-1)

            KL[:, idx_order] = entropy_term - cross_term + log_Z_trunc

        KL_dict[L] = {"orders": orders, "KL": KL}

    # =========================================================================
    # PLOT 1: All Times, Single L (One figure per L)
    # =========================================================================
    t_plot_indices = np.linspace(0, Ndt, 15, dtype=int)
    time_cmap = cm.viridis
    norm_t = Normalize(vmin=times[0], vmax=times[-1])

    for L in Ls:
        fig, ax = plt.subplots(figsize=(7, 5))
        orders = KL_dict[L]["orders"]
        KL_data = KL_dict[L]["KL"]

        for tt in t_plot_indices:
            t_val = times[tt]
            color = time_cmap(norm_t(t_val))
            ax.plot(
                orders,
                KL_data[tt, :],
                marker="o",
                markersize=4,
                lw=1.5,
                color=color,
                label=rf"$t = {t_val:.2f}$" if tt in t_plot_indices[::4] else None,
            )

        ax.set_xlabel(r"Truncation Order $k$", fontsize=11)
        ax.set_ylabel(r"$D_{\mathrm{KL}}(P(t) \parallel P_{\mathrm{trunc}}^{(k)}(t))$", fontsize=11)
        ax.set_title(rf"$D_{{\mathrm{{KL}}}}$ vs Truncation Order across Time ($L={L}$)", fontsize=12)
        ax.set_xticks(orders)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_yscale("log")

        # Colorbar for times
        sm = plt.cm.ScalarMappable(cmap=time_cmap, norm=norm_t)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Time $t$", fontsize=10)

        plt.tight_layout()
        plot1_path = os.path.join(output_dir, f"KL_vs_order_all_times_L{L}.png")
        plt.savefig(plot1_path, dpi=300)
        plt.close(fig)
        print(f"Saved Plot 1 (all times, L={L}) to: {plot1_path}")

    # =========================================================================
    # PLOT 2: Single Time, All Ls (One unified figure)
    # =========================================================================
    t_fixed = times[t_idx_target]
    fig, (ax_raw, ax_scaled) = plt.subplots(1, 2, figsize=(14, 5))
    L_cmap = cm.plasma
    norm_L = Normalize(vmin=min(Ls), vmax=max(Ls))

    for L in Ls:
        orders = KL_dict[L]["orders"]
        kl_t = KL_dict[L]["KL"][t_idx_target, :]
        color = L_cmap(norm_L(L))

        # (Left) Raw Order k
        ax_raw.plot(
            orders,
            kl_t,
            marker="o",
            markersize=5,
            lw=1.8,
            color=color,
            label=f"$L={L}$",
        )

        # (Right) Normalized Order k / L and intensive KL density D_KL / L
        ax_scaled.plot(
            orders,
            kl_t / L ,
            marker="s",
            markersize=4,
            lw=1.8,
            color=color,
            label=f"$L={L}$",
        )

    ax_raw.set_xlabel(r"Truncation Order $k$", fontsize=11)
    ax_raw.set_ylabel(r"$D_{\mathrm{KL}}(P \parallel P_{\mathrm{trunc}}^{(k)})$", fontsize=11)
    ax_raw.set_title(rf"Raw Truncation at $t={t_fixed:.2f}$", fontsize=12)
    ax_raw.set_yscale("log")
    ax_raw.grid(True, which="both", ls="--", alpha=0.4)
    ax_raw.legend(fontsize=9)

    ax_scaled.set_xlabel(r"Relative Order $kL$", fontsize=11)
    ax_scaled.set_ylabel(r"$D_{\mathrm{KL}} / L$ (Density)", fontsize=11)
    ax_scaled.set_title(rf"Finite-Size Collapse ($t={t_fixed:.2f}$)", fontsize=12)
    ax_scaled.set_yscale("log")
    ax_scaled.grid(True, which="both", ls="--", alpha=0.4)
    ax_scaled.legend(fontsize=9)

    plt.suptitle(
        rf"Truncation Error across System Sizes $L$ ($g_i={gi},\; g_f={gf},\; t={t_fixed:.2f}$)",
        fontsize=13,
    )
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, f"KL_vs_order_all_Ls_t{t_idx_target}.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close(fig)
    print(f"Saved Plot 2 (single time, all Ls) to: {plot2_path}")
