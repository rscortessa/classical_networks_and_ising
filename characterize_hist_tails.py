#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def analyze_tail_unbinned(x_finite, tail_quantile=0.95, thresholds=(1.0, 2.0, 3.0)):
    """
    Computes exact tail metrics from unbinned finite log10 flip ratios X.
    Uses |X| to exploit the exact s <-> s' inversion symmetry.
    """
    n_finite = len(x_finite)
    if n_finite < 10:
        return None

    abs_x = np.abs(x_finite)
    abs_x_sorted = np.sort(abs_x)

    # 1. Scale and Kurtosis
    std_x = np.std(x_finite)
    if std_x > 0:
        kurtosis_excess = np.mean((x_finite / std_x) ** 4) - 3.0
    else:
        kurtosis_excess = np.nan

    # 2. Quantiles of |X|
    q90, q95, q99, q999 = np.quantile(abs_x_sorted, [0.90, 0.95, 0.99, 0.999])
    max_abs_x = abs_x_sorted[-1]
    q_ratio = q999 / q95 if q95 > 0 else np.nan

    # 3. Tail mass above fixed decade thresholds (|X| > 1, 2, 3 -> ratio > 10, 100, 1000)
    tail_probs = {
        th: float(np.sum(abs_x_sorted > th)) / n_finite for th in thresholds
    }

    # 4. Hill / POT Estimator & Log-CCDF Fit for power-law exponent \alpha:
    # P(|X| > x) ~ 10^{-\alpha x}  <=>  P(R) ~ R^{-(1 + \alpha)}
    x_min = np.quantile(abs_x_sorted, tail_quantile)
    tail_samples = abs_x_sorted[abs_x_sorted > x_min]

    if len(tail_samples) >= 5:
        mean_excess = np.mean(tail_samples - x_min)
        alpha_hill = 1.0 / (np.log(10.0) * mean_excess) if mean_excess > 0 else np.nan
        alpha_hill_se = alpha_hill / np.sqrt(len(tail_samples))

        # Linear regression on log10(CCDF) vs x in the tail (excluding the single extreme point)
        idx_start = np.searchsorted(abs_x_sorted, x_min, side="right")
        idx_end = max(idx_start + 2, n_finite - 1)
        x_fit = abs_x_sorted[idx_start:idx_end]
        ccdf_fit = (n_finite - np.arange(idx_start, idx_end)) / float(n_finite)

        if len(x_fit) >= 3 and np.ptp(x_fit) > 1e-12:
            slope, intercept = np.polyfit(x_fit, np.log10(ccdf_fit), 1)
            alpha_ccdf = -slope
        else:
            alpha_ccdf, intercept = np.nan, np.nan
    else:
        alpha_hill, alpha_hill_se, alpha_ccdf, intercept = np.nan, np.nan, np.nan, np.nan

    return {
        "std": std_x,
        "kurtosis_excess": kurtosis_excess,
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "q999": q999,
        "max_abs": max_abs_x,
        "q_ratio_999_95": q_ratio,
        "tail_probs": tail_probs,
        "alpha_hill": alpha_hill,
        "alpha_hill_se": alpha_hill_se,
        "alpha_ccdf": alpha_ccdf,
        "fit_intercept": intercept,
        "x_min": x_min,
    }


def analyze_tail_from_npz(counts, bin_edges, tail_quantile=0.95, thresholds=(1.0, 2.0, 3.0)):
    """
    Fallback estimator that computes the same tail metrics solely from the saved
    hist_data_L...npz files (counts, bin_edges) when the raw memmap is unavailable.
    """
    if len(counts) == 0 or len(bin_edges) < 2:
        return None

    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    probs = counts * widths
    total_p = np.sum(probs)
    if total_p <= 0:
        return None
    probs /= total_p

    # Moments
    mean_x = np.sum(centers * probs)
    var_x = np.sum(((centers - mean_x) ** 2) * probs)
    std_x = np.sqrt(var_x)
    kurtosis_excess = (
        np.sum(((centers - mean_x) / std_x) ** 4 * probs) - 3.0 if std_x > 0 else np.nan
    )

    # Fold distribution to |X|
    abs_centers = np.abs(centers)
    sort_idx = np.argsort(abs_centers)
    abs_c_sorted = abs_centers[sort_idx]
    probs_sorted = probs[sort_idx]
    cdf_abs = np.cumsum(probs_sorted)
    ccdf_abs = np.clip(1.0 - cdf_abs + probs_sorted, 1e-15, 1.0)

    q90, q95, q99, q999 = np.interp([0.90, 0.95, 0.99, 0.999], cdf_abs, abs_c_sorted)
    nonzero_bins = abs_c_sorted[probs_sorted > 0]
    max_abs_x = nonzero_bins[-1] if len(nonzero_bins) > 0 else np.nan
    q_ratio = q999 / q95 if q95 > 0 else np.nan

    tail_probs = {
        th: float(np.sum(probs_sorted[abs_c_sorted > th])) for th in thresholds
    }

    x_min = np.interp(tail_quantile, cdf_abs, abs_c_sorted)
    mask_tail = (abs_c_sorted > x_min) & (probs_sorted > 0)

    if np.sum(mask_tail) >= 3:
        p_tail = probs_sorted[mask_tail]
        x_tail = abs_c_sorted[mask_tail]
        mean_excess = np.sum((x_tail - x_min) * p_tail) / np.sum(p_tail)
        alpha_hill = 1.0 / (np.log(10.0) * mean_excess) if mean_excess > 0 else np.nan

        slope, intercept = np.polyfit(x_tail, np.log10(ccdf_abs[mask_tail]), 1)
        alpha_ccdf = -slope
    else:
        alpha_hill, alpha_ccdf, intercept = np.nan, np.nan, np.nan

    return {
        "std": std_x,
        "kurtosis_excess": kurtosis_excess,
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "q999": q999,
        "max_abs": max_abs_x,
        "q_ratio_999_95": q_ratio,
        "tail_probs": tail_probs,
        "alpha_hill": alpha_hill,
        "alpha_hill_se": np.nan,
        "alpha_ccdf": alpha_ccdf,
        "fit_intercept": intercept,
        "x_min": x_min,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Characterize the tails of log10(|psi(s')/psi(s)|) histograms."
    )
    parser.add_argument("--Ls", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--tail_quantile", type=float, default=0.95,
                        help="Quantile threshold x_min for Hill and CCDF tail exponent fits.")
    parser.add_argument("--force_npz", action="store_true",
                        help="Force reading from hist_data_*.npz even if logflip_*.npy exists.")
    parser.add_argument("--t_snapshots", type=int, nargs="+", default=[0,40,45,50,60,65,70,75,80],
                        help="Time indices t_idx to plot on the CCDF tail-fit comparison figure.")
    args = parser.parse_args()

    pbc = True
    output_dir = "wavefunctions_data" + ("_pbc" if pbc else "")
    Ls = args.Ls
    Ndt = args.Ndt
    times = args.t_start + np.arange(Ndt + 1) * args.dt

    # Storage arrays: shape (len(Ls), Ndt + 1)
    alpha_hill_arr = np.full((len(Ls), Ndt + 1), np.nan)
    alpha_ccdf_arr = np.full((len(Ls), Ndt + 1), np.nan)
    kurtosis_arr = np.full((len(Ls), Ndt + 1), np.nan)
    q99_arr = np.full((len(Ls), Ndt + 1), np.nan)
    q999_arr = np.full((len(Ls), Ndt + 1), np.nan)
    max_abs_arr = np.full((len(Ls), Ndt + 1), np.nan)
    prob_gt1_arr = np.full((len(Ls), Ndt + 1), np.nan)
    prob_gt2_arr = np.full((len(Ls), Ndt + 1), np.nan)
    singular_frac_arr = np.full((len(Ls), Ndt + 1), np.nan)

    # Store CCDF curves at selected time snapshots for the largest available L
    ccdf_snapshots = {}
    largest_available_L = None

    for idx_L, L in enumerate(Ls):
        nstates = 1 << L
        memmap_path = os.path.join(output_dir, f"logflip_L{L}_gi{args.gi:.2f}_gf{args.gf:.2f}.npy")
        use_memmap = os.path.exists(memmap_path) and not args.force_npz

        if use_memmap:
            print(f"[L={L}] Reading exact unbinned memmap: {memmap_path}")
            log_flip_mm = np.lib.format.open_memmap(
                memmap_path, mode="r", dtype=np.float64, shape=(Ndt + 1, nstates)
            )
            largest_available_L = L
        else:
            print(f"[L={L}] Memmap not found or skipped; reading binned hist_data_L{L}_*.npz")

        for idx_t in range(Ndt + 1):
            if use_memmap:
                row = log_flip_mm[idx_t]
                mask = np.isfinite(row)
                singular_frac_arr[idx_L, idx_t] = 1.0 - (np.sum(mask) / float(nstates))
                x_finite = row[mask]
                res = analyze_tail_unbinned(x_finite, tail_quantile=args.tail_quantile)

                # Save snapshot CCDF for the largest L
                if idx_t in args.t_snapshots and res is not None:
                    abs_sorted = np.sort(np.abs(x_finite))
                    n_f = len(abs_sorted)
                    # Subsample points logarithmically in the tail to keep plotting fast
                    ccdf_y = (n_f - np.arange(n_f)) / float(n_f)
                    step = max(1, n_f // 2000)
                    idx_keep = np.unique(
                        np.concatenate([np.arange(0, n_f, step), np.arange(max(0, n_f - 500), n_f)])
                    )
                    ccdf_snapshots[(L, idx_t)] = (
                        abs_sorted[idx_keep],
                        ccdf_y[idx_keep],
                        res["x_min"],
                        res["alpha_ccdf"],
                        res["fit_intercept"],
                    )
            else:
                npz_path = os.path.join(
                    output_dir, f"hist_data_L{L}_t{idx_t}_gi{args.gi:.2f}_gf{args.gf:.2f}.npz"
                )
                if not os.path.exists(npz_path):
                    continue
                data = np.load(npz_path)
                res = analyze_tail_from_npz(
                    data["counts"], data["bin_edges"], tail_quantile=args.tail_quantile
                )
                largest_available_L = L

            if res is not None:
                alpha_hill_arr[idx_L, idx_t] = res["alpha_hill"]
                alpha_ccdf_arr[idx_L, idx_t] = res["alpha_ccdf"]
                kurtosis_arr[idx_L, idx_t] = res["kurtosis_excess"]
                q99_arr[idx_L, idx_t] = res["q99"]
                q999_arr[idx_L, idx_t] = res["q999"]
                max_abs_arr[idx_L, idx_t] = res["max_abs"]
                prob_gt1_arr[idx_L, idx_t] = res["tail_probs"][1.0]
                prob_gt2_arr[idx_L, idx_t] = res["tail_probs"][2.0]

        if use_memmap:
            del log_flip_mm

    # Save extracted tail summary metrics
    out_npz = os.path.join(output_dir, f"tail_metrics_gi{args.gi:.2f}_gf{args.gf:.2f}.npz")
    np.savez(
        out_npz,
        Ls=np.array(Ls),
        times=times,
        alpha_hill=alpha_hill_arr,
        alpha_ccdf=alpha_ccdf_arr,
        kurtosis_excess=kurtosis_arr,
        q99=q99_arr,
        q999=q999_arr,
        max_abs=max_abs_arr,
        prob_gt1=prob_gt1_arr,
        prob_gt2=prob_gt2_arr,
        singular_frac=singular_frac_arr,
    )
    print(f"Saved tail metrics summary to: {out_npz}")

    # =========================================================================
    # Plot 1: 4-Panel Dynamical Tail Characterization Across System Sizes L
    # =========================================================================
    colors = cm.viridis(np.linspace(0, 1, len(Ls)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(
        rf"Tail Characterization of $X = \log_{{10}}(|\psi(s^{{(i)}})/\psi(s)|)$ ($g_i={args.gi}, g_f={args.gf}$)",
        fontsize=13,
    )

    for idx_L, L in enumerate(Ls):
        if np.all(np.isnan(q99_arr[idx_L])):
            continue
        # Panel (0,0): Hill Tail Exponent \alpha
        axes[0, 0].plot(times, alpha_hill_arr[idx_L], color=colors[idx_L], lw=1.6, label=f"L={L}")
        # Panel (0,1): 99th Percentile (solid) and 99.9th Percentile (dashed)
        axes[0, 1].plot(times, q99_arr[idx_L], color=colors[idx_L], lw=1.6, label=f"L={L} ($Q_{{99}}$)")
        axes[0, 1].plot(times, q999_arr[idx_L], color=colors[idx_L], lw=1.1, ls="--")
        # Panel (1,0): Tail probability P(|X| > 1.0)
        axes[1, 0].plot(times, prob_gt1_arr[idx_L], color=colors[idx_L], lw=1.6, label=f"L={L}")
        # Panel (1,1): Excess Kurtosis \kappa_4
        axes[1, 1].plot(times, kurtosis_arr[idx_L], color=colors[idx_L], lw=1.6, label=f"L={L}")

    axes[0, 0].axhline(1.0, color="crimson", ls=":", lw=1.2, label=r"$\alpha=1$ (Cauchy / Linear Node)")
    axes[0, 0].set_ylabel(r"Tail Exponent $\hat{\alpha}_{\mathrm{Hill}}$ ($P(R) \propto R^{-1-\alpha}$)")
    axes[0, 0].set_title(rf"Power-Law Tail Exponent (Top {100*(1-args.tail_quantile):.0f}% Tail)")
    axes[0, 0].grid(True, ls="--", alpha=0.4)
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].set_ylabel(r"Quantiles $Q_{99}$ (solid), $Q_{99.9}$ (dashed)")
    axes[0, 1].set_title(r"Extreme Quantiles of $|X|$")
    axes[0, 1].grid(True, ls="--", alpha=0.4)

    axes[1, 0].set_xlabel("Time $t$")
    axes[1, 0].set_ylabel(r"Tail Mass $P(|X| > 1)$")
    axes[1, 0].set_title(r"Fraction of States with $|\psi(s')/\psi(s)| \notin [0.1, 10]$")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.4)

    axes[1, 1].axhline(3.0, color="crimson", ls=":", lw=1.2, label="Laplace ($\kappa_4 = 3$)")
    axes[1, 1].axhline(0.0, color="gray", ls="--", lw=1.0, label="Gaussian ($\kappa_4 = 0$)")
    axes[1, 1].set_xlabel("Time $t$")
    axes[1, 1].set_ylabel(r"Excess Kurtosis $\kappa_4$")
    axes[1, 1].set_title(r"Excess Kurtosis $\langle X^4 \rangle / \langle X^2 \rangle^2 - 3$")
    axes[1, 1].grid(True, ls="--", alpha=0.4)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    summary_plot = os.path.join(output_dir, f"tail_diagnostics_gi{args.gi:.2f}_gf{args.gf:.2f}.png")
    plt.savefig(summary_plot, dpi=300)
    print(f"Saved 4-panel tail diagnostic plot to: {summary_plot}")
    plt.close(fig)

    # =========================================================================
    # Plot 2: Complementary CDF (Survival Function) & Linear Tail Fits
    # =========================================================================
    if largest_available_L is not None and len(ccdf_snapshots) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 5.5))
        snap_colors = cm.plasma(np.linspace(0, 0.9, len(args.t_snapshots)))

        for c_i, t_idx in enumerate(args.t_snapshots):
            key = (largest_available_L, t_idx)
            if key not in ccdf_snapshots:
                continue
            x_vals, ccdf_vals, x_min, alpha_fit, intercept = ccdf_snapshots[key]
            t_val = args.t_start + t_idx * args.dt

            ax2.plot(
                x_vals,
                ccdf_vals,
                color=snap_colors[c_i],
                lw=1.8,
                label=rf"$t={t_val:.2f}$ ($\hat{{\alpha}}={alpha_fit:.2f}$)",
            )
            # Overlay the fitted exponential tail line 10^{-\alpha x + intercept}
            if np.isfinite(alpha_fit):
                x_line = np.linspace(x_min, x_vals[-1], 50)
                ax2.plot(
                    x_line,
                    10.0 ** (intercept - alpha_fit * x_line),
                    color=snap_colors[c_i],
                    ls="--",
                    lw=1.2,
                )

        ax2.set_yscale("log")
        ax2.set_xlabel(r"$x = |\log_{10}(|\psi(s^{(i)})/\psi(s)|)|$")
        ax2.set_ylabel(r"Complementary CDF $P(|X| > x)$")
        ax2.set_title(
            rf"Tail Survival Function & Power-Law Fits ($L={largest_available_L}, g_i={args.gi}, g_f={args.gf}$)"
        )
        ax2.grid(True, which="both", ls="--", alpha=0.4)
        ax2.legend(fontsize=9)
        plt.tight_layout()

        ccdf_plot = os.path.join(
            output_dir, f"tail_ccdf_fits_L{largest_available_L}_gi{args.gi:.2f}_gf{args.gf:.2f}.png"
        )
        plt.savefig(ccdf_plot, dpi=300)
        print(f"Saved CCDF tail-fit plot to: {ccdf_plot}")
        plt.close(fig2)
