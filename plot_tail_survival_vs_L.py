#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_survival_from_memmap(memmap_path, idx_t, L, tail_quantile=0.95):
    """
    Loads time slice idx_t from the exact unbinned memmap logflip_L...npy
    and returns subsampled (x, CCDF(x)) arrays along with the fitted tail exponent.
    """
    nstates = 1 << L
    mm = np.lib.format.open_memmap(
        memmap_path, mode="r", dtype=np.float64, shape=(idx_t + 1, nstates)
    )
    row = mm[idx_t]
    x_finite = row[np.isfinite(row)]
    del mm

    n_f = len(x_finite)
    if n_f < 10:
        return None

    abs_x = np.sort(np.abs(x_finite))
    # Complementary CDF: P(|X| >= x_k)
    ccdf = (n_f - np.arange(n_f)) / float(n_f)

    # Hill / POT estimator above tail_quantile
    x_min = np.quantile(abs_x, tail_quantile)
    tail_vals = abs_x[abs_x > x_min]
    if len(tail_vals) >= 5:
        mean_excess = np.mean(tail_vals - x_min)
        alpha_hill = 1.0 / (np.log(10.0) * mean_excess) if mean_excess > 0 else np.nan
    else:
        alpha_hill = np.nan

    # Subsample the bulk while keeping the extreme 2000 tail points exact
    # so plotting L=20 (1,048,576 states) remains fast and exact in the tail
    if n_f > 4000:
        step = max(1, (n_f - 2000) // 2000)
        idx_keep = np.unique(
            np.concatenate([np.arange(0, n_f - 2000, step), np.arange(n_f - 2000, n_f)])
        )
        abs_x = abs_x[idx_keep]
        ccdf = ccdf[idx_keep]

    return abs_x, ccdf, alpha_hill, x_min


def get_survival_from_npz(npz_path, tail_quantile=0.95):
    """
    Reconstructs (x, CCDF(x)) from binned hist_data_L...npz if the memmap is absent.
    """
    data = np.load(npz_path)
    counts = data["counts"]
    bin_edges = data["bin_edges"]
    if len(counts) == 0 or len(bin_edges) < 2:
        return None

    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    probs = counts * widths
    total_p = np.sum(probs)
    if total_p <= 0:
        return None
    probs /= total_p

    # Fold around X = 0 to obtain |X|

    abs_centers = np.abs(centers)
    sort_idx = np.argsort(abs_centers)
    abs_c = abs_centers[sort_idx]
    probs_s = probs[sort_idx]

    # Mask out empty bins in the tail so log-scale plotting is clean
    
    nonzero = probs_s > 0
    abs_c = abs_c[nonzero]
    probs_s = probs_s[nonzero]

    cdf = np.cumsum(probs_s)
    ccdf = np.clip(1.0 - cdf + probs_s, 1e-15, 1.0)

    x_min = np.interp(tail_quantile, cdf, abs_c)
    mask_tail = abs_c > x_min

    if np.sum(mask_tail) >= 3:
        p_tail = probs_s[mask_tail]
        x_tail = abs_c[mask_tail]
        mean_excess = np.sum((x_tail - x_min) * p_tail) / np.sum(p_tail)
        alpha_hill = 1.0 / (np.log(10.0) * mean_excess) if mean_excess > 0 else np.nan
    else:
        alpha_hill = np.nan

    return abs_c, ccdf, alpha_hill, x_min


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the tail survival function P(|X| > x) across system sizes L at fixed time t."
    )
    parser.add_argument("--Ls", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--t_target", type=float, default=1.00, help="Target physical time t (default: 1.00)")
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--tail_quantile", type=float, default=0.95)
    parser.add_argument("--force_npz", action="store_true",
                        help="Read from hist_data_*.npz even if logflip_*.npy exists.")
    args = parser.parse_args()

    idx_t = int( np.round((args.t_target - args.t_start) / args.dt) )
    t_actual = args.t_start + idx_t * args.dt

    pbc = True
    output_dir = "wavefunctions_data" + ("_pbc" if pbc else "")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = cm.viridis(np.linspace(0, 0.95, len(args.Ls)))

    max_x_plotted = 0.0
    ref_x_min = None

    for idx_L, L in enumerate(args.Ls):
        memmap_path = os.path.join(output_dir, f"logflip_L{L}_gi{args.gi:.2f}_gf{args.gf:.2f}.npy")
        npz_path = os.path.join(output_dir, f"hist_data_L{L}_t{idx_t}_gi{args.gi:.2f}_gf{args.gf:.2f}.npz")

        res = None
        if os.path.exists(memmap_path) and not args.force_npz:
            print(f"[L={L}, t={t_actual:.2f}] Loading unbinned memmap: {memmap_path}")
            res = get_survival_from_memmap(memmap_path, idx_t, L, tail_quantile=args.tail_quantile)
        elif os.path.exists(npz_path):
            print(f"[L={L}, t={t_actual:.2f}] Loading binned histogram: {npz_path}")
            res = get_survival_from_npz(npz_path, tail_quantile=args.tail_quantile)
        else:
            print(f"[L={L}, t={t_actual:.2f}] File not found, skipping L={L}.")
            continue

        if res is None:
            continue

        abs_x, ccdf, alpha_hill, x_min = res
        max_x_plotted = max(max_x_plotted, abs_x[-1])
        ref_x_min = x_min

        label_str = (
            rf"$L={L}$ ($\hat{{\alpha}}={alpha_hill:.2f}$)"
            if np.isfinite(alpha_hill)
            else rf"$L={L}$"
        )
        ax.plot(abs_x, ccdf, color=colors[idx_L], lw=1.8, label=label_str)

    # Overlay reference power-law slopes 10^{-\alpha x} in the tail region
    if ref_x_min is not None and max_x_plotted > ref_x_min:
        x_ref = np.linspace(ref_x_min, max_x_plotted, 100)
        y_anchor = 1.0 - args.tail_quantile  # Anchored at P(|X| > x_min) = 0.05

        #ax.plot(
        #    x_ref,
        #    y_anchor * (10.0 ** (-1.0 * (x_ref - ref_x_min))),
        #    color="crimson",
        #    ls="--",
        #    lw=1.4,
        #    label=r"Ref slope $\alpha = 1$ ($10^{-x}$, Cauchy/Real node)",
        #)
        #ax.plot(
        #    x_ref,
        #    y_anchor * (10.0 ** (-2.0 * (x_ref - ref_x_min))),
        #    color="black",
        #    ls=":",
        #    lw=1.5,
        #    label=r"Ref slope $\alpha = 2$ ($10^{-2x}$, Complex node)",
        #)

    ax.set_yscale("log")
    ax.set_xlabel(r"Threshold $x = |\log_{10}(|\psi(s^{(i)})/\psi(s)|)|$", fontsize=11)
    ax.set_ylabel(r"Tail Survival Function $P(|X| > x)$", fontsize=11)
    ax.set_title(
        rf"System-Size Scaling of Tail Survival Function at $t = {t_actual:.2f}$ ($g_i={args.gi}, g_f={args.gf}$)",
        fontsize=12,
    )
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    out_plot = os.path.join(
        output_dir, f"tail_survival_vs_L_t{idx_t}_gi{args.gi:.2f}_gf{args.gf:.2f}.png"
    )
    plt.savefig(out_plot, dpi=300)
    print(f"Saved survival function comparison plot to: {out_plot}")
    plt.show()
