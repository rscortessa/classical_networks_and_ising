"""
analyze_KL_scaling.py
=====================
Quantifies how the separation between KL curves at different times depends on
system size L.

Three complementary analyses are produced:

  Figure 1 — Normalized overlay
      KL vs n_c / N_total for every available L and time t.
      If the physics is size-independent the curves should collapse.

  Figure 2 — Vertical separation vs L
      At fixed normalized positions n_c / N_total = {0.1, 0.3, 0.5, 0.7},
      plot the ratio KL(t_max) / KL(t_min) as a function of L.
      A flat curve means the separation is size-independent; a growing
      (falling) curve means larger systems are more (less) separated.

  Figure 3 — Horizontal cost (n_c needed) vs L
      For each time t and each L, find the normalized n_c / N_total where
      the KL first drops below a threshold (default 1e-3).
      Plotting this vs L reveals whether longer times incur a relatively
      larger cost as L grows, and allows an estimate of the exponent.

Usage
-----
  python analyze_KL_scaling.py \\
      --L_list 10 12 14 16 18 \\
      --gi 1.5 --gf 0.5 \\
      --dt 0.01 --Ndt 100 \\
      --t_indices 0 10 20 30 40 50 60 70 80 90 100 \\
      --threshold 1e-3 \\
      --continuous True
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _time_cmap_norm(t_indices, Ntimes):
    """
    Return (cmap, norm) for a discrete time colormap that never raises ValueError.

    Boundaries are placed at the midpoints between adjacent normalised time
    values, with half-step extensions at both ends.  ncolors is set equal to
    the number of intervals (len(boundaries) - 1) and clip=True is used so
    matplotlib never counts under/over extensions as extra bins.
    """
    t_arr = np.array(sorted(t_indices), dtype=float) / Ntimes
    if len(t_arr) > 1:
        half = (t_arr[1] - t_arr[0]) / 2.0
        boundaries = np.concatenate([
            [t_arr[0] - half],
            (t_arr[:-1] + t_arr[1:]) / 2.0,
            [t_arr[-1] + half],
        ])
    else:
        boundaries = np.array([t_arr[0] - 0.05, t_arr[0] + 0.05])
    cmap = plt.get_cmap('viridis', len(boundaries) - 1)   # ncolors == n_intervals
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def load_kl_curve(output_dir, L, gi, gf, t_start, dt, t_idx, Ndt, continuous, angle=0.0):
    """
    Load a single KL curve and return (x_cumulative, kl_values).

    Returns (None, None) if any required file is missing.
    """
    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{angle}.npy")
    if not os.path.exists(global_filename):
        return None, None

    dense_groups_global = np.load(global_filename)
    rep_to_size = {}
    for g in dense_groups_global:
        valid_elements = g[g != -1]
        if len(valid_elements) > 0:
            rep = valid_elements[0]
            rep_to_size[rep] = len(valid_elements)

    inf_file  = os.path.join(output_dir,
        f"kl_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy")
    sort_file = os.path.join(output_dir,
        f"sort_KL_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}Nt{Ndt}C{continuous}.npy")

    if not os.path.exists(inf_file) or not os.path.exists(sort_file):
        return None, None

    infs  = np.load(inf_file).real
    order = np.load(sort_file)
    group_sizes = [1] + [rep_to_size[rep] for rep in order]
    x_cumulative = np.cumsum(group_sizes)[:-1]

    return x_cumulative, infs


def load_all(output_dir, L_list, gi, gf, t_start, dt, t_indices, Ndt, continuous, angle):
    """
    Returns nested dict:  data[L][t_idx] = (x_norm, kl)
    where x_norm = x_cumulative / N_total  (N_total = last cumulative index).
    Also returns dict N_total_dict[L].
    """
    data        = {}
    N_total_dict = {}
    for L in L_list:
        data[L] = {}
        N_total  = None
        for t_idx in t_indices:
            x, kl = load_kl_curve(output_dir, L, gi, gf, t_start, dt,
                                   t_idx, Ndt, continuous, angle)
            if x is None:
                print(f"  [skip] L={L}, t_idx={t_idx}: files not found")
                continue
            if N_total is None:
                N_total = int(x[-1])
            x_norm = x / N_total
            data[L][t_idx] = (x_norm, kl)

        if N_total is not None:
            N_total_dict[L] = N_total
            print(f"  L={L}: N_total={N_total}, loaded {len(data[L])} time slices")
        else:
            print(f"  L={L}: no data found in {output_dir}")

    return data, N_total_dict


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_normalized_overlay(data, L_list, t_indices, Ntimes, gi, gf, outdir):
    """Figure 1 — KL vs n_c/N_total, one panel per L."""
    n_L  = len(L_list)
    cmap_t, norm_t = _time_cmap_norm(t_indices, Ntimes)

    fig, axes = plt.subplots(1, n_L, figsize=(4.5 * n_L, 4.5), sharey=True)
    if n_L == 1:
        axes = [axes]

    for ax, L in zip(axes, L_list):
        for t_idx in t_indices:
            if t_idx not in data[L]:
                continue
            x_norm, kl = data[L][t_idx]
            ax.plot(x_norm, kl,
                    color=cmap_t(norm_t(t_idx / Ntimes)),
                    lw=0.9, alpha=0.85)
        ax.set_yscale('log')
        ax.set_ylim([1e-10, 1])
        ax.set_xlabel(r"$n_c \,/\, N_{\mathrm{total}}$")
        ax.set_title(f"L = {L}", fontsize=11)
        ax.grid(True, which='both', ls='--', alpha=0.4)

    axes[0].set_ylabel("KL")

    # shared colorbar
    sm = cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1],
                        ticks=np.array(t_indices) / Ntimes)
    cbar.set_label("time $t$")

    fig.suptitle(
        rf"KL vs normalized $n_c$ — gi={gi}, gf={gf}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, f"fig1_normalized_gi{gi}_gf{gf}.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    plt.close(fig)


def fig_vertical_separation(data, L_list, t_indices, Ntimes, gi, gf, outdir,
                              x_positions=(0.1, 0.3, 0.5, 0.7)):
    """
    Figure 2 — Ratio KL(t_max)/KL(t_min) at fixed n_c/N_total vs L.
    Also plots the same ratio for every consecutive pair of time curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: t_max / t_min ratio at several x positions ---
    ax = axes[0]
    colors_x = plt.get_cmap('tab10', len(x_positions))

    for i_x, x_pos in enumerate(x_positions):
        L_vals, sep_vals = [], []
        for L in L_list:
            t_min_key = min(data[L].keys()) if data[L] else None
            t_max_key = max(data[L].keys()) if data[L] else None
            if t_min_key is None or t_max_key is None:
                continue
            x0, kl0 = data[L][t_min_key]
            x1, kl1 = data[L][t_max_key]
            # Only interpolate inside the data range
            if x_pos < x0[0] or x_pos > x0[-1]:
                continue
            f0 = interp1d(x0, kl0, bounds_error=False, fill_value=np.nan)
            f1 = interp1d(x1, kl1, bounds_error=False, fill_value=np.nan)
            v0, v1 = float(f0(x_pos)), float(f1(x_pos))
            if np.isfinite(v0) and np.isfinite(v1) and v0 > 0 and v1 > 0:
                sep_vals.append(v1 / v0)
                L_vals.append(L)

        if L_vals:
            ax.plot(L_vals, sep_vals, 'o-',
                    color=colors_x(i_x),
                    label=rf"$n_c/N={x_pos}$")

    ax.set_xlabel("L")
    ax.set_ylabel(r"KL$(t_{\max})$ / KL$(t_{\min})$")
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.set_title("Vertical separation (t_max / t_min)")

    # --- Panel B: heatmap — separation between consecutive time pairs vs L ---
    # Build matrix: rows = L, cols = consecutive time pair
    t_sorted = sorted(t_indices)
    pairs    = [(t_sorted[i], t_sorted[i+1]) for i in range(len(t_sorted)-1)]
    x_ref    = 0.5   # fixed normalized position for the heatmap

    matrix = np.full((len(L_list), len(pairs)), np.nan)
    for i_L, L in enumerate(L_list):
        for i_p, (ta, tb) in enumerate(pairs):
            if ta not in data[L] or tb not in data[L]:
                continue
            xa, ka = data[L][ta]
            xb, kb = data[L][tb]
            if x_ref < xa[0] or x_ref > xa[-1]:
                continue
            fa = interp1d(xa, ka, bounds_error=False, fill_value=np.nan)
            fb = interp1d(xb, kb, bounds_error=False, fill_value=np.nan)
            va, vb = float(fa(x_ref)), float(fb(x_ref))
            if np.isfinite(va) and np.isfinite(vb) and va > 0 and vb > 0:
                matrix[i_L, i_p] = np.log10(vb / va)

    ax2 = axes[1]
    pair_labels = [rf"$t$={tb/Ntimes:.1f}" for (ta, tb) in pairs]
    im = ax2.imshow(matrix, aspect='auto', cmap='RdBu',
                    vmin=-np.nanmax(np.abs(matrix)), vmax=np.nanmax(np.abs(matrix)))
    ax2.set_xticks(range(len(pairs)))
    ax2.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(len(L_list)))
    ax2.set_yticklabels([str(L) for L in L_list])
    ax2.set_xlabel("Consecutive time step")
    ax2.set_ylabel("L")
    ax2.set_title(rf"$\log_{{10}}[\mathrm{{KL}}(t_{{b}})/\mathrm{{KL}}(t_{{a}})]$ at $n_c/N={x_ref}$")
    fig.colorbar(im, ax=ax2, label=r"$\log_{10}$ ratio")

    fig.suptitle(rf"Vertical separation — gi={gi}, gf={gf}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, f"fig2_vertical_separation_gi{gi}_gf{gf}.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    plt.close(fig)


def fig_horizontal_cost(data, N_total_dict, L_list, t_indices, Ntimes,
                         gi, gf, outdir, threshold=1e-3):
    """
    Figure 3 — Normalized n_c to reach KL < threshold vs L, for each time.

    Also fits a power-law n_c / N_total ~ L^alpha for each time to extract
    how the compression cost scales with system size.
    """
    cmap_t, norm_t = _time_cmap_norm(t_indices, Ntimes)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax  = axes[0]   # n_c/N_total vs L for each time
    ax2 = axes[1]   # exponents alpha vs time

    exponents, times_fitted = [], []

    for t_idx in t_indices:
        L_vals, nc_norm_vals = [], []
        for L in L_list:
            if t_idx not in data[L]:
                continue
            x_norm, kl = data[L][t_idx]
            # find first index where kl drops below threshold
            below = np.where(kl < threshold)[0]
            if len(below) == 0:
                continue
            nc_norm_vals.append(float(x_norm[below[0]]))
            L_vals.append(L)

        if len(L_vals) < 2:
            continue

        color = cmap_t(norm_t(t_idx / Ntimes))
        ax.plot(L_vals, nc_norm_vals, 'o-', color=color,
                label=rf"$t$={t_idx/Ntimes:.1f}")

        # Power-law fit: nc/N ~ c * L^alpha  → log(nc/N) = alpha*log(L) + log(c)
        log_L  = np.log(L_vals)
        log_nc = np.log(nc_norm_vals)
        if len(log_L) >= 2:
            coeffs = np.polyfit(log_L, log_nc, 1)
            exponents.append(coeffs[0])
            times_fitted.append(t_idx / Ntimes)

    ax.set_xlabel("L")
    ax.set_ylabel(rf"$n_c\,/\,N_{{\mathrm{{total}}}}$ at KL = {threshold:.0e}")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.set_title(rf"Compression cost at KL = {threshold:.0e}")

    # Exponent plot
    if exponents:
        ax2.plot(times_fitted, exponents, 'o-', color='steelblue')
        ax2.axhline(0, ls='--', color='gray', lw=0.8)
        ax2.set_xlabel("time $t$")
        ax2.set_ylabel(r"exponent $\alpha$   ($n_c/N \sim L^\alpha$)")
        ax2.set_title("Scaling exponent vs time")
        ax2.grid(True, ls='--', alpha=0.4)

    fig.suptitle(rf"Horizontal cost — gi={gi}, gf={gf}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, f"fig3_horizontal_cost_gi{gi}_gf{gf}.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    plt.close(fig)


def fig_collapse_check(data, N_total_dict, L_list, t_indices, Ntimes, gi, gf, outdir):
    """
    Figure 4 — Collapse check.
    Overlay KL(n_c/N_total) for all L at the same time.
    One panel per time t. If curves stack on top of each other the physics
    is size-independent in normalized units.
    """
    t_subset = t_indices[::max(1, len(t_indices)//6)]   # up to 6 panels
    n_panels = len(t_subset)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    cmap_L = plt.get_cmap('plasma', len(L_list))

    for ax, t_idx in zip(axes, t_subset):
        for i_L, L in enumerate(L_list):
            if t_idx not in data[L]:
                continue
            x_norm, kl = data[L][t_idx]
            ax.plot(x_norm, kl, color=cmap_L(i_L / max(1, len(L_list)-1)),
                    lw=1.0, alpha=0.85, label=f"L={L}")
        ax.set_yscale('log')
        ax.set_ylim([1e-10, 1])
        ax.set_xlabel(r"$n_c / N_{\mathrm{total}}$")
        ax.set_title(rf"$t={t_idx/Ntimes:.1f}$", fontsize=11)
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("KL")
    fig.suptitle(rf"Collapse check — gi={gi}, gf={gf}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, f"fig4_collapse_gi{gi}_gf{gf}.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KL curve separation vs system size L")
    parser.add_argument("--L_list",    type=int,   nargs='+', default=[10, 12, 14, 16, 18])
    parser.add_argument("--gi",        type=float, default=1.5)
    parser.add_argument("--gf",        type=float, default=0.5)
    parser.add_argument("--angle",     type=float, default=0.0)
    parser.add_argument("--t_start",   type=float, default=0.0)
    parser.add_argument("--dt",        type=float, default=0.01)
    parser.add_argument("--Ndt",       type=int,   default=100)
    parser.add_argument("--continuous",type=str,   default="True",
                        help="True or False (matches the filename suffix)")
    parser.add_argument("--t_indices", type=int,   nargs='+',
                        default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="KL threshold for horizontal-cost analysis")
    parser.add_argument("--x_positions", type=float, nargs='+',
                        default=[0.1, 0.3, 0.5, 0.7],
                        help="Normalized n_c positions for vertical-separation analysis")
    parser.add_argument("--outdir",    type=str,   default=".",
                        help="Directory to save figures")
    args = parser.parse_args()

    # Convert "True"/"False" string to Python bool
    continuous = args.continuous.strip().lower() not in ("false", "0", "no")

    pbc = True
    add = "_pbc" if pbc else ""
    if args.angle != 0.0:
        add += f"_{args.angle}"
    output_dir = "wavefunctions_data" + add

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Data directory : {output_dir}")
    print(f"L_list         : {args.L_list}")
    print(f"t_indices      : {args.t_indices}")
    print(f"continuous     : {continuous}")
    print()

    # ---- Load everything ----
    data, N_total_dict = load_all(
        output_dir, args.L_list, args.gi, args.gf,
        args.t_start, args.dt, args.t_indices,
        args.Ndt, continuous, args.angle)

    present_L = [L for L in args.L_list if data[L]]
    if not present_L:
        print("No data found for any L. Check the directory and filenames.")
        return

    # ---- Produce figures ----
    print("\nGenerating figures...")
    fig_normalized_overlay(data, present_L, args.t_indices,
                           args.Ndt, args.gi, args.gf, args.outdir)

    fig_vertical_separation(data, present_L, args.t_indices,
                            args.Ndt, args.gi, args.gf, args.outdir,
                            x_positions=args.x_positions)

    fig_horizontal_cost(data, N_total_dict, present_L, args.t_indices,
                        args.Ndt, args.gi, args.gf, args.outdir,
                        threshold=args.threshold)

    fig_collapse_check(data, N_total_dict, present_L, args.t_indices,
                       args.Ndt, args.gi, args.gf, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
