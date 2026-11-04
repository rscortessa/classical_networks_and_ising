#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def main():
    parser = argparse.ArgumentParser(
        description="Plot Linear and Non-Linear (Vacuum) Parts of KL vs Number of Removed Coefficients"
    )
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument(
        "--log_abs",
        action="store_true",
        help="Plot absolute values |KL_linear| and |KL_vacuum| on a log scale",
    )

    args = parser.parse_args()

    pbc = True
    add = "_pbc" if pbc else ""
    if args.angle != 0.0:
        add += f"_{args.angle}"

    output_dir = "wavefunctions_data" + add
    t_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    L = args.L
    Ntimes = 100
    continuous_list = [True, False]

    n_values = np.arange(0, Ntimes + 1, 10)
    cmap = plt.get_cmap("viridis", len(n_values))
    norm = mcolors.BoundaryNorm(np.arange(-5, Ntimes + 6, 10) / Ntimes, cmap.N)

    # 1. Load the groups once to figure out the size of each group
    global_filename = os.path.join(output_dir, f"groups_global_L{L}_a{args.angle}.npy")
    if not os.path.exists(global_filename):
        raise FileNotFoundError(f"Cannot find {global_filename}. Make sure groups are generated.")

    dense_groups_global = np.load(global_filename)

    # Map each group's representative index (the first element) to the size of that group
    rep_to_size = {}
    for g in dense_groups_global:
        valid_elements = g[g != -1]
        if len(valid_elements) > 0:
            rep = valid_elements[0]
            rep_to_size[rep] = len(valid_elements)

    for continuous in continuous_list:
        fig, (ax_lin, ax_vac) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

        for t_idx in t_list:
            suffix = (
                f"L{L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}"
                f"dt{args.dt}t{t_idx}Nt{args.Ndt}C{continuous}.npy"
            )
            lin_file = os.path.join(output_dir, f"kl_linear_{suffix}")
            vac_file = os.path.join(output_dir, f"kl_vacuum_{suffix}")
            sort_file = os.path.join(output_dir, f"sort_KL_{suffix}")

            if not (os.path.exists(lin_file) and os.path.exists(vac_file) and os.path.exists(sort_file)):
                print(f"Warning: Missing data files for t_idx={t_idx}, Continuous={continuous}. Skipping...")
                continue

            # 2. Load the arrays (take real part as they are stored in complex128 format)
            kl_lin = np.real(np.load(lin_file))
            kl_vac = np.real(np.load(vac_file))
            order = np.load(sort_file)

            # 3. Reconstruct the x-axis (cumulative sum of removed coefficients)
            group_sizes = [1] + [rep_to_size[rep] for rep in order]
            x_cumulative_removed = np.cumsum(group_sizes)[:-1]

            color = cmap(norm(t_idx / Ntimes))
            label = f"{t_idx / Ntimes:.2f}"

            y_lin = np.abs(kl_lin) if args.log_abs else kl_lin
            y_vac = np.abs(kl_vac) if args.log_abs else kl_vac

            # 4. Plot in separate subplots
            ax_lin.plot(
                x_cumulative_removed, y_lin, color=color, marker=".", markersize=4, label=label
            )
            ax_vac.plot(
                x_cumulative_removed, y_vac, color=color, marker=".", markersize=4, label=label
            )

        # Format Subplot 1: Linear part of KL
        ax_lin.set_xlabel("$n_c$")
        ax_lin.set_ylabel("$|\\mathrm{KL}_{\\mathrm{linear}}|$" if args.log_abs else "$\\mathrm{KL}_{\\mathrm{linear}}$")
        ax_lin.set_title(f"Linear Part of KL vs $n_c$ (L={L}, gi={args.gi}, gf={args.gf})")
        if args.log_abs:
            ax_lin.set_yscale("log")
            ax_lin.set_ylim([1e-10, 1e2])
        ax_lin.grid(True, which="both", ls="--", alpha=0.5)
        ax_lin.legend(title="time $t$", fontsize=8)

        # Format Subplot 2: Non-Linear (Vacuum) part of KL
        ax_vac.set_xlabel("$n_c$")
        ax_vac.set_ylabel("$|\\mathrm{KL}_{\\mathrm{vacuum}}|$" if args.log_abs else "$\\mathrm{KL}_{\\mathrm{vacuum}}$")
        ax_vac.set_title(f"Non-Linear (Vacuum) Part of KL vs $n_c$ (L={L}, gi={args.gi}, gf={args.gf})")
        if args.log_abs:
            ax_vac.set_yscale("log")
            ax_vac.set_ylim([1e-10, 1e2])
        ax_vac.grid(True, which="both", ls="--", alpha=0.5)
        ax_vac.legend(title="time $t$", fontsize=8)

        # Shared colorbar for time t
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_lin, ax_vac], ticks=n_values / Ntimes, pad=0.02)
        cbar.set_label("time $t$")

        scale_tag = "_logabs" if args.log_abs else ""
        plot_path = os.path.join(
            output_dir,
            f"kl_linear_vacuum_plot_L{args.L}_gi{args.gi}_gf_{args.gf}continuous{continuous}{scale_tag}.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved successfully to {plot_path}")


if __name__ == "__main__":
    main()
