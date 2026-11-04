#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


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


def reconstruct_and_compute_infidelity(target_psi, sort_array, dense_groups_global, mode="erasure"):
    """
    Reconstructs the approximate wavefunction for each group in the sorting file
    and computes the exact infidelity against target_psi.
    """
    nstates = target_psi.shape[-1]
    
    # Normalize target wavefunction
    target_psi = target_psi / np.sqrt(np.sum(target_psi ** 2))
    
    # Compute full WHT of log(target_psi)
    wht_full = Fast_Hadamard(np.log(target_psi))
    
    # Map each group's representative index (g[0]) to its full list of valid indices
    rep_to_group = {}
    for g in dense_groups_global:
        valid_g = g[g != -1]
        if len(valid_g) > 0:
            rep_to_group[valid_g[0]] = valid_g

    if mode == "erasure":
        # Reverse back to chronological removal order: order_exact
        sequence = sort_array[::-1]
        current_wht = wht_full.copy()
    elif mode == "construction":
        # Keep saved order: starts from the last group removed (most important)
        sequence = sort_array
        current_wht = np.zeros_like(wht_full)
    else:
        raise ValueError("mode must be 'erasure' or 'construction'")

    ngroups = len(sequence)
    infs = np.zeros(ngroups, dtype=float)
    group_sizes = np.zeros(ngroups, dtype=int)

    for kk, rep in enumerate(sequence):
        group_indices = rep_to_group[rep]
        group_sizes[kk] = len(group_indices)

        if mode == "erasure":
            # Erase the group's WHT coefficients (skip g=0 global norm offset if present)
            valid_erase = group_indices[group_indices != 0]
            current_wht[valid_erase] = 0.0
        else:
            # Add the group's WHT coefficients into our approximate state
            current_wht[group_indices] = wht_full[group_indices]

        # Reconstruct log(psi_approx) via inverse WHT
        log_psi_approx = Fast_Hadamard(current_wht) * nstates
        
        # Shift by max to prevent exponential overflow during np.exp()
        log_psi_approx -= np.max(log_psi_approx)
        psi_approx = np.exp(log_psi_approx)
        
        # Normalize the reconstructed approximate wavefunction
        psi_approx /= np.sqrt(np.sum(psi_approx ** 2))

        # Compute exact infidelity: 1 - |<target_psi | psi_approx>|^2
        overlap = np.sum(target_psi * psi_approx)
        infs[kk] = 1.0 - overlap ** 2

    num_coeffs = np.cumsum(group_sizes)
    return num_coeffs, infs


def align_saved_infs(saved_infs_raw, mode="erasure"):
    """
    Aligns the saved inf_...npy array with the chosen plotting mode.
    Note: saved_infs_raw is stored on disk as inf_exact[::-1].
    """
    # Chronological erasure order: inf_exact[0] is after 1st group removed
    infs_chronological = saved_infs_raw[::-1]
    
    if mode == "erasure":
        return infs_chronological
    elif mode == "construction":
        # In construction mode:
        # Step k (0-indexed) has (k+1) most important groups kept, 
        # which is identical to having (ngroups - 1 - k) least important groups erased!
        # For the final step (all groups kept), infidelity is 0.
        infs_constructed = np.zeros_like(infs_chronological)
        # Reversing infs_chronological[:-1] gives the state with 1, 2, ..., ngroups-1 groups kept
        infs_constructed[:-1] = infs_chronological[:-1][::-1]
        infs_constructed[-1] = 0.0
        return infs_constructed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare reconstructed WF infidelity with saved inf file.")
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--gi", type=float, default=1.5)
    parser.add_argument("--gf", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_idx", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Ndt", type=int, default=100)
    parser.add_argument("--mode", type=str, default="erasure", choices=["erasure", "construction"],
                        help="'erasure' removes groups top-down; 'construction' adds groups bottom-up.")

    args = parser.parse_args()

    pbc = True
    add = "_pbc" if pbc else ""
    if args.angle != 0.0:
        add += f"_{args.angle}"

    output_dir = "wavefunctions_data" + add

    # 1. Load target wavefunction
    wf_base = f"wavefunctions_L{args.L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}Nt{args.Ndt}.npy"
    wf_file = os.path.join(output_dir, wf_base)
    psi_history = np.lib.format.open_memmap(wf_file, mode='r', dtype=np.complex128, shape=(args.Ndt + 1, 2**args.L))
    exact_psi = np.abs(psi_history[args.t_idx])

    # 2. Load global groups
    global_filename = os.path.join(output_dir, f"groups_global_L{args.L}_a{args.angle}.npy")
    dense_groups_global = np.load(global_filename)

    # Create a 2-panel plot: Top = Overlay Comparison, Bottom = Absolute Difference
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    colors = {False: "tab:blue", True: "tab:orange"}

    for continuous in [False, True]:
        sort_base = f"sort_L{args.L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{args.t_idx}Nt{args.Ndt}C{continuous}.npy"
        inf_base = f"inf_L{args.L}gi{args.gi:.2f}gf{args.gf:.2f}ti{args.t_start}dt{args.dt}t{args.t_idx}Nt{args.Ndt}C{continuous}.npy"
        
        sort_file = os.path.join(output_dir, sort_base)
        inf_file = os.path.join(output_dir, inf_base)
        
        if not os.path.exists(sort_file) or not os.path.exists(inf_file):
            print(f"Skipping Continuous={continuous}: Missing sort or inf file.")
            continue

        # Load the memmapped sorting and infidelity arrays
        sort_array = np.lib.format.open_memmap(sort_file, mode='r', dtype=int)
        saved_infs_raw = np.real(np.lib.format.open_memmap(inf_file, mode='r', dtype=np.complex128))

        # Reconstruct wavefunctions and compute infidelities
        num_coeffs, infs_reconstructed = reconstruct_and_compute_infidelity(
            exact_psi, sort_array, dense_groups_global, mode=args.mode
        )
        
        # Align saved infidelities to match the mode
        infs_saved = align_saved_infs(saved_infs_raw, mode=args.mode)

        # Top Panel: Plot both curves
        ax1.plot(num_coeffs, infs_reconstructed, marker='o', markersize=4, linestyle='-', 
                 color=colors[continuous], alpha=0.8, label=f"Reconstructed WF (C={continuous})")
        ax1.plot(num_coeffs, infs_saved, marker='x', markersize=4, linestyle='--', 
                 color="black" if not continuous else "red", alpha=0.9, label=f"Saved inf file (C={continuous})")
        print(infs_saved)
        # Bottom Panel: Plot the absolute residual error between the two methods
        abs_diff = np.abs(infs_reconstructed - infs_saved)
        ax2.plot(num_coeffs, abs_diff, marker='.', markersize=4, linestyle='-',
                 color=colors[continuous], label=f"|Reconstructed - Saved| (C={continuous})")

    xlabel = "Number of Coefficients Removed" if args.mode == "erasure" else "Number of Coefficients Kept"
    
    ax1.set_ylabel("Infidelity")
    ax1.set_title(f"Comparison: Reconstructed WF vs Saved Inf File (L={args.L}, t={args.t_idx * args.dt:.2f}, mode={args.mode})")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Abs. Difference")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()

    plot_file = os.path.join(output_dir, f"compare_inf_{args.mode}_L{args.L}_t{args.t_idx}.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Saved comparison plot to {plot_file}")
    plt.show()
