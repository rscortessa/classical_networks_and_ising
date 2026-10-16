#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import netket as nk

def group_equivalent_states(L, hi, states, pad_to_dense=True):
    """
    Groups Hilbert space states of a 1D spin-1/2 chain based on 
    translation (cyclic) and reflection symmetries.
    """
    batch_global_indices = hi.states_to_numbers(states)
    
    lookup = np.full(hi.n_states, -1, dtype=int)
    lookup[batch_global_indices] = np.arange(len(states))
    
    visited = np.zeros(len(states), dtype=bool)
    
    groups_local = []
    groups_global = []
    
    for idx, state in enumerate(states):
        if visited[idx]:
            continue
            
        symm_states = np.empty((2 * L, L))
        for shift in range(L):
            rolled = np.roll(state, shift)
            symm_states[shift] = rolled
            symm_states[L + shift] = rolled[::-1] 
            
        symm_global_all = np.unique(hi.states_to_numbers(symm_states))
        symm_local_all = lookup[symm_global_all]
        
        valid_mask = symm_local_all != -1
        symm_local = symm_local_all[valid_mask]
        symm_global = symm_global_all[valid_mask]
        
        groups_local.append(symm_local.tolist())
        groups_global.append(symm_global.tolist())
        
        visited[symm_local] = True
        
    if pad_to_dense:
        max_len = max(len(g) for g in groups_local)
        
        dense_local = np.full((len(groups_local), max_len), -1, dtype=int)
        dense_global = np.full((len(groups_global), max_len), -1, dtype=int)
        
        for i in range(len(groups_local)):
            dense_local[i, :len(groups_local[i])] = groups_local[i]
            dense_global[i, :len(groups_global[i])] = groups_global[i]
            
        return dense_local, dense_global
        
    return groups_local, groups_global


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--angle", type=float, default=0.0)
    args, _ = parser.parse_known_args()
    pbc=True
    L = args.L
    angle = args.angle
    hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)
    add=""
    
    if pbc:
        add+="_pbc"
    if angle != 0.0:
        add+="_"+str(angle)
    
    output_dir = "wavefunctions_data"+add
    os.makedirs(output_dir, exist_ok=True)

    # Mirror the initial_mask logic from your main script
    if angle != 0.0:
        initial_mask = np.ones(2**L, dtype=bool)
    else:
        initial_mask = (np.arange(2**L, dtype=int) % 2 == 0)

    states = hi.all_states()[initial_mask]
    
    print(f"Generating groups for L={L}, number of valid states={len(states)}...")
    
    # Must use pad_to_dense=True to save as a standard numpy array
    dense_local, dense_global = group_equivalent_states(L, hi, states, pad_to_dense=True)
    
    local_filename = f"groups_local_L{L}_a{angle}.npy"
    global_filename = f"groups_global_L{L}_a{angle}.npy"

    filename_local = os.path.join(output_dir,local_filename)
    filename_global = os.path.join(output_dir,global_filename)
    
    np.save(filename_local, dense_local)
    np.save(filename_global, dense_global)
    
    print(f"Saved {filename_local} with shape {dense_local.shape}")
    print(f"Saved {filename_global} with shape {dense_global.shape}")
