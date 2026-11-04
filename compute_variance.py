import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters matching your simulation script
Ls = [12, 14, 16, 18, 20]
gi = 1.5
gf = 0.5
t_start = 0.0
dt = 0.01
Ndt = 100
pbc = True
add = "_pbc" if pbc else ""
output_dir = "wavefunctions_data" + add

times = np.linspace(t_start, t_start + Ndt * dt, Ndt + 1)

plt.figure(figsize=(10, 6))

variances_all = {}

for L in Ls:
    log_flip_filename = os.path.join(output_dir, f"logflip_L{L}_gi{gi:.2f}_gf{gf:.2f}.npy")
    
    if not os.path.exists(log_flip_filename):
        print(f"File not found: {log_flip_filename}, skipping L={L}")
        continue
        
    # Open the memory-mapped log-flip ratio file
    # Shape: (Ndt + 1, 2**L)
    log_flip_ratio_file = np.lib.format.open_memmap(
        log_flip_filename,
        mode='r',
        dtype=np.float64
    )
    
    variances = []
    for idx_t in range(log_flip_ratio_file.shape[0]):
        log_flip_ratio = log_flip_ratio_file[idx_t]
        mask = np.isfinite(log_flip_ratio)
        data = log_flip_ratio[mask]
        
        if len(data) > 0:
            # Compute sample variance (ddof=1)
            var_val = np.var(data, ddof=1)
        else:
            var_val = np.nan
            
        variances.append(var_val)
        
    variances = np.sqrt(np.array(variances))
    variances_all[L] = np.sqrt(variances)
    
    # Plot variance vs time for this system size on a single plot
    plt.plot(times[:len(variances)], variances, label=f"L = {L}", linewidth=1.5)

plt.xlabel(r"Time $t$", fontsize=12)
plt.ylabel(r"$\sigma(\log_{10}(|\psi(s') / \psi(s)|)$", fontsize=12)
plt.title(rf"$\sigma$ of the ratios ($g_i={gi}, g_f={gf}$)", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save the plot and computed variance data
plot_filename = os.path.join(output_dir, f"std_vs_time_gi{gi:.2f}_gf{gf:.2f}.png")
plt.savefig(plot_filename, dpi=300)
print(f"Plot saved to {plot_filename}")

data_filename = os.path.join(output_dir, f"std_data_gi{gi:.2f}_gf{gf:.2f}.npz")
np.savez(data_filename, Ls=Ls, times=times, **{str(L): variances_all[L] for L in variances_all})
print(f"Variance data saved to {data_filename}")

plt.show()
