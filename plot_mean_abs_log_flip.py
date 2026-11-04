import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters matching the generation script
Ls = [8, 10, 12, 14, 16, 18, 20, 22]
gi = 1.5
gf = 0.5
dt = 0.01
Ndt = 100
t_start = 0.0

pbc = True
add = "_pbc" if pbc else ""
output_dir = "wavefunctions_data" + add

# Create time array for the x-axis
time_steps = np.linspace(t_start, t_start + Ndt * dt, Ndt + 1)

plt.figure(figsize=(10, 6))

for L in Ls:
    log_flip_filename = os.path.join(output_dir, f"logflip_L{L}_gi{gi:.2f}_gf{gf:.2f}.npy")
    
    if not os.path.exists(log_flip_filename):
        print(f"File not found: {log_flip_filename}, skipping L={L}")
        continue
        
    # Open the memory-mapped file in read-only mode
    log_flip_ratio_file = np.lib.format.open_memmap(
        log_flip_filename,
        mode='r',
        dtype=np.float64,
        shape=(Ndt + 1, 2**L)
    )
    
    # Calculate the mean of the absolute value for each time step.
    # Iterating over idx_t prevents loading the entire 2**L x Ndt array into RAM at once.
    mean_abs_log = np.zeros(Ndt + 1)
    for idx_t in range(Ndt + 1):
        mean_abs_log[idx_t] = np.mean(np.abs(log_flip_ratio_file[idx_t]))
        
    plt.plot(time_steps, mean_abs_log, marker='o', markersize=3, label=f'L={L}')

# Plot aesthetics
plt.ylim([0,4])
plt.title(f"Mean Absolute log(flip_ratio) over Time [gi={gi}, gf={gf}]")
plt.xlabel("Time (t)")
plt.ylabel(r"$\langle |\log(\text{flip\_ratio})| \rangle$")
plt.legend(title="System Size (L)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save and display
plot_filename = os.path.join(output_dir, f"mean_abs_logflip_gi{gi:.2f}_gf{gf:.2f}.png")
plt.savefig(plot_filename, dpi=300)
plt.show()

print(f"Plot successfully saved to {plot_filename}")
