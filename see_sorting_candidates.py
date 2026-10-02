import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters as defined in see_sorting.py[cite: 2]
L = 10
gi = 1.5
gf = 0.5
angle = 0.0
t_start = 0.0
dt = 0.01
Ndt = 100
t_idx = 60

# Calculate the physical time step t[cite: 2]
t = t_start + t_idx * dt

# List of candidate values to compare
candidate_list = [1,10, 50, 100, 150, 200]

output_dir = "wavefunctions_data" #[cite: 2]

plt.figure(figsize=(8, 5))

# Loop through each candidate value, load the respective file, and plot
for candidates in candidate_list:
    base_name_inf = f"inf_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}t{t_idx}candidates{candidates}Nt{Ndt}.npy" #[cite: 2]
    filename_inf = os.path.join(output_dir, base_name_inf) #[cite: 2]

    if not os.path.exists(filename_inf):
        print(f"Warning: Cannot find {filename_inf}. Skipping.")
        continue

    # Load the data[cite: 2]
    inf_data = np.load(filename_inf)
    
    # The x-axis represents the number of parameters added[cite: 2]
    num_parameters = np.arange(1, len(inf_data) + 1)

    # Plotting the infidelity curve for the current candidate[cite: 2]
    plt.plot(num_parameters, inf_data.real, marker='o', markersize=2, linestyle='-', label=f'candidates={candidates}')

# Formatting the plot[cite: 2]
plt.yscale('log')
plt.xlabel('Number of Parameters')
plt.ylabel('Infidelity')
plt.title(f'Infidelity vs Parameters | L={L}, $g_i$={gi}, $g_f$={gf}, angle={angle}, t={t:.2f}')
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the figure with an updated filename for the comparison
plot_filename = os.path.join(output_dir, f"plot_inf_compare_candidates_L{L}gi{gi:.2f}gf{gf:.2f}t{t_idx}.png")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to: {plot_filename}")

# Display the plot[cite: 2]
plt.show()
