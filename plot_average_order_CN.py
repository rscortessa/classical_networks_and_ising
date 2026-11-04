import os
import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters matching compute_average_order.py
Ls = [8, 10, 12, 14, 16, 18, 20, 22]
pbc = True
add = "_pbc" if pbc else ""
output_dir = f"wavefunctions_data{add}"

t_start = 0.0
dt = 0.01
Ndt = 100
time = t_start + np.arange(Ndt + 1) * dt

# Load the saved array (shape: (len(Ls), Ndt + 1))
data_path = os.path.join(output_dir, "average_order_CN_array.npy")
average_order = np.load(data_path)

# Plot average order vs. time for each system size L
plt.figure(figsize=(8, 5))

for idx_L, L in enumerate(Ls):
    plt.plot(time, average_order[idx_L]/L, label=f"$L = {L}$")

plt.xlabel("Time $t$")
plt.ylabel(r"Average Order $\langle |A| \rangle_{CN}/L$")
plt.title(r"$\langle |A| \rangle_{CN}/L$ ($g_i=1.50 \to g_f=0.50$)")
plt.legend(title="System Size", bbox_to_anchor=(1.04, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save and display figure
plt.savefig("average_order_CN_plot.png", dpi=300)
plt.show()
