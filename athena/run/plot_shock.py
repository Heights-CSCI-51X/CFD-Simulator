import matplotlib.pyplot as plt
import numpy as np
import glob

# 1. Find all .tab files and sort them to get the final time step
tab_files = sorted(glob.glob("*.tab"))
if not tab_files:
    print("No .tab files found in this directory!")
    exit()

final_file = tab_files[-1]
print(f"Plotting final state from: {final_file}")

# 2. Read the data columns, skipping the header lines starting with '#'
# Typically: col 1 = x coordinate, col 2 = density (rho), col 4 = pressure
# Verify your column indices using the 'head' command if needed!
data = np.loadtxt(final_file, comments='#')

x = data[:, 1]       # x-coordinate (cell center)
rho = data[:, 2]     # Density
press = data[:, 4]   # Pressure

# 3. Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(x, rho, 'b-', label='Density (rho)', linewidth=2)
ax1.set_ylabel('Density')
ax1.grid(True)
ax1.legend()
ax1.set_title(f"Athena++ Shock Tube Profile ({final_file})")

ax2.plot(x, press, 'r-', label='Pressure', linewidth=2)
ax2.set_xlabel('Position (X)')
ax2.set_ylabel('Pressure')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Save the plot to an image file since you are working via terminal
plt.savefig("shock_profile.png", dpi=150)
print("Saved plot to shock_profile.png!")
