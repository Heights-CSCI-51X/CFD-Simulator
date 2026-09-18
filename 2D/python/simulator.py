#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# 1. Define your grid dimensions (rows, columns)
rows, cols = 5, 5

# 2. Initialize a blank canvas grid (filled with white squares)
# The shape is (rows, columns, 3) to represent Red, Green, and Blue channels
grid = np.ones((rows, cols, 3))
u = np.ones((rows, cols, 0))
v = np.ones((rows, cols, 0))

# 3. Define your custom colors (RGB scaled between 0.0 and 1.0)
RED = [1.0, 0.0, 0.0]
MINT = [0.4, 0.9, 0.7]
GOLD = [1.0, 0.8, 0.0]
DEEP_BLUE = [0.1, 0.2, 0.6]

plt.ion()


for i in range(rows):
    for j in range(cols):
        grid[i, j] = MINT


plt.draw()
plt.pause(4)

for i in range(rows):
    for j in range(cols):
        grid[i, j] = GOLD

plt.ioff()

# 5. Plot the grid
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(grid, interpolation='nearest')
ax.axis('off')

# Display the final graphic
plt.show()