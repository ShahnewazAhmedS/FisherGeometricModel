# MIT License
#
# Copyright (c) 2025 Shahnewaz Ahmed
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This script generates a 3D surface plot to visualize the fitness landscape
as described by Fisher's Geometric Model. The plot represents fitness as a
function of two biological state variables, with arrows indicating the
evolutionary path towards the fitness optimum.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='3d')

num1 = np.arange(-1.5, 1.5, 0.025)
num2 = np.arange(-1.5, 1.5, 0.025)
X, Y = np.meshgrid(num1, num2)
Z =np.exp(-(X**2 + Y**2))

# Create the surface plot.
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7) # Added alpha for better arrow visibility

# Add a color bar.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# Set labels and title (optional).
ax.set_xlabel('Biological state $z_1$')
ax.set_ylabel('Biological state $z_2$')
ax.set_zlabel('$W(z_1,z_2)$')
ax.set_title('Fitness function', fontweight='bold')
# ax.set_title('Potential function', fontweight='bold')

# Show the plot.
# plt.show()

# Add arrows climbing from (1, 0, Z(1,0)) towards (0, 0, Z(0,0))
start_point = np.array([1.5, 0])
end_point = np.array([0, 0])
start_z = np.exp(-(start_point[0]**2 + start_point[1]**2))
end_z = np.exp(-(end_point[0]**2 + end_point[1]**2))

num_arrows = 10  # Adjust for more or fewer arrows
arrow_starts = np.linspace(start_point, end_point, num_arrows)
arrow_zs = np.exp(-(arrow_starts[:, 0]**2 + arrow_starts[:, 1]**2))

# Calculate the direction vectors for the arrows
arrow_ends = np.roll(arrow_starts, -1, axis=0)
arrow_ends_z = np.roll(arrow_zs, -1)

# Remove the last point to avoid an invalid direction
arrow_starts = arrow_starts[:-1]
arrow_zs = arrow_zs[:-1]
arrow_ends = arrow_ends[:-1]
arrow_ends_z = arrow_ends_z[:-1]

u = arrow_ends[:, 0] - arrow_starts[:, 0]
v = arrow_ends[:, 1] - arrow_starts[:, 1]
w = arrow_ends_z - arrow_zs

ax.quiver(arrow_starts[:, 0], arrow_starts[:, 1], arrow_zs, u, v, w,
          length=0.7, alpha = 1, color='black', arrow_length_ratio=0.4) # Adjust length and other parameters
ax.view_init(elev=60, azim=80)
ax.grid(True)
ax.xaxis.set_tick_params(color='white')
ax.xaxis.set_ticklabels(['0'], color='white')
ax.yaxis.set_tick_params(color='white')
ax.yaxis.set_ticklabels(['0'], color='white')
ax.zaxis.set_tick_params(color='white')
ax.zaxis.set_ticklabels(['0'], color='white')

# Show the plot.
# plt.savefig("fitness_function.pdf", dpi=300, bbox_inches='tight')
plt.savefig("fitness_potential.png", dpi=300, bbox_inches='tight')
plt.show()


