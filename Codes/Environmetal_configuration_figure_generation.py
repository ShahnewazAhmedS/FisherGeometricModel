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
This script generates a 3D scatter plot to visualize different environment
configurations. It uses a multivariate normal distribution to create a cloud of
points representing mutations and then adds specific points to denote stress and
non-stress environments, along with their optimal phenotypes.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


# Define the mean and covariance matrix
mean = [0, 0, 0]
covariance = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])*0.2

# mean1 = [0, 5, 0]
# covariance1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])*0.002

# Generate random samples from a multivariate Gaussian distribution
num_samples = 1000
data = np.random.multivariate_normal(mean, covariance, num_samples)
# data1 = np.random.multivariate_normal(mean1, covariance1, num_samples)

# Create the 3D scatter plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams.update({'font.size': 16})

# Plot the data points
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o',s=1, label='Mutation', alpha = 0.2)
# ax.scatter(np.linspace(0.1,2,5), np.random.uniform(0,0.1,5), np.random.uniform(0,0.1,5),c='g', marker='x',s=300,  label='Non-stress environment')
cxc = np.random.uniform(0,0.1,2)
cxc2 = np.random.uniform(0,0.1,2)
ax.scatter(np.linspace(1,2,2), cxc, cxc2,c='g', marker='o',s=100,  label='Non-stress environment')
ax.scatter(1, cxc[0]+0.5, cxc2[0]-0.5,marker='$z\'_{opt,1}$', s=1600, c='g' )
ax.scatter(2, cxc[1]+0.5, cxc2[1]-0.5,marker='$z\'_{opt,2}$', s=1600, c='g')
ax.scatter([0],[5],[0],c='r',marker='o',s=50,  label='Stress environment')
ax.scatter([0],[6.5],[0.5],c='r',marker='$z_{opt,1}$',s=1600)
# ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2],c='r',marker='.',s=1,  label='Stress environment')
ax.scatter([0],[0],[4],c='r',marker='o',s=50)
ax.scatter([0],[1.5],[4.5],c='r',marker='$z_{opt,2}$',s=1600)
line = np.linspace(-6, 6, 100)
ax.plot(0*line, line, 0*line, 'k', linewidth=2, linestyle='-')
ax.plot(line, 0*line, 0*line, 'k', linewidth=2, linestyle='-')
ax.plot(0*line, 0*line, line, 'k', linewidth=2, linestyle='-')

# Set axis labels
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_zlabel('$z_3$')
ax.xaxis.set_tick_params(color='white')
ax.xaxis.set_ticklabels(['0'], color='white')
ax.yaxis.set_tick_params(color='white')
ax.yaxis.set_ticklabels(['0'], color='white')
ax.zaxis.set_tick_params(color='white')
ax.zaxis.set_ticklabels(['0'], color='white')
# Set plot title
ax.set_title('Environment configurations', fontweight = 'bold')
# plt.axis([-4, 4, -4, 4, -4, 4])
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)
ax.view_init(elev=20, azim=70)
# Set the aspect ratio
# Show the plot
plt.legend()
plt.savefig("environment_configuration.pdf", dpi=300, bbox_inches='tight')
plt.show()