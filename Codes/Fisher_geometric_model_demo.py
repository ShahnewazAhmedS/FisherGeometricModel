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
This script generates an interactive plot to visualize Fisher's Geometric Model.
It displays a 2D fitness landscape based on a normal distribution and shows the
distribution of fitness effects (DFE) for a given set of mutations.

Note: The interactive sliders for adjusting the optimal phenotype will only
work when this code is run in a Jupyter Notebook or a similar interactive
environment that supports ipywidgets.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Define the 2D normal distribution function F
def normal_distribution_2d(x, y, mu_x, mu_y, sigma_x=1, sigma_y=1):
    return np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2))) / (2 * np.pi * sigma_x * sigma_y)

# Create a grid of x and y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Evaluate the normal distribution function F on the grid

num_samples = 100000
np.random.seed(42)
samples_x = np.random.normal(0, 0.2, num_samples)-1.5
# divisions = 8
# angles_used_for_things = np.linspace(0,np.pi,divisions)
# probabilities = np.ones(divisions)/divisions
# angles_random_distribution = np.random.choice(angles_used_for_things, num_samples, p=probabilities)
# samples_xx = samples_x*np.cos(angles_random_distribution)
# samples_y = samples_x*np.sin(angles_random_distribution)
# samples_x = samples_xx
samples_y = np.random.normal(0, 0.2, num_samples)+2.5

# samples_X = np.linspace(-0.3, 0.3, 100)
# samples_Y = np.linspace(-0.3, 0.3, 100)
# samples_x, samples_y = np.meshgrid(samples_X, samples_Y)
# samples_x = samples_x.flatten()
# samples_y = samples_y.flatten()

fixed_environments = np.linspace(-5,5,50)
fitnesses_for_fixed_environments = np.zeros((len(fixed_environments),num_samples))
for i in range(0,len(fixed_environments)):
  sample_values = normal_distribution_2d(samples_x, samples_y,fixed_environments[i], 0)
  max_value = normal_distribution_2d(0,0,fixed_environments[i], 0)
  fitness = np.log(sample_values/max_value)
  fitnesses_for_fixed_environments[i]=fitness

# Function to generate the plot and histogram
def plot_and_histogram(mu_x, mu_y):
    # Take 100 samples close to the random point with normal distribution

    # samples_x = np.random.normal(mu_x, 0.2, num_samples)
    # samples_y = np.random.normal(mu_y, 0.2, num_samples)
    # Evaluate F at these sample points
    sample_values = normal_distribution_2d(samples_x, samples_y,mu_x,mu_y)
    max_value = normal_distribution_2d(0,0,mu_x, mu_y)
    fitness = np.log(sample_values/max_value)
    # sample_values = normal_distribution_2d(samples_x, samples_y)

    # Create a figure with two subplots
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the 2D normal distribution and the sampled points
    # plt.subplot(1,2, 1)
    F = normal_distribution_2d(X, Y, mu_x, mu_y )
    plt.rcParams.update({'font.size': 20})
    plt.contourf(X, Y, F, levels=100, cmap='viridis')
    plt.colorbar(label= 'Fitness Function $W(z_1,z_2)$')
    plt.scatter(samples_x, samples_y, c='red', s=10, label='Mutations')
    plt.scatter(mu_x, mu_y, c='red', marker = 'x' ,s=100, label='Optimum')
    # plt.scatter(0, 0, c='black', s=150)
    # plt.scatter(fixed_environments, fixed_environments*0, c='pink', s=10, label='fixed environment')
    # ax.annotate('', xy=(-1.5, 2.5), xytext=(0, 0), arrowprops=dict(facecolor='white', shrink=0))
    ax.annotate('', xy=(-1.25, 3.35), xytext=(-1.5, 2.5), arrowprops=dict(facecolor='white', shrink=0))
    # ax.annotate('', xy=(1.0, 1.6), xytext=(0, 0), arrowprops=dict(facecolor='white', shrink=0))
    # ax.annotate('$z_0$', xy=(-0.75, 1.25), xytext=(-0.75, 1.25), color='white',arrowprops=dict(facecolor='white', shrink=0.0))
    plt.text(-1.25, 2.25, '$z_0$', fontsize=30, fontweight = 'bold', color='white', rotation=0)
    plt.text(mu_x+0.1, mu_y-0.3, '$z_{opt}$', fontsize=30,  fontweight = 'bold',color='red', rotation=0)
    plt.text(-2, 3.25, '$dz$', fontsize=20,  fontweight = 'bold', color='white', rotation=0)
    plt.scatter(-1.5, 2.5, c='blue', s=50, label='Ancestor')
    plt.title('Fisher\'s Geometric Model', fontweight = 'bold')
    plt.xlabel('Phenotype $z_1$')
    plt.ylabel('Phenotype $z_2$')
    plt.legend()

    # Plot the histogram of F values at the sample points
    # plt.subplot( 1,2, 2)
    left, bottom, width, height = [0.5, 0.25, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.xaxis.set_tick_params(color='white')


    ax2.yaxis.set_tick_params(color='white')
    fitness_delete = fitness[fitness<-0.01]
    fitness_beneficial = fitness[fitness>=0]
    # n, bins, patches = plt.hist(data, bins=10)
    n, bins, patches= plt.hist(fitness,  bins=50, edgecolor='none')
    # plt.hist([fitness_delete, fitness_beneficial], color= ['r','g'], label=['Deleterious', 'Beneficial'], bins=20, edgecolor='black')
    ax2.xaxis.set_ticklabels([-2, -1, 0, 1] ,color='white')
    ax2.yaxis.set_ticklabels([0, 100, 500] ,color='white')
    thhh = 27
    patches[0].set_label('Deleterious')
    patches[thhh].set_label('Neutral')

    patches[-1].set_label('Beneficial')

    for i in range(len(patches)):
      if i < thhh:
        patches[i].set_facecolor('red')
      else:
        patches[i].set_facecolor('green')

    patches[thhh].set_facecolor('Grey')
    # plt.legend()
    # plt.title('Distribution of Fitness Effect (DFE)', fontweight = 'bold')
    plt.xlabel('Relative fitness', c='w')
    plt.ylabel('Frequency',c='w')




    # Plot the autocorrelation from target point to points on x axis
    # plt.subplot(2,2,4)
    # autocorrelation_values = []
    # for i in range(0,len(fixed_environments)):
    #   correlation_coefficient, p_value = pearsonr(fitness,fitnesses_for_fixed_environments[i])
    #   autocorrelation_values = np.append(autocorrelation_values, correlation_coefficient)
    # plt.plot(fixed_environments, autocorrelation_values)
    # plt.title('Correlation with line')
    # plt.xlabel('x values')
    # plt.ylabel('Correlation')
    # plt.ylim([-1.1, 1.1])
    # plt.axis('square')
    # plt.show()

    # Show the plots
    # plt.tight_layout()
    plt.savefig("Fisher_geometric_model.pdf", dpi=300, bbox_inches='tight')
    # plt.savefig("DFEs.pdf", dpi=300, bbox_inches='tight')
    plt.show()



# Create sliders for random_point_x and random_point_y
random_point_x_slider = FloatSlider(min=-2.0, max=2.0, step=0.1, value=0, description='Random Point X:')
random_point_y_slider = FloatSlider(min=-2.0, max=2.0, step=0.1, value=1, description='Random Point Y:')

# Use interact to create the GUI
interact(plot_and_histogram, mu_x=random_point_x_slider, mu_y=random_point_y_slider);