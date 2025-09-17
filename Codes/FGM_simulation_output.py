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
This script generates a histogram to visualize and compare the distribution
of the variance of fitness values for two distinct categories of environments:
'Stress' and 'Non-stress'. These environmental coordinates are produces using
the scheme presented in the manuscript. Check the "Environmental_configuration
_figure_generation.py" for the graphical representation.It uses matplotlib to
create the plot and saves the resulting figure as a PDF file.This code might 
take a while to run and generate the plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from scipy.stats import pearsonr


np.random.seed(786)
# Function to generate a random positive definite covariance matrix
def random_covariance_matrix(N):
    A = np.random.rand(N, N)
    return np.dot(A, A.T)  # Ensure the matrix is positive definite

# Function to generate samples from a multivariate normal distribution
def generate_mutation(mean, cov, num_samples):
    return np.random.multivariate_normal(mean, cov, num_samples)

# Function to compute the PDF of a multivariate normal distribution
def multivariate_normal_pdf(x, mean, cov):
    N = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / ((2 * np.pi) ** (N / 2) * np.sqrt(det_cov))
    diff = x - mean
    exponent = -1 * np.dot(np.dot(diff.T, inv_cov), diff)**(1/4)
    return norm_const * np.exp(exponent)

# Parameters
N = 51  # Number of variables
num_samples = 10000  # Number of samples to generate

# Generate random covariance matrices and sampling means
S = random_covariance_matrix(N)  # Random covariance matrix for the original distribution
S = np.eye(N)
Y = np.random.rand(N)*0  # Random means for the sampling distribution
# M = random_covariance_matrix(N)  # Random covariance matrix for the sampling distribution
M = np.eye(N)*0.1

sampled_points = generate_mutation(Y, M, num_samples)

number_of_stressed_environment = N-1
number_of_non_stressed_environment = N-1
number_of_environment = number_of_stressed_environment + number_of_non_stressed_environment

non_stressed_environments = np.random.uniform(0,2,number_of_stressed_environment)

# A = np.transpose(np.array([fixed_environments,np.zeros(5)]))
non_stressed_coordinates = np.transpose(np.concatenate((np.array([non_stressed_environments]),
                                            0.4*np.random.uniform(low=-1, high=0, size=(N-1, len(non_stressed_environments)))), axis = 0))
# print(non_stressed_coordinates.shape)
# stressed_coordinates = np.diag(np.random.rand(number_of_stressed_environment+1))
stressed_coordinates = np.diag(np.random.uniform(low=3, high=5, size=number_of_stressed_environment+1))
stressed_coordinates2 = 0.25*np.random.uniform(-1,1,(number_of_stressed_environment+1,number_of_stressed_environment+1))
stressed_coordinates = stressed_coordinates+stressed_coordinates2
# print(stressed_coordinates.shape)
all_environment = np.concatenate((non_stressed_coordinates,stressed_coordinates[1:,:]), axis = 0)

# print(all_environment)
fitnesses_for_all_environments = np.zeros(((number_of_environment),num_samples))
for i in range(0,number_of_environment):
  sample_values = np.zeros(num_samples)
  for j in range(0,num_samples):
    sample_values[j] = multivariate_normal_pdf(sampled_points[j],all_environment[i], S)
  max_value = multivariate_normal_pdf(np.zeros(N),all_environment[i], S)
  # print(max_value)
  fitness = np.log(sample_values/max_value)
  fitnesses_for_all_environments[i]=fitness


corr1 = []
corr2 = []
corr3 = []
for i in range(0,number_of_non_stressed_environment):
  for j in range(0,i):
    correlation_coefficient, p_value = pearsonr(fitnesses_for_all_environments[i],fitnesses_for_all_environments[j])
    corr1 = np.append(corr1,correlation_coefficient)
for i in range(0,number_of_non_stressed_environment):
  for j in range(number_of_non_stressed_environment,number_of_environment):
    correlation_coefficient, p_value = pearsonr(fitnesses_for_all_environments[i],fitnesses_for_all_environments[j])
    corr2 = np.append(corr2,correlation_coefficient)
for i in range(number_of_non_stressed_environment,number_of_environment):
  for j in range(number_of_non_stressed_environment,i):
    correlation_coefficient, p_value = pearsonr(fitnesses_for_all_environments[i],fitnesses_for_all_environments[j])
    corr3 = np.append(corr3,correlation_coefficient)
    


# plt.hist(std, bins=100, edgecolor='black')
binss = 30
plt.figure(figsize=(8, 8))
plt.hist(corr1, bins=25, color='green', alpha=0.7, label='Non-stress vs Non-stress',edgecolor='black', density=False)
plt.hist(corr2, bins=25, color='blue', alpha=0.5, label='Non-stress vs Stress', edgecolor='black', density=False)
plt.hist(corr3, bins=25, color='red', alpha=0.6, label='Stress vs Stress',edgecolor='black', density=False)
plt.title('Distribution of Correlation Coefficients', fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of pair of environments')
LLLL=plt.legend()
LLLL.set_title('Environment Pairs')
plt.savefig("correlation_simulation.pdf", dpi=300, bbox_inches='tight')
plt.show()

# plt.hist(difference_between_mean, bins=100, edgecolor='black')
mean_for_data = np.zeros(number_of_environment)
std_for_data = np.zeros(number_of_environment)
for i in range(0,number_of_environment):
  mean_for_data[i] = np.mean(fitnesses_for_all_environments[i])
  std_for_data[i] = np.std(fitnesses_for_all_environments[i])

plt.figure(figsize=(12, 8))
# plt.subplot(211)
# plt.hist([mean_for_data[0:number_of_non_stressed_environment],mean_for_data[number_of_non_stressed_environment:]], bins=20, color=['green', 'red'], alpha=0.5, label=['Non-stress','Stress'],edgecolor='black', density=False)
plt.hist(mean_for_data[0:number_of_non_stressed_environment], bins=10, color='green', alpha=0.5, label='Non-stress',edgecolor='black', density=False)
plt.hist(mean_for_data[number_of_non_stressed_environment:], bins=5, color='red', alpha=0.7, label='Stress', edgecolor='black', density=False)
plt.title('Distribution of Mean of Fitness Values', fontweight = 'bold')
plt.xlabel('Mean')
plt.ylabel('Number of environments')
LLLL = plt.legend()
LLLL.set_title('Environments')
# plt.subplot(212)
# plt.hist(std_for_data[0:number_of_non_stressed_environment], bins=20, color='blue', alpha=0.3, label='N',edgecolor='black', density=True)
# plt.hist(std_for_data[number_of_non_stressed_environment:], bins=20, color='green', alpha=0.5, label='S', edgecolor='black', density=True)
# plt.title('Histogram of standard deviation from different types of samples')
# plt.xlabel(' std fitness')
# plt.ylabel('Frequency')
# plt.legend()
plt.savefig("mean_simulation.pdf", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8))
# plt.subplot(211)
# plt.hist([std_for_data[0:number_of_non_stressed_environment],std_for_data[number_of_non_stressed_environment:]], bins=10, color=['green', 'red'],rwidth=1, alpha=0.5, label=['Non-stress','Stress'],edgecolor='black', density=False)
plt.hist(std_for_data[0:number_of_non_stressed_environment], bins=3, color='green', alpha=0.5, label='Non-stress',edgecolor='black', density=False)
plt.hist(std_for_data[number_of_non_stressed_environment:], bins=8, color='red', alpha=0.7, label='Stress', edgecolor='black', density=False)
plt.title('Distribution of Variance of Fitness Values', fontweight = 'bold')
plt.xlabel('Variance')
plt.ylabel('Number of environments')
LLLL = plt.legend()
LLLL.set_title('Environments')
# plt.subplot(212)
# plt.hist(std_for_data[0:number_of_non_stressed_environment], bins=20, color='blue', alpha=0.3, label='N',edgecolor='black', density=True)
# plt.hist(std_for_data[number_of_non_stressed_environment:], bins=20, color='green', alpha=0.5, label='S', edgecolor='black', density=True)
# plt.title('Histogram of standard deviation from different types of samples')
# plt.xlabel(' std fitness')
# plt.ylabel('Frequency')
# plt.legend()
plt.savefig("standard_deviation_simulation.pdf", dpi=300, bbox_inches='tight')
plt.show()