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
This script performs an analysis of E. coli fitness data from an Excel file.
It visualizes the distribution of fitness effects (DFE) under different
environmental conditions (stress and non-stress), calculates the Pearson
correlation coefficients between these environments, and plots the
distributions of these correlations. It also analyzes and plots the
distributions of the mean and variance of fitness values for both stress
and non-stress environments.

Note: This script is designed for use in a Jupyter Notebook or a similar
environment that supports inline plotting with matplotlib. While it imports
ipywidgets, the current version does not feature interactive components.
"""

import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from scipy.stats import pearsonr

# plt.rcParams['font.serif'] = 'Times New Roman'
# Load the Excel file
file_path = "./EColi2.xlsx"
df = pd.read_excel(file_path)

# Extract numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

whole_list = [120,118,119,121,122,109,159,99,93,91,92,95,98,97,96,94,130,160,47,45,151,149,150,1,35,147,17,35,13,29,27,25,51,9,37,11,43,5,7,79,61,77,69,73,63,71,75,65,21,19,39,67,53,49,57,3,157,155,83,85,89,87,33,23,15,41,31,110,143,142,141,140,124,123,107,105,106,100,101,137,136,139,116,111,132,134,133,128,115,114,103,131,129,161,104,144,102,112,113,125,126]
stressed = [120,118,119,121,122,109,159,99,93,91,92,95,98,97,96,94,130,160,157,155,110,143,142,141,140,124,123,107,105,106,100,101,137,136,139,116,111,132,134,133,128,115,114,103,131,129,161,104,144,102,112,113,125,126]
not_stressed = [47,45,151,149,150,1,35,147,17,35,13,29,27,25,51,9,37,11,43,5,7,79,61,77,69,73,63,71,75,65,21,19,39,67,53,49,57,3,83,85,89,87,33,23,15,41,31]


plt.figure(figsize=(12, 12))
column_index = 98
plt.subplot(221)
for column_index in [98]:
  plt.hist(df[numerical_columns[column_index]], bins=30, alpha=1, color='red', edgecolor='black',density='False')
  plt.xlabel('Relative fitness')
  plt.ylabel('Frequency')
  plt.title(f'Stress Environments', fontweight = 'bold')
  plt.grid(True)
  # plt.xscale('log')
  # plt.yscale('log')
plt.subplot(222)
for column_index in [51]:
  plt.hist(df[numerical_columns[column_index]], bins=30, alpha=1, color='green', edgecolor='black',density='False')
  plt.xlabel('Relative fitness')
  plt.ylabel('Frequency')
  plt.title(f'Non-stress Environments', fontweight = 'bold')
  plt.grid(True)
  # plt.xscale('log')
  # plt.yscale('log')
plt.subplot(223)
for column_index in [130]:
  plt.hist(df[numerical_columns[column_index]], bins=30, alpha=1, color='red', edgecolor='black',density='False')
  plt.xlabel('Relative fitness')
  plt.ylabel('Frequency')
  # plt.title(f'Stress Environment', fontweight = 'bold')
  plt.grid(True)
  # plt.xscale('log')
  # plt.yscale('log')
plt.subplot(224)
for column_index in [69]:
  plt.hist(df[numerical_columns[column_index]], bins=30, alpha=1, color='green', edgecolor='black',density='False')
  plt.xlabel('Relative fitness')
  plt.ylabel('Frequency')
  # plt.title(f'Non-stress Environments', fontweight = 'bold')
  plt.grid(True)
  # plt.xscale('log')
  # plt.yscale('log')
plt.savefig("exp_examples.pdf", dpi=300, bbox_inches='tight')
plt.show()


bacteria_data = np.zeros((len(numerical_columns),3789))
for i in range(0,len(numerical_columns)):
  bacteria_data[i] = np.array(df[numerical_columns[i]])

correlation_of_data_SS = []
correlation_of_data_NS = []
correlation_of_data_NN = []
correlation_of_data_SS_2d = np.eye(len(stressed))
correlation_of_data_NS_2d = np.eye(len(not_stressed))

for i in range(0,len(stressed)):
  for j in range(0,i):
    correlation_coefficient, p_value = pearsonr(bacteria_data[stressed[i]],bacteria_data[stressed[j]])
    correlation_of_data_SS.append(correlation_coefficient)
      # print(i,j)
    correlation_of_data_SS_2d[i,j] = correlation_coefficient
    correlation_of_data_SS_2d[j,i] = correlation_coefficient

for i in range(0,len(stressed)):
  for j in range(0,len(not_stressed)):
    correlation_coefficient, p_value = pearsonr(bacteria_data[stressed[i]],bacteria_data[not_stressed[j]])
    correlation_of_data_NS.append(correlation_coefficient)
      # print(i,j)

for i in range(0,len(not_stressed)):
  for j in range(0,i):
    correlation_coefficient, p_value = pearsonr(bacteria_data[not_stressed[i]],bacteria_data[not_stressed[j]])
    correlation_of_data_NN.append(correlation_coefficient)
    correlation_of_data_NS_2d[i,j] = correlation_coefficient
    correlation_of_data_NS_2d[j,i] = correlation_coefficient
      # print(i,j)



binss = 40
plt.figure(figsize=(14, 14))
plt.hist(np.array(correlation_of_data_SS),   bins=binss, color='red', alpha=0.7, label='Stress vs Stress',edgecolor='black', density=False)
plt.hist(np.array(correlation_of_data_NS),   bins=binss, color='blue', alpha=0.5, label='Non-stress vs Stress',edgecolor='black', density=False)
plt.hist(np.array(correlation_of_data_NN),   bins=binss, color='green', alpha=0.6, label='Non-stress vs Non-stress',edgecolor='black', density=False)
plt.title('Distribution of Correlation Coeffecients between Environment Pairs', fontweight = 'bold')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of Pair of Environments')
LLLL = plt.legend()
LLLL.set_title('Environment Pairs')
plt.savefig("exp_corr_samples.pdf", dpi=300, bbox_inches='tight')
plt.show()



mean_stressed = []
mean_not_stressed = []
std_stressed = []
std_not_stressed = []
for i in range(0,len(stressed)):
  mean_stressed.append(np.mean(bacteria_data[stressed[i]]))
  std_stressed.append(np.std(bacteria_data[stressed[i]]))
for i in range(0,len(not_stressed)):
  mean_not_stressed.append(np.mean(bacteria_data[not_stressed[i]]))
  std_not_stressed.append(np.std(bacteria_data[not_stressed[i]]))
  
binss = 10
plt.figure(figsize=(12, 8))
plt.hist(np.array(mean_not_stressed),   bins=binss, color='green', alpha=0.5, label='Non-stress',edgecolor='black', density=False)
plt.hist(np.array(mean_stressed),   bins=binss, color='red', alpha=0.7, label='Stress',edgecolor='black', density=False)
plt.title('Distribution of Mean of Fitness Values', fontweight = 'bold')
plt.xlabel('Mean')
plt.ylabel('Number of Environments')
LLLL = plt.legend()
LLLL.set_title('Environments')
plt.savefig("exp_mean_samples.pdf", dpi=300, bbox_inches='tight')
plt.show()



binss = 10
plt.figure(figsize=(12, 9))
plt.hist(np.array(std_not_stressed),   bins=binss, color='green', alpha=0.5, label='Non-stress',edgecolor='black', density=False)
plt.hist(np.array(std_stressed),   bins=binss, color='red', alpha=0.7, label='Stress',edgecolor='black', density=False)
plt.title('Distribution of Variance of Fitness Values', fontweight = 'bold')
plt.xlabel('Variance')
plt.ylabel('Number of Environments')
LLLL = plt.legend()
LLLL.set_title('Environments')
plt.savefig("exp_std_samples.pdf", dpi=300, bbox_inches='tight')
plt.show()


