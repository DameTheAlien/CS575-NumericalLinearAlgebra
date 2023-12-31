# -*- coding: utf-8 -*-
"""HW5Code_DamianFranco.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H3eYYbgzWlNYHJVqebEi6WGgqJjTpZC9

# Homework 4
## Damian Franco
## CS-575
"""

# Commented out IPython magic to ensure Python compatibility.
# Importing the required modules
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import time
from prettytable import PrettyTable
import scipy
import scipy.linalg 
from pprint import pprint
# %matplotlib inline
import math

# Saves the current plot to desktop since working in Google Colab
from google.colab import files
#plt.savefig("my_plot.png", bbox_inches='tight', dpi=300)
#files.download("my_plot.png")

"""# Molecular Weights (#1)"""

# W_a,b = W_N*a + W_O*b
# W_a,b = a*(14.007) + b*(15.999)
# W_N = 14.007
# W_O = 15.999
W_N = np.array([1, 2, 1, 2, 2, 2])
W_O = np.array([1, 1, 2, 3, 4, 5])
W_AB_ALL = np.array([30.006, 44.013, 46.006, 76.012, 92.011, 108.010])

# Create coefficent matrix
A = np.vstack([W_N, W_O, np.ones(len(W_N))]).T
print(A)

# Fit the system using least squares
m1, m2, c = np.linalg.lstsq(A, W_AB_ALL, rcond=None)[0]

print('W_N Estimate:', m1)
print('W_O Estimate:', m2)
print('c Estimate:', c)

# Plot the data and the line of best fit
plt.plot(W_N, W_AB_ALL, 'o', label='Nitrogin Data')
plt.plot(W_N,  m2*W_O+m1*W_N + c + c, 'r', label='Best fit')
plt.legend()
plt.show()

# Plot the data and the line of best fit
plt.plot(W_O, W_AB_ALL, 'o', label='Oxygen data')
plt.plot(W_O, m2*W_O+m1*W_N + c, 'r', label='Best fit')
plt.legend()
plt.show()