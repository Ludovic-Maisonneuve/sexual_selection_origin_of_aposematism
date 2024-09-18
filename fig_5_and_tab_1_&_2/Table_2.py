import numpy as np
import pandas as pd
from tabulate import tabulate

# Read data from a CSV file
data = pd.read_csv('results/results_Model_2.csv')


# Define a function to calculate the correlation coefficient between two variables
def get_r(cov_matrix, t1, t2):
    return cov_matrix.loc[t1, t2] / (np.sqrt(cov_matrix.loc[t1, t1] * cov_matrix.loc[t2, t2]))


# Calculate the covariance matrix of the entire dataset
cov_matrix = data.cov()

# List of parameters
list_parameters = ['gamma_cs1', 'gamma_cs2', 'gamma_cd1', 'gamma_cd2', 'betaD', 'betaN', 'betaL', 'betaE', 'l', 'rho',
                   'PD0', 'p']
# List of variables
list_vars = ['sf', 'df', 'sm', 'dm', 'LSDs', 'LSDd', 'Theta', 'P_killed_male', 'P_killed_female']

list_cov_par = []

for i, par1 in enumerate(list_parameters):
    for j, par2 in enumerate(list_parameters):
        if i > j:
            list_cov_par.append(np.abs(get_r(cov_matrix, par1, par2)))

# Print the mean of the absolute correlation coefficients

print('mean of the absolute correlation coefficients', np.mean(list_cov_par))
table = np.zeros((len(list_parameters), len(list_vars)))

for i, par in enumerate(list_parameters):
    for j, var in enumerate(list_vars):
        table[i, j] = get_r(cov_matrix, par, var)

print('r_rho_theta', get_r(cov_matrix, 'rho', 'Theta'), 'a')

print('r_LSDs_theta', get_r(cov_matrix, 'LSDs', 'Theta'), 'a')
print('r_LSDd_theta', get_r(cov_matrix, 'LSDd', 'Theta'), 'a')

print('table')
print(tabulate(table, tablefmt="latex", floatfmt=".5f"))
