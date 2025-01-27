from functions.visualize_data import *

for i in range(1, 11):
    save_data_for_mathematica('results_' + str(i), 'Mathematica_data/r' + str(i), ngmin=150000,
                              ngmax=None)