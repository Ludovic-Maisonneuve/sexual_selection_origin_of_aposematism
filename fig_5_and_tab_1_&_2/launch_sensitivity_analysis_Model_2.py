import os
import random
import time

from functions import *

# Folder to store simulation BRUTresults
data_folder = 'BRUTresults'
# Check if the folder exists
if not os.path.exists(data_folder):
    # If it doesn't exist, create the folder
    os.makedirs(data_folder)

# Specify the file name to save BRUTresults. You can run multiple files in parallel by updating the number in the file name.
name_file = 'results_Model_2_1'
# Set the total number of runs
n_runs = 100000

# Start measuring time
start_time = time.time()

# Loop through the specified number of runs
for i in range(n_runs):
    # Generate random values for model parameters
    gamma_cs1 = np.exp(random.uniform(-2, 2) * np.log(10))
    gamma_cs2 = np.exp(random.uniform(-2, 2) * np.log(10))
    gamma_cd1 = np.exp(random.uniform(-2, 2) * np.log(10))
    gamma_cd2 = np.exp(random.uniform(-2, 2) * np.log(10))
    betaD = np.exp(random.uniform(-2, 2) * np.log(10))
    betaN = np.exp(random.uniform(-2, 2) * np.log(10))
    betaL = np.exp(random.uniform(-2, 2) * np.log(10))
    betaE = np.exp(random.uniform(-2, 2) * np.log(10))
    l = np.exp(random.uniform(-2, 2) * np.log(10))
    rho = np.exp(random.uniform(-2, 2) * np.log(10))
    PD0 = random.uniform(0, 1)
    p = random.uniform(0, 0.5)
    R0 = random.uniform(0, 1)

    args = (gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, R0)

    # Calculate equilibrium strategy
    X = get_equilibrium_strategy(args)

    # Save the BRUTresults for each run
    save_run_Model_2(X, args, name_file)

    # Print the number of runs every 100 runs
    if (i + 1) % 100 == 0:
        # End measuring time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Completed {i + 1} runs, the last 100 runs were completed in {elapsed_time:.2f} seconds.")
        # Start measuring time
        start_time = time.time()

print("All runs completed.")
