import time
from functions.visualize_data import *
from functions.dynamics import *
from functions.classes import *

gamma_cs1, gamma_cs2 = 0.01, 0.35
gamma_cd1, gamma_cd2 = 0.08, 0.01
betaD = 1.25
betaN = 17.5
betaL = 0.5
betaE = 0.9
l = 10
rho = 1.2
PD0 = 0.25
p = 0.3
N = 2 * int(10 ** 2)
P_mut = 0.02
mutation_step_s = 0.0005
mutation_step_d = 0.0005

## Initial conditions
sm_0 = 0.766131
dm_0 = 0.772154
sf_0 = 0
df_0 = 0

## Number of generations
gmax = 250000

E = Environment(gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d)

for replicate in range(1,11):
    print('replicate', replicate)
    data_folder = 'results_' + str(replicate)

    ## Creating the ancestral population
    G = initial_generation(sm_0, dm_0, sf_0, df_0, E)

    ## Return to previous simulation
    if os.path.exists(data_folder + '/G'):
        g0, G = open_last_generation(data_folder, E)
    else:
        g0 = 0
        save_generation(G, 0, data_folder)

    nb_gen_print = 100

    tic = time.perf_counter()
    for gi in range(g0 + 1, gmax + 1):
        G = next_generation(G)
        if gi % 10 == 0:
            save_generation(G, gi, data_folder)
        if gi % nb_gen_print== 0:
            toc = time.perf_counter()
            print('##', 'Generation:', gi, '##')
            print(f"get {nb_gen_print:0.4f} iterations in {toc - tic:0.4f} seconds")
            print('##   ##')
            tic = time.perf_counter()
