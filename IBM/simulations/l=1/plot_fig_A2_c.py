from functions.visualize_data import *
from functions.dynamics import *
from functions.classes import *

gamma_cs1, gamma_cs2 = 0.01, 0.35
gamma_cd1, gamma_cd2 = 0.08, 0.01
betaD = 1.25
betaN = 17.5
betaL = 0.5
betaE = 0.9
l = 1
rho = 1.2
PD0 = 0.25
p = 0.3
N = 2 * int(10 ** 2)
P_mut = 0.02
mutation_step_s = 0.0005
mutation_step_d = 0.0005

E = Environment(gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut,
                mutation_step_s, mutation_step_d)

data_folder = 'results_1'
g0, G = open_last_generation(data_folder, E)

def get_sex_ratio(G):
    Nf = 0
    for i in G:
        if i.sex == 'female':
            Nf += 1
    return Nf / len(G)

def get_change_sex_ratio(G):
    # Extract parameters from the first individual's environment
    E = G[0].E
    gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d = E.gamma_cs1, E.gamma_cs2, E.gamma_cd1, E.gamma_cd2, E.betaD, E.betaN, E.betaL, E.betaE, E.l, E.rho, E.PD0, E.p, E.N, E.P_mut, E.mutation_step_s, E.mutation_step_d

    L_sex_ratio = [get_sex_ratio(G)]
    L_time = [0]

    ##1
    t = 0
    while t < 1:
        theta = get_theta(G, E)
        list_death_rate = [ind.get_death_rate(theta) for ind in G]
        sum_death_rates = sum(list_death_rate)
        average_time_death = 1 / sum_death_rates
        t = t + np.random.exponential(average_time_death, size=None)
        if t < 1:
            p_death = list_death_rate / sum_death_rates
            dead_ind = np.random.choice(G, p=p_death)
            G.remove(dead_ind)
            L_sex_ratio.append(get_sex_ratio(G))
            L_time.append(t)
    L_sex_ratio.append(L_sex_ratio[-1])
    L_time.append(1)
    return L_sex_ratio, L_time

L_sex_ratio, L_time = get_change_sex_ratio(G)

SMALL_SIZE = 19
MEDIUM_SIZE = 23
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if not os.path.exists('figures'):
    os.mkdir('figures')

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i in range(len(L_sex_ratio)-1):
    T = np.linspace(L_time[i], L_time[i+1], 10)
    S = [L_sex_ratio[i]] * 10
    plt.plot(T, S, color="#EE220C", linewidth=3)
plt.savefig('figures/fig_A2_c.pdf')