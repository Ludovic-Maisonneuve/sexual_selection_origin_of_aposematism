import random

from functions.classes import *

def get_theta(G, E):
    gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d = E.gamma_cs1, E.gamma_cs2, E.gamma_cd1, E.gamma_cd2, E.betaD, E.betaN, E.betaL, E.betaE, E.l, E.rho, E.PD0, E.p, E.N, E.P_mut, E.mutation_step_s, E.mutation_step_d

    S = 0
    Ne = len(G)
    for ind in G:
        PD = PD0 + (1 - PD0) * (1 - np.exp(-betaD * ind.s))
        PN = 1 - np.exp(-betaN * ind.s)
        PL = 1 - np.exp(-betaL * ind.d)
        S += p * PD * PN * PL / Ne

    return S / (l + S)


# Function to generate the next generation of individuals
def next_generation(G, sm_evolve=True, dm_evolve=True, sf_evolve=True, df_evolve=True):
    # Extract parameters from the first individual's environment
    E = G[0].E
    gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d = E.gamma_cs1, E.gamma_cs2, E.gamma_cd1, E.gamma_cd2, E.betaD, E.betaN, E.betaL, E.betaE, E.l, E.rho, E.PD0, E.p, E.N, E.P_mut, E.mutation_step_s, E.mutation_step_d

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
    list_surviving = G
    # t = 0
    # d_bounded = p + gamma_cs1 * 1 + gamma_cs2 * 1**2 + gamma_cd1 * 1 + gamma_cd2 * 1**2
    # while t < 1:
    #    Ne = len(G)
    #    theta = get_theta(G, E)
    #    t = t + np.random.exponential(1, size=None) / (d_bounded * Ne)
    #    if t <1:
    #        i = np.random.randint(0, Ne)
    #        m = G[i].get_death_rate(theta) / d_bounded
    #        if np.random.random() < m:
    #            G.remove(G[i])
    # list_surviving = G

    ##2
    # list_surviving = []
    # for ind in G:
    #     t = np.random.exponential(1 / ind.get_death_rate(theta), size=None)
    #     if t > 1:
    #         list_surviving.append(ind)

    ## P t


    ##3
    # list_surviving = []
    # for ind in G:
    #     if np.random.random() < np.exp(-ind.get_death_rate(theta)):
    #         list_surviving.append(ind)


    list_surviving_male = []
    list_surviving_female = []
    for ind in list_surviving:
        if ind.sex == 'male':
            list_surviving_male.append(ind)
        else:
            list_surviving_female.append(ind)

    N_female = len(list_surviving_female)
    mean_male_s = np.mean([ind.s for ind in list_surviving_male])
    p_mated = [ind.get_prob_mated(mean_male_s) for ind in list_surviving_male]
    p_mated = p_mated / np.sum(p_mated)
    list_mated_male =  list(np.random.choice(list_surviving_male, N_female, p=p_mated))

    list_index_mated_pair = [i for i in range(N_female)]
    list_index_parent = list(np.random.choice(list_index_mated_pair, N))
    G_offspring = []
    N_mut = np.random.binomial(N, P_mut)
    for ni, index in enumerate(list_index_parent):
        sm_offspring = np.random.choice([list_mated_male[index].sm, list_surviving_female[index].sm])
        dm_offspring = np.random.choice([list_mated_male[index].dm, list_surviving_female[index].dm])
        sf_offspring = np.random.choice([list_mated_male[index].sf, list_surviving_female[index].sf])
        df_offspring = np.random.choice([list_mated_male[index].df, list_surviving_female[index].df])

        if ni <= N_mut:  # If within the mutation quota
            # Mutate traits if within mutation probability
            if sm_evolve:  # Mutate v if it evolves
                sm_offspring = max(random.gauss(sm_offspring, mutation_step_s), 0)
            if dm_evolve:  # Mutate o if it evolves
                dm_offspring = max(random.gauss(dm_offspring, mutation_step_d), 0)
            if sf_evolve:  # Mutate v if it evolves
                sf_offspring = max(random.gauss(sf_offspring, mutation_step_s), 0)
            if df_evolve:  # Mutate o if it evolves
                df_offspring = max(random.gauss(df_offspring, mutation_step_d), 0)

        G_offspring.append(Individual(sm_offspring, dm_offspring, sf_offspring, df_offspring, E))

    return G_offspring