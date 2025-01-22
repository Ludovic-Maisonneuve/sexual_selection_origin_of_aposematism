import numpy as np


class Environment:
    def __init__(self, gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d):
        # Initialize environment parameters
        self.gamma_cs1, self.gamma_cs2, self.gamma_cd1, self.gamma_cd2, self.betaD, self.betaN, self.betaL, self.betaE, self.l, self.rho, self.PD0, self.p, self.N, self.P_mut, self.mutation_step_s, self.mutation_step_d = gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d

class Individual:
    def __init__(self, sm, dm, sf, df, E, d=None, s=None):
        # Initialize individual parameters
        if np.random.random() < 1/2:
            self.sex = 'male'
            self.s, self.d = sm, dm
        else:
            self.sex = 'female'
            self.s, self.d = sf, df

        self.sm, self.dm, self.sf, self.df = sm, dm, sf, df
        self.E = E

    def get_death_rate(self, theta):
        # Get parameters' values
        E = self.E
        gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d = E.gamma_cs1, E.gamma_cs2, E.gamma_cd1, E.gamma_cd2, E.betaD, E.betaN, E.betaL, E.betaE, E.l, E.rho, E.PD0, E.p, E.N, E.P_mut, E.mutation_step_s, E.mutation_step_d
        s, d = self.s, self.d
        d_C = gamma_cs1 * s + gamma_cs2 * s ** 2 + gamma_cd1 * d + gamma_cd2 * d ** 2

        PD = PD0 + (1 - PD0) * (1 - np.exp(-betaD * s))
        PN = 1 - np.exp(-betaN * s)
        PA = (1 - PN) + PN * (1 - theta)
        PE = 1 - np.exp(-betaE * d)
        d_P = p * PD * PA * (1 - PE)

        return d_C + d_P
    def print_S_df(self, theta):
        # Get parameters' values
        E = self.E
        gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d = E.gamma_cs1, E.gamma_cs2, E.gamma_cd1, E.gamma_cd2, E.betaD, E.betaN, E.betaL, E.betaE, E.l, E.rho, E.PD0, E.p, E.N, E.P_mut, E.mutation_step_s, E.mutation_step_d
        s, d = self.sf, self.df
        d_C = gamma_cs1 * s + gamma_cs2 * s ** 2 + gamma_cd1 * d + gamma_cd2 * d ** 2

        PD = PD0 + (1 - PD0) * (1 - np.exp(-betaD * s))
        PN = 1 - np.exp(-betaN * s)
        PA = (1 - PN) + PN * (1 - theta)
        PE = 1 - np.exp(-betaE * d)
        d_P = p * PD * PA * (1 - PE)
        print(- gamma_cd1 - 2 * gamma_cd2 * d + p * (betaE * PD * (1 - PE) * PA))

    def get_prob_mated(self, mean_male_s):
        E = self.E
        gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, N, P_mut, mutation_step_s, mutation_step_d = E.gamma_cs1, E.gamma_cs2, E.gamma_cd1, E.gamma_cd2, E.betaD, E.betaN, E.betaL, E.betaE, E.l, E.rho, E.PD0, E.p, E.N, E.P_mut, E.mutation_step_s, E.mutation_step_d
        return 1 / (1 + np.exp(- rho * (self.s - mean_male_s)))


def initial_generation(sm_0, dm_0, sf_0, df_0, E):
    # Generate a list of individuals with the specified parameters
    G = [Individual(sm_0, dm_0, sf_0, df_0, E) for i in range(E.N)]
    return G
