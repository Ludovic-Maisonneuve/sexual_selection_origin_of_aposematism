import csv

import numpy as np

header = ['gamma_cs1', 'gamma_cs2', 'gamma_cd1', 'gamma_cd2', 'betaD', 'betaN', 'betaL', 'betaE', 'l', 'rho', 'PD0',
          'p', 'R0', 's', 'd', 'theta', 'P_killed']
data = []

list_file_name = ['results/BRUTresults/results_Model_1_' + str(i) for i in range(1, 11)]
# Adjust the range based on the number of results files you have

for file_name in list_file_name:

    counter = 0
    with open(file_name, 'r') as file:
        for line in file:
            counter += 1
            if counter == 100001:
                break
            list_sym_value = line.split(' ')
            gamma_cs1 = float(list_sym_value[0].split('=')[1])
            gamma_cs2 = float(list_sym_value[1].split('=')[1])
            gamma_cd1 = float(list_sym_value[2].split('=')[1])
            gamma_cd2 = float(list_sym_value[3].split('=')[1])
            betaD = float(list_sym_value[4].split('=')[1])
            betaN = float(list_sym_value[5].split('=')[1])
            betaL = float(list_sym_value[6].split('=')[1])
            betaE = float(list_sym_value[7].split('=')[1])
            l = float(list_sym_value[8].split('=')[1])
            rho = float(list_sym_value[9].split('=')[1])
            PD0 = float(list_sym_value[10].split('=')[1])
            p = float(list_sym_value[11].split('=')[1])
            R0 = float(list_sym_value[12].split('=')[1])
            s = float(list_sym_value[13].split('=')[1])
            d = float(list_sym_value[14].split('=')[1])

            ancestrally_defended = (p * PD0 * betaE) / (1 - p + p * (1 - PD0)) > gamma_cd1

            # Compute the probability of being detected
            PD = PD0 + (1 - PD0) * (1 - np.exp(- betaD * s))

            # Compute the probability that the prey signal is noticed by predators
            PN = 1 - np.exp(- betaN * s)

            # Compute the probability that the predators learn to avoid the signal after attacking a prey
            PL = 1 - np.exp(- betaL * d)

            # Proportion of naive predators
            theta = 1 - l / (p * (PD * PN * PL) + l)

            # Compute the probability that a predator attacks a prey
            PA = (1 - PN) + PN * (1 - theta)

            # Compute the probability that a prey escape after an attack
            PE = 1 - np.exp(- betaE * d)

            # Probability a prey gets killed by predators
            P_killed = p * PD * PA * (1 - PE)

            data.append(
                [gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, R0, s, d,
                 theta, P_killed])

    with open('results/results_Model_1.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
