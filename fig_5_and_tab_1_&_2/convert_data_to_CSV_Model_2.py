import csv

import numpy as np


def level_of_sexual_dimophism(trait_f, trait_m):
    trait_max = max(trait_f, trait_m)
    trait_min = min(trait_f, trait_m)
    if trait_max > 0:
        return (trait_max - trait_min) / trait_max
    else:
        return 0


header = ['gamma_cs1', 'gamma_cs2', 'gamma_cd1', 'gamma_cd2', 'betaD', 'betaN', 'betaL', 'betaE', 'l', 'rho', 'PD0',
          'p', 'R0', 'sf', 'df', 'sm', 'dm', 'LSDs', 'LSDd', 'Theta', 'P_killed_male', 'P_killed_female']
data = []

list_file_name = ['results/BRUTresults/results_Model_2_' + str(i) for i in range(1, 11)]
# Adjust the range based on the number of results files you have

for file_name in list_file_name:

    counter = 0
    with open(file_name, 'r') as file:
        for line in file:
            counter += 1
            if counter == 10000:
                break
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
            sf = float(list_sym_value[13].split('=')[1])
            df = float(list_sym_value[14].split('=')[1])
            sm = float(list_sym_value[15].split('=')[1])
            dm = float(list_sym_value[16].split('=')[1][:-1])
            LSDs = level_of_sexual_dimophism(sf, sm)
            LSDd = level_of_sexual_dimophism(df, dm)

            ancestrally_defended = (p * PD0 * betaE) / (1 - p + p * (1 - PD0)) > gamma_cd1

            # Compute the probability of being detected for females and males
            PDf = PD0 + (1 - PD0) * (1 - np.exp(- betaD * sf))
            PDm = PD0 + (1 - PD0) * (1 - np.exp(- betaD * sm))

            # Compute the probability that the females and males signal is noticed by predators
            PNf = 1 - np.exp(- betaN * sf)
            PNm = 1 - np.exp(- betaN * sm)

            # Compute the probability that the predators learn to avoid the signal after attacking a female and a male
            PLf = 1 - np.exp(- betaL * df)
            PLm = 1 - np.exp(- betaL * dm)

            # Proportion of naive predators
            theta = 1 - l / (p / 2 * (PDf * PNf * PLf + PDm * PNm * PLm) + l)

            # Compute the probability that a predator attacks a female and a male
            PAf = (1 - PNf) + PNf * (1 - theta)
            PAm = (1 - PNm) + PNm * (1 - theta)

            # Compute the probability that a female and a male escape after an attack
            PEf = 1 - np.exp(- betaE * df)
            PEm = 1 - np.exp(- betaE * dm)

            # Probability a male and a female gets killed by predators
            P_killed_male = p * PDm * PAm * (1 - PEm)
            P_killed_female = p * PDf * PAf * (1 - PEf)

            data.append(
                [gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, R0, sf, df, sm,
                 dm, LSDs, LSDd, theta, P_killed_male, P_killed_female])

    with open('results/results_Model_2.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
