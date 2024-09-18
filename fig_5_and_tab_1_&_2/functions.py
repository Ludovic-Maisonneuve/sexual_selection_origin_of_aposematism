import numpy as np


def get_S(X, args):
    # Unpack input variables
    sf, df, sm, dm = X
    gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, R0 = args

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
    theta = l / (p / 2 * (PDf * PNf * PLf + PDm * PNm * PLm) + l)

    # Compute the probability that a predator attacks a female and a male
    PAf = (1 - PNf) + PNf * theta
    PAm = (1 - PNm) + PNm * theta

    # Compute the probability that a female and a male escape after an attack
    PEf = 1 - np.exp(- betaE * df)
    PEm = 1 - np.exp(- betaE * dm)

    # Compute the female and male probability of not being killed by a predator
    Pof = 1 - p + p * ((1 - PDf) + PDf * (1 - PAf + PAf * PEf))
    Pom = 1 - p + p * ((1 - PDm) + PDm * (1 - PAm + PAm * PEm))

    # Calculate the male reproductive sucess
    Rm = R0 + (1 - R0) * (1 - np.exp(- rho * sm))

    # Calculate the selection gradients for each trait
    s_sf = - gamma_cs1 - 2 * gamma_cs2 * sf \
           + p / Pof * (- betaD * (1 - PDf) * (1 - PEf) * PAf + (1 - theta) * betaN * (1 - PNf) * PDf * (1 - PEf))
    s_df = - gamma_cd1 - 2 * gamma_cd2 * df \
           + p / Pof * (betaE * PDf * (1 - PEf) * PAf)
    s_sm = 1 / (Rm) * (1 - R0) * rho * np.exp(- rho * sm) - gamma_cs1 - 2 * gamma_cs2 * sm + p / Pom * (
            - betaD * (1 - PDm) * (1 - PEm) * PAm + (1 - theta) * betaN * (1 - PNm) * PDm * (1 - PEm))
    s_dm = - gamma_cd1 - 2 * gamma_cd2 * dm + p / Pom * (betaE * PDm * (1 - PEm) * PAm)

    # Return the selection gradients
    return np.array([s_sf / 2, s_df / 2, s_sm / 2, s_dm / 2])


def get_S_mono(X, args):
    # Unpack input variables
    s, d = X
    gamma_cs1, gamma_cs2, gamma_cd1, gamma_cd2, betaD, betaN, betaL, betaE, l, rho, PD0, p, R0 = args

    # Compute the probability of being detected
    PD = PD0 + (1 - PD0) * (1 - np.exp(- betaD * s))

    # Compute the probability that the signal is noticed by predators
    PN = 1 - np.exp(- betaN * s)

    # Compute the probability that the predators learn to avoid the signal after attacking a prey
    PL = 1 - np.exp(- betaL * d)

    # Proportion of naive predators
    theta = l / (p * (PD * PN * PL) + l)

    # Compute the probability that a predator attacks a prey
    PA = (1 - PN) + PN * theta

    # Compute the probability that a prey escape after an attack
    PE = 1 - np.exp(- betaE * d)

    # Compute the female and male probability of not being killed by a predator
    Po = 1 - p + p * ((1 - PD) + PD * (1 - PA + PA * PE))

    # Calculate the male reproductive sucess
    R = R0 + (1 - R0) * (1 - np.exp(- rho * s))

    # Calculate the selection gradients for each trait
    s_s = 1 / (R) * (1 - R0) * rho * np.exp(- rho * s) / 2 - gamma_cs1 - 2 * gamma_cs2 * s + p / Po * (
            - betaD * (1 - PD) * (1 - PE) * PA + (1 - theta) * betaN * (1 - PN) * PD * (1 - PE))
    s_d = - gamma_cd1 - 2 * gamma_cd2 * d + p / Po * (betaE * PD * (1 - PE) * PA)

    # Return the selection gradients
    return np.array([s_s, s_d])


def get_dyn(X, args):
    # Create a tuple for the arguments

    # Calculate the derivatives using the get_S function
    S = get_S(X, args)

    # Update the state variables with a small time step
    Xn = X + 0.001 * np.dot(np.eye(4), S)

    # Ensure state variables are non-negative
    Xn = np.maximum(Xn, 0)

    return Xn


def get_dyn_mono(X, args):
    # Create a tuple for the arguments

    # Calculate the derivatives using the get_S function
    S = get_S_mono(X, args)

    # Update the state variables with a small time step
    Xn = X + 0.001 * np.dot(np.eye(2), S)

    # Ensure state variables are non-negative
    Xn = np.maximum(Xn, 0)

    return Xn


def get_equilibrium_strategy(
        args):  # The loop continues until either the change in traits is below 1e-06 or the maximum number of iterations is reached, whichever comes first
    # Initial traits
    X = np.array([0, 0, 0, 0])  # Initialize an array X to store traits

    dX = 1  # Initialize the change in traits (used to control the loop)
    max_iterations = 1000000  # Set your maximum number of iterations here
    counter = 0  # Initialize a counter to keep track of iterations

    while dX > 1e-06 and counter < max_iterations:
        # Calculate the next set of traits using the get_dyn function
        Xn = get_dyn(X, args)

        # Calculate the change in traits by finding the Euclidean distance
        dX = np.linalg.norm(Xn - X)

        # Update the current traits with the new traits
        X = Xn

        # Increment the counter to keep track of iterations
        counter += 1
        if counter == max_iterations:
            print('not convergence yet')

    return X  # Return the equilibrium traits when the loop exits


def get_equilibrium_strategy_mono(
        args):  # The loop continues until either the change in traits is below 1e-06 or the maximum number of iterations is reached, whichever comes first
    # Initial traits
    X = np.array([0, 0])  # Initialize an array X to store traits

    dX = 1  # Initialize the change in traits (used to control the loop)
    max_iterations = 1000000  # Set your maximum number of iterations here
    counter = 0  # Initialize a counter to keep track of iterations

    while dX > 1e-06 and counter < max_iterations:
        # Calculate the next set of traits using the get_dyn function
        Xn = get_dyn_mono(X, args)

        # Calculate the change in traits by finding the Euclidean distance
        dX = np.linalg.norm(Xn - X)

        # Update the current traits with the new traits
        X = Xn

        # Increment the counter to keep track of iterations
        counter += 1
        if counter == max_iterations:
            print('not convergence yet')

    return X


def save_run_Model_2(X, args, name_file):
    # Define the file path using the provided 'name_file' and a subfolder 'BRUTresults'
    file_path = 'BRUTresults/' + name_file

    # Create a string containing argument names and values separated by spaces
    txt_results = (
            'gamma_cs1=' + str(args[0]) + ' ' +
            'gamma_cs2=' + str(args[1]) + ' ' +
            'gamma_cd1=' + str(args[2]) + ' ' +
            'gamma_cd2=' + str(args[3]) + ' ' +
            'betaD=' + str(args[4]) + ' ' +
            'betaN=' + str(args[5]) + ' ' +
            'betaL=' + str(args[6]) + ' ' +
            'betaE=' + str(args[7]) + ' ' +
            'l=' + str(args[8]) + ' ' +
            'rho=' + str(args[9]) + ' ' +
            'PD0=' + str(args[10]) + ' ' +
            'p=' + str(args[11]) + ' ' +
            'R0=' + str(args[12]) + ' ' +  # Argument names and values

            'sf=' + str(X[0]) + ' ' +
            'df=' + str(X[1]) + ' ' +
            'sm=' + str(X[2]) + ' ' +  # sm
            'dm=' + str(X[3])  # Traits names and values #dm
    )

    # Open the file in append mode and write the BRUTresults string followed by a newline
    with open(file_path, 'a') as file:
        file.write(txt_results + '\n')

    # Close the file
    file.close()


def save_run_Model_1(X, args, name_file):
    # Define the file path using the provided 'name_file' and a subfolder 'BRUTresults'
    file_path = 'BRUTresults/' + name_file

    # Create a string containing argument names and values separated by spaces
    txt_results = (
            'gamma_cs1=' + str(args[0]) + ' ' +
            'gamma_cs2=' + str(args[1]) + ' ' +
            'gamma_cd1=' + str(args[2]) + ' ' +
            'gamma_cd2=' + str(args[3]) + ' ' +
            'betaD=' + str(args[4]) + ' ' +
            'betaN=' + str(args[5]) + ' ' +
            'betaL=' + str(args[6]) + ' ' +
            'betaE=' + str(args[7]) + ' ' +
            'l=' + str(args[8]) + ' ' +
            'rho=' + str(args[9]) + ' ' +
            'PD0=' + str(args[10]) + ' ' +
            'p=' + str(args[11]) + ' ' +
            'R0=' + str(args[12]) + ' ' +  # Argument names and values

            's=' + str(X[0]) + ' ' +
            'd=' + str(X[1]) + ' '
    )

    # Open the file in append mode and write the BRUTresults string followed by a newline
    with open(file_path, 'a') as file:
        file.write(txt_results + '\n')

    # Close the file
    file.close()
