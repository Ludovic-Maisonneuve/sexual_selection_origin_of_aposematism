import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from functions.save_and_open import *


def indice_born_trait(array_density):
    # Find the indices of the first and last non-zero elements in the array
    return min(np.nonzero(array_density)[0]), max(np.nonzero(array_density)[0])


def plot_traits_evolution(data_folder, ngmin=0, ngmax=None, density_max=100, plot_v=True, plot_o=True):
    # Get the traits distribution across generations
    array_distribution_across_generations_v, array_distribution_across_generations_o = get_traits_distribution_across_generations(
        data_folder)

    # Get array dimensions and resolution
    resolution = np.shape(array_distribution_across_generations_v)[1]
    nb_of_gen = np.shape(array_distribution_across_generations_v)[0] * 10

    # Define the range of generations to consider
    indice_genmin = ngmin // 10
    indice_genmax = (nb_of_gen // 10) if ngmax is None else (ngmax // 10)

    # Compute the sum of trait distributions over generations
    array_sum_distribution_over_generations_v = np.sum(
        array_distribution_across_generations_v[indice_genmin:indice_genmax + 1, :], axis=0)
    array_sum_distribution_over_generations_o = np.sum(
        array_distribution_across_generations_o[indice_genmin:indice_genmax + 1, :], axis=0)

    # Get limit values for each trait
    indice_vmin, indice_vmax = indice_born_trait(array_sum_distribution_over_generations_v)
    indice_omin, indice_omax = indice_born_trait(array_sum_distribution_over_generations_o)

    # Create arrays for trait values
    list_v = np.linspace(0, 1, resolution)
    list_o = np.linspace(0, 1, resolution)

    # Create colormap
    Blues = matplotlib.cm.get_cmap('Blues')
    L = [np.array([1, 1, 1, 1]) if t == 0 else Blues(t) for t in np.linspace(0, 1, 501)]
    cm = LinearSegmentedColormap.from_list('cmap', L, N=501)

    if plot_v:
        plt.figure()
        ax = plt.gca()
        ax.spines[['top', 'right']].set_visible(False)
        plt.locator_params(axis='y', nbins=4)
        plt.locator_params(axis='x', nbins=3)

        ax.imshow(
            array_distribution_across_generations_v[indice_genmin:indice_genmax + 1, indice_vmin:indice_vmax].T,
            extent=[0 * 10, (indice_genmax - indice_genmin) * 10, list_v[indice_vmin], list_v[indice_vmax]],
            aspect=1 / 1.62 * (indice_genmax * 10 - indice_genmin * 10 + 1) / (
                    list_v[indice_vmax] - list_v[indice_vmin]), cmap=cm, vmin=0, vmax=density_max,
            origin='lower')

    if plot_o:
        plt.figure()
        ax = plt.gca()
        ax.spines[['top', 'right']].set_visible(False)
        plt.locator_params(axis='y', nbins=4)
        plt.locator_params(axis='x', nbins=3)

        ax.imshow(
            array_distribution_across_generations_o[indice_genmin:indice_genmax + 1, indice_omin:indice_omax].T,
            extent=[0 * 10, (indice_genmax - indice_genmin) * 10, list_o[indice_omin], list_o[indice_omax]],
            aspect=1 / 1.62 * (indice_genmax * 10 - indice_genmin * 10 + 1) / (
                    list_o[indice_omax] - list_o[indice_omin]), cmap=cm, vmin=0, vmax=density_max,
            origin='lower')
    plt.show()


def plot_mean_values_evolution(data_folder, list_plot, list_colors=None, ngmin=0, ngmax=None,
                               plot_legend=True):  # Example usage: plot_mean_values_evolution("your_data_folder_path_here")
    # Get the list of generations from the mean values folder
    list_gen = sorted([int(i[:-4]) for i in os.listdir(data_folder + '/mean_values') if i != '.DS_Store'])

    # Iterate over generations and extract mean values
    for gen in list_gen:
        data_file_name = f"{data_folder}/mean_values/{gen}.txt"
        with open(data_file_name, 'r') as open_data:
            # Split data into values and variable names
            list_data = [value for value in open_data.read().split(' ')]
            list_vars_name = [i.split('=')[0] for i in list_data]
            list_vars = [float(i.split('=')[1]) for i in list_data]
        # Initialize lists for the first generation
        if gen == 0:
            attribute_data_list = {f'list_{var_name}_mean': [] for var_name in list_vars_name}

        # Append mean values to the corresponding lists
        for var, var_name in zip(list_vars, list_vars_name):
            attribute_data_list[f'list_{var_name}_mean'].append(var)

    # Define the range of generations to plot
    imin = ngmin // 10
    imax = (list_gen[-1] // 10) if ngmax is None else (ngmax // 10)

    # Plot mean values with or without colors

    # Check if custom colors are provided
    if list_colors is None:
        # Iterate over lists of variables to plot
        for list_vars_plot in list_plot:
            # Create a new figure and axis
            fig, axs = plt.subplots(1, 1)

            # Iterate over variables in the current list
            for var in list_vars_plot:
                # Plot mean values for the variable
                axs.plot(list_gen[0:imax - imin], attribute_data_list[f'list_{var}_mean'][imin:imax], label=var)

            # Add legend if specified
            if plot_legend:
                axs.legend()
    else:
        # Iterate over lists of variables and corresponding colors
        for list_vars_plot, list_colors_single_plot in zip(list_plot, list_colors):
            # Create a new figure and axis
            fig, axs = plt.subplots(1, 1)

            # Iterate over variables and colors in the current lists
            for var, c in zip(list_vars_plot, list_colors_single_plot):
                # Plot mean values for the variable with the specified color
                # PB here
                axs.plot(list_gen[0:imax - imin], attribute_data_list[f'list_{var}_mean'][imin:imax], label=var,
                         color=c)

            # Add legend if specified
            if plot_legend:
                axs.legend()

    # Display the plot
    # plt.show()


def get_list_mean(data_folder, list_var):  # Example usage: plot_mean_values_evolution("your_data_folder_path_here")
    # Get the list of generations from the mean values folder
    list_gen = sorted([int(i[:-4]) for i in os.listdir(data_folder + '/mean_values') if i != '.DS_Store'])

    # Iterate over generations and extract mean values
    for gen in list_gen:
        data_file_name = f"{data_folder}/mean_values/{gen}.txt"
        with open(data_file_name, 'r') as open_data:
            # Split data into values and variable names
            list_data = [value for value in open_data.read().split(' ')]
            list_vars_name = [i.split('=')[0] for i in list_data]
            list_vars = [float(i.split('=')[1]) for i in list_data]
        # Initialize lists for the first generation
        if gen == 0:
            attribute_data_list = {f'list_{var_name}_mean': [] for var_name in list_vars_name}

        # Append mean values to the corresponding lists
        for var, var_name in zip(list_vars, list_vars_name):
            attribute_data_list[f'list_{var_name}_mean'].append(var)

    results = []
    for var_name in list_var:
        results.append(attribute_data_list[f'list_{var_name}_mean'])
    return tuple(results)


# NOT SURE BELOW

def plot_distribution_two_vars(var1, var2, data_folder, gen, cov=None, s=5, alpha=0.5):
    # Initialize lists to store variable values
    list_var1, list_var2 = [], []

    # Read data from the file for the specified generation
    data_file_name = data_folder + '/G/' + str(gen) + '.txt'
    open_data = open(data_file_name, 'r')
    list_data = open_data.read().split('\n')

    # Extract variable values from each line of the data file
    for line_txt in list_data:
        if line_txt != '':
            list_attributes_txt = line_txt.split(' ')
            list_attributes = [float(i) for i in list_attributes_txt]

            # Assign values to variables
            l, v, tau, k, tauv, K, f, N = list_attributes
            list_var1.append(locals()[var1])
            list_var2.append(locals()[var2])

    # Create a scatter plot
    fig, axs = plt.subplots()
    axs.locator_params(axis='y', nbins=3)
    axs.locator_params(axis='x', nbins=3)
    axs.scatter(list_var1, list_var2, s=s, alpha=alpha)

    # Adjust the aspect ratio of the plot
    x0, x1 = axs.get_xlim()
    y0, y1 = axs.get_ylim()
    axs.set_aspect(abs(x1 - x0) / abs(y1 - y0))


def save_data_for_mathematica(data_folder, ngmin=0,
                              ngmax=None):  # Example usage: plot_mean_values_evolution("your_data_folder_path_here")

    if not os.path.exists(data_folder + '/Mathematica_'):
        os.mkdir(data_folder + '/Mathematica_')
    else:
        delete_folder(data_folder + '/Mathematica_')
        os.mkdir(data_folder + '/Mathematica_')

    # Get the list of generations from the mean values folder
    list_gen_ = sorted([int(i[:-4]) for i in os.listdir(data_folder + '/mean_values') if i != '.DS_Store'])
    list_gen = []
    for gen in list_gen_:
        if gen % 100 == 0:
            list_gen.append(gen)

    # Iterate over generations and extract mean values
    for gen in list_gen:
        data_file_name = f"{data_folder}/mean_values/{gen}.txt"
        with open(data_file_name, 'r') as open_data:
            # Split data into values and variable names
            list_data = [value for value in open_data.read().split(' ')]
            list_vars_name = [i.split('=')[0] for i in list_data]
            list_vars = [float(i.split('=')[1]) for i in list_data]
        # Initialize lists for the first generation
        if gen == 0:
            for var_name in list_vars_name:
                locals()[f'list_{var_name}_mean'] = []

        # Append mean values to the corresponding lists
        for var, var_name in zip(list_vars, list_vars_name):
            locals()[f'list_{var_name}_mean'].append(var)

    # Define the range of generations to plot
    imin = ngmin // 100
    imax = (list_gen[-1] // 100) if ngmax is None else (ngmax // 100)

    for var, var_name in zip(list_vars, list_vars_name):
        with open(data_folder + '/Mathematica_/' + var_name + ".json", 'w') as file:
            json.dump(locals()[f'list_{var_name}_mean'][imin:imax], file)

    with open(data_folder + '/Mathematica_/gen.json', 'w') as file:
        json.dump(list_gen[0:imax - imin], file)
