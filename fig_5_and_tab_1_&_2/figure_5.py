import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read data from a CSV file
data = pd.read_csv('results/results_Model_2.csv')

# Check if the folder exists
if not os.path.exists('figures'):
    # If it doesn't exist, create the folder
    os.makedirs('figures')

plt.figure()
plot = sns.lmplot(x="rho", y="Theta", logx=True, data=data, scatter_kws={'s': 2.5, 'alpha': 0.75, 'color': 'grey'},
                  aspect=1.5)
plot.set_axis_labels("", "")
plt.xscale('log')
plt.ylim(0, 0.01)
plt.yticks([0, 0.005, 0.01])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.savefig('figures/figure_5_a.png')

plt.figure()
plot = sns.lmplot(x="Theta", y="LSDs", data=data, scatter_kws={'s': 2.5, 'alpha': 0.75, 'color': 'grey'}, aspect=1.5,
                  truncate=False)
plot.set_axis_labels("", "")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig('figures/figure_5_b.png')

plt.figure()
plot = sns.lmplot(x="Theta", y="LSDd", data=data, scatter_kws={'s': 2.5, 'alpha': 0.75, 'color': 'grey'}, aspect=1.5,
                  truncate=False)
plot.set_axis_labels("", "")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig('figures/figure_5_c.png')
