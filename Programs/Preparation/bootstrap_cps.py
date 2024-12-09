# Import libraries
import os
import sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Import functions and directories
sys.path.append(os.path.join(os.getcwd(), 'Preparation'))
from functions import *
from directories import *

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'), usecols=['year', 'weight', 'leisure', 'race', 'age'])
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Define a function to calculate CPS leisure statistics across bootstrap samples
def bootstrap(b, cps):
    # Sample from the data
    df_b = pd.DataFrame()
    for year in range(1984, 2022 + 1):
        df_b = pd.concat([df_b, cps.loc[cps.year == year, :].sample(frac=1, replace=True, random_state=b)], ignore_index=True)
    del cps
    
    # Define functions to perform the aggregation
    def f(x):
        d = {}
        d['Ev_of_ell'] = np.average(v_of_ell(x.leisure), weights=x.weight)
        d['ell_bar'] = np.average(x.leisure, weights=x.weight)
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_simple(x):
        d = {}
        d['leisure_average'] = np.average(x.leisure, weights=x.weight)
        d['leisure_sd'] = np.sqrt(np.average((x.leisure - np.average(x.leisure, weights=x.weight))**2, weights=x.weight))
        return pd.Series(d, index=[key for key, value in d.items()])

    # Instantiate an empty data frame
    df_cps = pd.DataFrame()

    # Calculate CPS leisure statistics by year, race and age in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'bootstrap': [b], 'simple': [False]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_cps = pd.concat([df_cps, df], ignore_index=True)

    # Calculate CPS leisure statistics by year and race in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'bootstrap': [b], 'simple': [True]}), df, how='left')
    df_cps = pd.concat([df_cps, df], ignore_index=True)

    # Save the data frame
    df_cps.to_csv(os.path.join(cps_f_data, 'dignity_cps_bootstrap_' + str(b) + '.csv'), index=False)
    del df_b, df_cps, df

# Calculate CPS leisure statistics across 1000 bootstrap samples
for b in range(1, 1000 + 1):
    bootstrap(b, cps)

# Append all bootstrap samples in a single data frame
dignity_cps_bootstrap = pd.DataFrame()
for b in range(1, 1000 + 1, 1):
    df_cps = pd.read_csv(os.path.join(cps_f_data, 'dignity_cps_bootstrap_' + str(b) + '.csv'))
    dignity_cps_bootstrap = pd.concat([dignity_cps_bootstrap, df_cps], ignore_index=True)
    os.remove(os.path.join(cps_f_data, 'dignity_cps_bootstrap_' + str(b) + '.csv'))
    del df_cps
dignity_cps_bootstrap.to_csv(os.path.join(cps_f_data, 'dignity_cps_bootstrap.csv'), index=False)