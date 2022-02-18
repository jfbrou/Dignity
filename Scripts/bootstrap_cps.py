# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from joblib import Parallel, delayed
import os

# Find the number of available CPUs
n_cpu = os.cpu_count()

# Import functions
from functions import *

# Set the Sherlock data directory
data = '/scratch/users/jfbrou/Dignity'

# Define a function to calculate CPS leisure statistics across bootstrap samples
def bootstrap(n):
    # Load the CPS data
    cps = pd.read_csv(os.path.join(data, 'cps.csv'))
    cps = cps.loc[cps.year.isin(range(1985, 2021 + 1)), :]
    cps.loc[:, 'year'] = cps.year - 1

    # Define the bootstrap sample and the sampling method
    b = np.mod(n, 1000)
    m = int(np.floor(n / 1000) + 1)

    # Sample from the data
    df_b = pd.DataFrame()
    for year in range(1984, 2020 + 1):
        if m == 1:
            df_b = df_b.append(cps.loc[cps.year == year, :].sample(n=cps.loc[cps.year == year, :].shape[0], replace=True, weights='weight', random_state=b), ignore_index=True)
        else:
            df_b = df_b.append(cps.loc[cps.year == year, :].sample(n=cps.loc[cps.year == year, :].shape[0], replace=True, random_state=b), ignore_index=True)
    del cps
    
    # Define functions to perform the aggregation
    if m == 1:
        def f(x):
            d = {}
            d['Ev_of_ell'] = np.average(v_of_ell(x.leisure))
            d['ell_bar'] = np.average(x.leisure)
            return pd.Series(d, index=[key for key, value in d.items()])
        def f_simple(x):
            d = {}
            d['leisure_average'] = np.average(x.leisure)
            d['leisure_sd'] = np.std(x.leisure)
            return pd.Series(d, index=[key for key, value in d.items()])
    else:
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
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [m], 'simple': [False]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_cps = df_cps.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [m], 'simple': [True]}), df, how='left')
    df_cps = df_cps.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and age for Latinos in the current bootstrap sample
    df = df_b.loc[(df_b.latin == 1) & (df_b.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [m], 'simple': [False]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby('year', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_cps = df_cps.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year for Latinos in the current bootstrap sample
    df = df_b.loc[(df_b.latin == 1) & (df_b.year >= 2006), :].groupby('year', as_index=False).apply(f_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [m], 'simple': [True]}), df, how='left')
    df_cps = df_cps.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]) & (df_b.latin == 0) & (df_b.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [m], 'simple': [False]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_cps = df_cps.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]) & (df_b.latin == 0) & (df_b.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [m], 'simple': [True]}), df, how='left')
    df_cps = df_cps.append(df, ignore_index=True)

    # Save the data frame
    df_cps.to_csv(os.path.join(data, 'dignity_cps_bootstrap_' + str(b) + '_method_' + str(m) + '.csv'), index=False)
    del df_b, df_cps, df

# Calculate CPS leisure statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(bootstrap)(n) for n in range(2000))