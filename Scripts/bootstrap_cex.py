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

# Load the CEX data
cex = pd.read_csv(os.path.join(data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2020 + 1)), :]

# Define a function to calculate CEX consumption statistics across bootstrap samples
def bootstrap(n):
    # Define the bootstrap sample and the sampling method
    b = np.mod(n, 1000)
    m = int(np.floor(n / 1000) + 1)

    # Sample from the data
    df_b = pd.DataFrame()
    for year in range(1984, 2020 + 1):
        if m == 1:
            df_b = df_b.append(cex.loc[cex.year == year, :].sample(n=cex.loc[cex.year == year, :].shape[0], replace=True, weights='weight', random_state=b), ignore_index=True)
        else:
            df_b = df_b.append(cex.loc[cex.year == year, :].sample(n=cex.loc[cex.year == year, :].shape[0], replace=True, random_state=b), ignore_index=True)

    # Normalize consumption
    for column in ['consumption', 'consumption_nd']:
        if m == 1:
            df_b.loc[:, column + '_simple'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1), column])
            df_b.loc[:, column + '_simple_latin'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1) & (df_b.latin == 0), column])
            df_b.loc[:, column] = df_b.loc[:, column] / np.average(df_b.loc[df_b.year == 2019, column])
            
        else:
            df_b.loc[:, column + '_simple'] = df_b.loc[:, column] / np.average(df_b.loc[df_b.year == 2019, column], weights=df_b.loc[df_b.year == 2019, 'weight'])
            df_b.loc[:, column + '_simple_latin'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1), column], weights=df_b.loc[(df_b.year == 2019) & (df_b.race == 1), 'weight'])
            df_b.loc[:, column] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1) & (df_b.latin == 0), column], weights=df_b.loc[(df_b.year == 2019) & (df_b.race == 1) & (df_b.latin == 0), 'weight'])

    # Define functions to perform the aggregation
    if m == 1:
        def f(x):
            d = {}
            columns = ['consumption', 'consumption_nd']
            for column in columns:
                d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]))
                d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column])
            return pd.Series(d, index=[key for key, value in d.items()])
        def f_simple(x):
            d = {}
            d['consumption_average'] = np.log(np.average(x.consumption_simple))
            d['consumption_sd'] = np.std(np.log(x.consumption_nd_simple))
            return pd.Series(d, index=[key for key, value in d.items()])
        def f_simple_latin(x):
            d = {}
            d['consumption_average'] = np.log(np.average(x.consumption_simple_latin))
            d['consumption_sd'] = np.std(np.log(x.consumption_nd_simple_latin))
            return pd.Series(d, index=[key for key, value in d.items()])
    else:
        def f(x):
            d = {}
            columns = ['consumption', 'consumption_nd']
            for column in columns:
                d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]), weights=x.weight)
                d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column], weights=x.weight)
            return pd.Series(d, index=[key for key, value in d.items()])
        def f_simple(x):
            d = {}
            d['consumption_average'] = np.log(np.average(x.consumption_simple, weights=x.weight))
            d['consumption_sd'] = np.sqrt(np.average((np.log(x.consumption_nd_simple) - np.average(np.log(x.consumption_nd_simple), weights=x.weight))**2, weights=x.weight))
            return pd.Series(d, index=[key for key, value in d.items()])
        def f_simple_latin(x):
            d = {}
            d['consumption_average'] = np.log(np.average(x.consumption_simple_latin, weights=x.weight))
            d['consumption_sd'] = np.sqrt(np.average((np.log(x.consumption_nd_simple_latin) - np.average(np.log(x.consumption_nd_simple_latin), weights=x.weight))**2, weights=x.weight))
            return pd.Series(d, index=[key for key, value in d.items()])

    # Instantiate an empty data frame
    df_cex = pd.DataFrame()

    # Define a list of CEX column names
    columns = ['Elog_of_c', 'Elog_of_c_nd', 'c_bar', 'c_bar_nd']

    # Calculate CEX consumption statistics by year, race and age in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [m], 'simple': [False]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_cex = df_cex.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [m], 'simple': [True]}), df, how='left')
    df_cex = df_cex.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and age for Latinos in the current bootstrap sample
    df = df_b.loc[(df_b.latin == 1) & (df_b.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [m], 'simple': [False]}), df, how='left')
    df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_cex = df_cex.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year for Latinos in the current bootstrap sample
    df = df_b.loc[(df_b.latin == 1) & (df_b.year >= 2006), :].groupby('year', as_index=False).apply(f_simple_latin)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [m], 'simple': [True]}), df, how='left')
    df_cex = df_cex.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]) & (df_b.latin == 0) & (df_b.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [m], 'simple': [False]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_cex = df_cex.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]) & (df_b.latin == 0) & (df_b.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_simple_latin)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [m], 'simple': [True]}), df, how='left')
    df_cex = df_cex.append(df, ignore_index=True)

    # Save the data frame
    df_cex.to_csv(os.path.join(data, 'dignity_cex_bootstrap_' + str(b) + '_method_' + str(m) + '.csv'), index=False)
    del df_b, df_cex, df

# Calculate CEX consumption statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(bootstrap)(n) for n in range(2000))