# Import libraries
import os
import sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Set the job index
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

# Import functions
sys.path.append(os.path.dirname(os.getcwd()))
from functions import *

# Set the Sherlock data directory
data = '/scratch/jfbrou/Dignity'

# Define a function to calculate CEX consumption statistics across bootstrap samples
def bootstrap(b):
    # Load the CEX data
    cex = pd.read_csv(os.path.join(data, 'cex.csv'))
    cex = cex.loc[cex.year.isin(range(1984, 2022 + 1)), :]

    # Sample from the data
    df_b = pd.DataFrame()
    for year in range(1984, 2022 + 1):
        df_b = pd.concat([df_b, cex.loc[cex.year == year, :].sample(frac=1, replace=True, random_state=b)], ignore_index=True)
    del cex
    
    # Normalize consumption
    for column in ['consumption', 'consumption_nd']:
        df_b.loc[:, column + '_simple'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2022) & (df_b.race == 1), column], weights=df_b.loc[(df_b.year == 2022) & (df_b.race == 1), 'weight'])
        df_b.loc[:, column] = df_b.loc[:, column] / np.average(df_b.loc[df_b.year == 2022, column], weights=df_b.loc[df_b.year == 2022, 'weight'])

    # Define functions to perform the aggregation
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
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'bootstrap': [b], 'simple': [False]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_cex = pd.concat([df_cex, df], ignore_index=True)

    # Calculate CEX consumption statistics by year and race in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'bootstrap': [b], 'simple': [True]}), df, how='left')
    df_cex = pd.concat([df_cex, df], ignore_index=True)

    # Save the data frame
    df_cex.to_csv(os.path.join(data, 'dignity_cex_bootstrap_' + str(b) + '.csv'), index=False)
    del df_b, df_cex, df

# Calculate CEX consumption statistics across 1000 bootstrap samples
samples = range((idx - 1) * 5 + 1, np.minimum(idx * 5, 1000) + 1, 1)
for sample in samples:
    bootstrap(sample)