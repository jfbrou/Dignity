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

################################################################################
#                                                                              #
# This section of the script produces the bootstrap consumption and leisure    #
# statistics between 1984 and 2020.                                            #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2020 + 1)), :]

# Load the CPS data
cps = pd.read_csv(os.path.join(data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2021 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Define a function to calculate CEX consumption and CPS leisure statistics across bootstrap samples
def bootstrap_statistics(b):
    # Instantiate empty data frames
    df_consumption = pd.DataFrame()
    df_consumption_simple = pd.DataFrame()
    df_cex = pd.DataFrame()
    df_leisure = pd.DataFrame()
    df_leisure_simple = pd.DataFrame()
    df_cps = pd.DataFrame()

    # Sample from the CEX and CPS data
    for year in range(1984, 2020 + 1):
        df_cex = df_cex.append(cex.loc[cex.year == year, :].sample(n=cex.loc[cex.year == year, :].shape[0], replace=True, weights='weight', random_state=b), ignore_index=True)
        df_cps = df_cps.append(cps.loc[cps.year == year, :].sample(n=cps.loc[cps.year == year, :].shape[0], replace=True, weights='weight', random_state=b), ignore_index=True)

    # Normalize consumption in the CEX data
    df_cex_simple = df_cex
    for column in ['consumption', 'consumption_nd']:
        df_cex_simple.loc[:, column] = df_cex_simple.loc[:, column] / np.average(df_cex_simple.loc[(df_cex_simple.year == 2019) & (df_cex_simple.race == 1), column])
        df_cex.loc[:, column] = df_cex.loc[:, column] / np.average(df_cex.loc[df_cex.year == 2019, column])

    # Define functions to perform the CEX aggregation
    def f_cex(x):
        d = {}
        columns = ['consumption', 'consumption_nd']
        for column in columns:
            d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]))
            d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column])
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cex_simple(x):
        d = {}
        d['consumption_average'] = np.log(np.average(x.consumption))
        d['consumption_sd'] = np.std(np.log(x.consumption_nd))
        return pd.Series(d, index=[key for key, value in d.items()])

    # Define a list of column names
    columns = ['Elog_of_c', 'Elog_of_c_nd', 'c_bar', 'c_bar_nd']

    # Calculate CEX consumption statistics by year and age in the current bootstrap sample
    df = df_cex.groupby(['year', 'age'], as_index=False).apply(f_cex)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'bootstrap': [b]}), df, how='left')
    df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year, race and age in the current bootstrap sample
    df = df_cex.loc[df_cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race in the current bootstrap sample
    df = df_cex_simple.loc[df_cex_simple.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_cex_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and age for Latinos in the current bootstrap sample
    df = df_cex.loc[(df_cex.latin == 1) & (df_cex.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cex)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b]}), df, how='left')
    df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year for Latinos in the current bootstrap sample
    df = df_cex_simple.loc[(df_cex_simple.latin == 1) & (df_cex_simple.year >= 2006), :].groupby('year', as_index=False).apply(f_cex_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [-1], 'latin': [1], 'bootstrap': [b]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_cex.loc[df_cex.race.isin([1, 2]) & (df_cex.latin == 0) & (df_cex.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_cex_simple.loc[df_cex_simple.race.isin([1, 2]) & (df_cex_simple.latin == 0) & (df_cex_simple.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_cex_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [0], 'bootstrap': [b]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Define functions to perform the CEX aggregation
    def f_cps(x):
        d = {}
        d['Ev_of_ℓ'] = np.average(v_of_ℓ(x.leisure))
        d['ℓ_bar'] = np.average(x.leisure)
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cps_simple(x):
        d = {}
        d['leisure_average'] = np.average(x.leisure)
        d['leisure_sd'] = np.std(x.leisure)
        return pd.Series(d, index=[key for key, value in d.items()])

    # Calculate CPS leisure statistics by year and age in the current bootstrap sample
    df = df_cps.groupby(['year', 'age'], as_index=False).apply(f_cps)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'bootstrap': [b]}), df, how='left')
    df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby('year', as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
    df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year, race and age in the current bootstrap sample
    df = df_cps.loc[df_cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b]}), df, how='left')
    df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
    df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race in the current bootstrap sample
    df = df_cps.loc[df_cps.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_cps_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and age for Latinos in the current bootstrap sample
    df = df_cps.loc[(df_cps.latin == 1) & (df_cps.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cps)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b]}), df, how='left')
    df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby('year', as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
    df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year for Latinos in the current bootstrap sample
    df = df_cps.loc[(df_cps.latin == 1) & (df_cps.year >= 2006), :].groupby('year', as_index=False).apply(f_cps_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [-1], 'latin': [1], 'bootstrap': [b]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_cps.loc[df_cps.race.isin([1, 2]) & (df_cps.latin == 0) & (df_cps.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b]}), df, how='left')
    df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
    df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_cps.loc[df_cps.race.isin([1, 2]) & (df_cps.latin == 0) & (df_cps.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_cps_simple)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [0], 'bootstrap': [b]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Merge and return the data frames
    df = pd.merge(df_consumption, df_leisure, how='left')
    df_simple = pd.merge(df_consumption_simple, df_leisure_simple, how='left')
    return df, df_simple

# Calculate CEX consumption and CPS leisure statistics across bootstrap samples
results = Parallel(n_jobs=n_cpu)(delayed(bootstrap_statistics)(b) for b in range(1000))
df_bootstrap = pd.DataFrame()
df_bootstrap_simple = pd.DataFrame()
for result in results:
    df_bootstrap = df_bootstrap.append(result[0], ignore_index=True)
    df_bootstrap_simple = df_bootstrap_simple.append(result[1], ignore_index=True)
del cex, cps

# Save the data frame
df_bootstrap.to_csv(os.path.join(data, 'dignity_bootstrap.csv'), index=False)
df_bootstrap_simple.to_csv(os.path.join(data, 'dignity_bootstrap_simple.csv'), index=False)
