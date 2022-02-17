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
    # Sample from the CEX and CPS data
    df_cex_1 = pd.DataFrame()
    df_cex_2 = pd.DataFrame()
    df_cps_1 = pd.DataFrame()
    df_cps_2 = pd.DataFrame()
    for year in range(1984, 2020 + 1):
        df_cex_1 = df_cex_1.append(cex.loc[cex.year == year, :].sample(n=cex.loc[cex.year == year, :].shape[0], replace=True, weights='weight', random_state=b), ignore_index=True)
        df_cex_2 = df_cex_2.append(cex.loc[cex.year == year, :].sample(n=cex.loc[cex.year == year, :].shape[0], replace=True, random_state=b), ignore_index=True)
        df_cps_1 = df_cps_1.append(cps.loc[cps.year == year, :].sample(n=cps.loc[cps.year == year, :].shape[0], replace=True, weights='weight', random_state=b), ignore_index=True)
        df_cps_2 = df_cps_2.append(cps.loc[cps.year == year, :].sample(n=cps.loc[cps.year == year, :].shape[0], replace=True, random_state=b), ignore_index=True)

    # Normalize consumption in the CEX data
    df_cex_simple_1 = df_cex_1.copy()
    df_cex_simple_2 = df_cex_2.copy()
    df_cex_simple_latin_1 = df_cex_1.copy()
    df_cex_simple_latin_2 = df_cex_2.copy()
    for column in ['consumption', 'consumption_nd']:
        df_cex_1.loc[:, column] = df_cex_1.loc[:, column] / np.average(df_cex_1.loc[df_cex_1.year == 2019, column])
        df_cex_2.loc[:, column] = df_cex_2.loc[:, column] / np.average(df_cex_2.loc[df_cex_2.year == 2019, column], weights=df_cex_2.loc[df_cex_2.year == 2019, 'weight'])
        df_cex_simple_1.loc[:, column] = df_cex_simple_1.loc[:, column] / np.average(df_cex_simple_1.loc[(df_cex_simple_1.year == 2019) & (df_cex_simple_1.race == 1), column])
        df_cex_simple_2.loc[:, column] = df_cex_simple_2.loc[:, column] / np.average(df_cex_simple_2.loc[(df_cex_simple_2.year == 2019) & (df_cex_simple_2.race == 1), column], weights=df_cex_simple_2.loc[(df_cex_simple_2.year == 2019) & (df_cex_simple_2.race == 1), 'weight'])
        df_cex_simple_latin_1.loc[:, column] = df_cex_simple_latin_1.loc[:, column] / np.average(df_cex_simple_latin_1.loc[(df_cex_simple_latin_1.year == 2019) & (df_cex_simple_latin_1.race == 1) & (df_cex_simple_latin_1.latin == 0), column])
        df_cex_simple_latin_2.loc[:, column] = df_cex_simple_latin_2.loc[:, column] / np.average(df_cex_simple_latin_2.loc[(df_cex_simple_latin_2.year == 2019) & (df_cex_simple_latin_2.race == 1) & (df_cex_simple_latin_2.latin == 0), column], weights=df_cex_simple_latin_2.loc[(df_cex_simple_latin_2.year == 2019) & (df_cex_simple_latin_2.race == 1) & (df_cex_simple_latin_2.latin == 0), 'weight'])

    # Define functions to perform the CEX aggregation
    def f_cex_1(x):
        d = {}
        columns = ['consumption', 'consumption_nd']
        for column in columns:
            d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]))
            d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column])
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cex_2(x):
        d = {}
        columns = ['consumption', 'consumption_nd']
        for column in columns:
            d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]), weights=x.weight)
            d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column], weights=x.weight)
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cex_simple_1(x):
        d = {}
        d['consumption_average'] = np.log(np.average(x.consumption))
        d['consumption_sd'] = np.std(np.log(x.consumption_nd))
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cex_simple_2(x):
        d = {}
        d['consumption_average'] = np.log(np.average(x.consumption), weights=x.weight)
        d['consumption_sd'] = np.sqrt(np.average((np.log(x.consumption_nd) - np.average(np.log(x.consumption_nd), weights=x.weight))**2, weights=x.weight))
        return pd.Series(d, index=[key for key, value in d.items()])

    # Instantiate empty data frames
    df_consumption = pd.DataFrame()
    df_consumption_simple = pd.DataFrame()

    # Define a list of CEX column names
    columns = ['Elog_of_c', 'Elog_of_c_nd', 'c_bar', 'c_bar_nd']

    # Calculate CEX consumption statistics by year, race and age in the current bootstrap sample
    df = df_cex_1.loc[df_cex_1.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex_1)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year, race and age in the current bootstrap sample
    df = df_cex_2.loc[df_cex_2.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex_2)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race in the current bootstrap sample
    df = df_cex_simple_1.loc[df_cex_simple_1.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_cex_simple_1)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race in the current bootstrap sample
    df = df_cex_simple_2.loc[df_cex_simple_2.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_cex_simple_2)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and age for Latinos in the current bootstrap sample
    df = df_cex_1.loc[(df_cex_1.latin == 1) & (df_cex_1.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cex_1)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and age for Latinos in the current bootstrap sample
    df = df_cex_2.loc[(df_cex_2.latin == 1) & (df_cex_2.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cex_2)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year for Latinos in the current bootstrap sample
    df = df_cex_simple_latin_1.loc[(df_cex_simple_latin_1.latin == 1) & (df_cex_simple_latin_1.year >= 2006), :].groupby('year', as_index=False).apply(f_cex_simple_1)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year for Latinos in the current bootstrap sample
    df = df_cex_simple_latin_2.loc[(df_cex_simple_latin_2.latin == 1) & (df_cex_simple_latin_2.year >= 2006), :].groupby('year', as_index=False).apply(f_cex_simple_2)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_cex_1.loc[df_cex_1.race.isin([1, 2]) & (df_cex_1.latin == 0) & (df_cex_1.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex_1)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_cex_2.loc[df_cex_2.race.isin([1, 2]) & (df_cex_2.latin == 0) & (df_cex_2.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex_2)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
    df_consumption = df_consumption.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_cex_simple_latin_1.loc[df_cex_simple_latin_1.race.isin([1, 2]) & (df_cex_simple_latin_1.latin == 0) & (df_cex_simple_latin_1.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_cex_simple_1)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Calculate CEX consumption statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_cex_simple_latin_2.loc[df_cex_simple_latin_2.race.isin([1, 2]) & (df_cex_simple_latin_2.latin == 0) & (df_cex_simple_latin_2.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_cex_simple_2)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df_consumption_simple = df_consumption_simple.append(df, ignore_index=True)

    # Clear the CEX bootstrap sample from memory
    del df_cex_1, df_cex_2, df_cex_simple_1, df_cex_simple_2, df_cex_simple_latin_1, df_cex_simple_latin_2

    # Define functions to perform the CEX aggregation
    def f_cps_1(x):
        d = {}
        d['Ev_of_ell'] = np.average(v_of_ell(x.leisure))
        d['ell_bar'] = np.average(x.leisure)
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cps_2(x):
        d = {}
        d['Ev_of_ell'] = np.average(v_of_ell(x.leisure), weights=x.weight)
        d['ell_bar'] = np.average(x.leisure, weights=x.weight)
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cps_simple_1(x):
        d = {}
        d['leisure_average'] = np.average(x.leisure)
        d['leisure_sd'] = np.std(x.leisure)
        return pd.Series(d, index=[key for key, value in d.items()])
    def f_cps_simple_2(x):
        d = {}
        d['leisure_average'] = np.average(x.leisure, weights=x.weight)
        d['leisure_sd'] = np.sqrt(np.average((x.leisure - np.average(x.leisure, weights=x.weight))**2, weights=x.weight))
        return pd.Series(d, index=[key for key, value in d.items()])

    # Instantiate empty data frames
    df_leisure = pd.DataFrame()
    df_leisure_simple = pd.DataFrame()

    # Calculate CPS leisure statistics by year, race and age in the current bootstrap sample
    df = df_cps_1.loc[df_cps_1.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps_1)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year, race and age in the current bootstrap sample
    df = df_cps_2.loc[df_cps_2.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps_2)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race in the current bootstrap sample
    df = df_cps_1.loc[df_cps_1.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_cps_simple_1)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race in the current bootstrap sample
    df = df_cps_2.loc[df_cps_1.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(f_cps_simple_2)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [-1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and age for Latinos in the current bootstrap sample
    df = df_cps_1.loc[(df_cps_1.latin == 1) & (df_cps_1.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cps_1)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby('year', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and age for Latinos in the current bootstrap sample
    df = df_cps_2.loc[(df_cps_2.latin == 1) & (df_cps_2.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cps_2)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby('year', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year for Latinos in the current bootstrap sample
    df = df_cps_1.loc[(df_cps_1.latin == 1) & (df_cps_1.year >= 2006), :].groupby('year', as_index=False).apply(f_cps_simple_1)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year for Latinos in the current bootstrap sample
    df = df_cps_2.loc[(df_cps_2.latin == 1) & (df_cps_2.year >= 2006), :].groupby('year', as_index=False).apply(f_cps_simple_2)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [-1], 'latin': [1], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_cps_1.loc[df_cps_1.race.isin([1, 2]) & (df_cps_1.latin == 0) & (df_cps_1.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps_1)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year, race and age for non-Latinos in the current bootstrap sample
    df = df_cps_2.loc[df_cps_2.race.isin([1, 2]) & (df_cps_2.latin == 0) & (df_cps_2.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps_2)
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_leisure = df_leisure.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_cps_1.loc[df_cps_1.race.isin([1, 2]) & (df_cps_1.latin == 0) & (df_cps_1.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_cps_simple_1)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [1]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Calculate CPS leisure statistics by year and race for non-Latinos in the current bootstrap sample
    df = df_cps_2.loc[df_cps_2.race.isin([1, 2]) & (df_cps_2.latin == 0) & (df_cps_2.year >= 2006), :].groupby(['year', 'race'], as_index=False).apply(f_cps_simple_2)
    df = pd.merge(expand({'year': df.year.unique(), 'race': [1, 2], 'latin': [0], 'bootstrap': [b], 'method': [2]}), df, how='left')
    df_leisure_simple = df_leisure_simple.append(df, ignore_index=True)

    # Clear the CEX bootstrap sample from memory
    del df_cps_1, df_cps_2

    # Merge and save the data frames
    df = pd.merge(df_consumption, df_leisure, how='left')
    df.to_csv(os.path.join(data, 'dignity_bootstrap_ ' + str(b) + '.csv'), index=False)
    del df
    df_simple = pd.merge(df_consumption_simple, df_leisure_simple, how='left')
    df_simple.to_csv(os.path.join(data, 'dignity_bootstrap_simple_ ' + str(b) + '.csv'), index=False)
    del df_simple

# Calculate CEX consumption and CPS leisure statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(bootstrap_statistics)(b) for b in range(1000))

# Append all bootstrap samples in a single data frame
dignity_bootstrap = pd.DataFrame()
dignity_bootstrap_simple = pd.DataFrame()
for b in range(1000):
    df = pd.read_csv(os.path.join(data, 'dignity_bootstrap_ ' + str(b) + '.csv'))
    df_simple = pd.read_csv(os.path.join(data, 'dignity_bootstrap_simple_ ' + str(b) + '.csv'))
    dignity_bootstrap = dignity_bootstrap.append(df, ignore_index=True)
    dignity_bootstrap_simple = dignity_bootstrap_simple.append(df_simple, ignore_index=True)
    del df, df_simple
dignity_bootstrap.to_csv(os.path.join(data, 'dignity_bootstrap.csv'), index=False)
dignity_bootstrap_simple.to_csv(os.path.join(data, 'dignity_bootstrap_simple.csv'), index=False)

# Perform the consumption-equivalent welfare calculations on each bootstrap sample
dignity = pd.read_csv(os.path.join(data, 'dignity.csv'))
dignity_u_bar = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity_bootstrap = pd.read_csv(os.path.join(data, 'dignity_bootstrap.csv'))
c_nominal = 31046.442985362326
def cew(b):
    # Use the data for the consumption-equivalent welfare of Black relative to White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 1) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the consumption-equivalent welfare of Black relative to White Americans
    df_1 = expand({'year': years, 'race': [2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [1], 'description': ['black']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Ev_of_ell'].values
        df_1.loc[df_1.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                              S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                              inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Black relative to White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 2) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the consumption-equivalent welfare of Black relative to White Americans
    df_2 = expand({'year': years, 'race': [2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [2], 'description': ['black']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Ev_of_ell'].values
        df_2.loc[df_2.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                              S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                              inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 1) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == 0), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == 0), :]

    # Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans
    df_black_1 = expand({'year': years, 'race': [2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [1], 'description': ['black non-latino']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Ev_of_ell'].values
        df_black_1.loc[df_black_1.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 2) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == 0), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == 0), :]

    # Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans
    df_black_2 = expand({'year': years, 'race': [2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [2], 'description': ['black non-latino']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Ev_of_ell'].values
        df_black_2.loc[df_black_2.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False), :]
    
    # Calculate the consumption-equivalent welfare of Latino relative to White non-Latino Americans
    df_latin_1 = expand({'year': years, 'race': [-1], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [1], 'description': ['latino']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1) & (df_survival.latin == 0), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == -1) & (df_survival.latin == 1), 'S'].values
        c_bar_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar'].values
        c_bar_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar'].values
        ell_bar_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'ell_bar'].values
        ell_bar_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Ev_of_ell'].values
        df_latin_1.loc[df_latin_1.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 2), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False), :]
    
    # Calculate the consumption-equivalent welfare of Latino relative to White non-Latino Americans
    df_latin_2 = expand({'year': years, 'race': [-1], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [2], 'description': ['latino']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1) & (df_survival.latin == 0), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == -1) & (df_survival.latin == 1), 'S'].values
        c_bar_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar'].values
        c_bar_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar'].values
        ell_bar_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'ell_bar'].values
        ell_bar_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Ev_of_ell'].values
        df_latin_2.loc[df_latin_2.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the cumulative welfare growth of Black and White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 1) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the cumulative welfare growth of Black and White Americans
    df_growth_1 = expand({'year': years[1:], 'race': [1, 2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [1], 'description': ['growth']})
    for race in [1, 2]:
        for year in years[1:]:
            S_i = df_survival.loc[(df_survival.year == years[years.index(year) - 1]) & (df_survival.race == race), 'S'].values
            S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == race), 'S'].values
            c_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar'].values
            c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar'].values
            ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'ell_bar'].values
            ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'ell_bar'].values
            T = year - years[years.index(year) - 1]
            S_intercept = dignity_u_bar.loc[:, 'S'].values
            c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
            ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
            c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar_nd'].values
            c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar_nd'].values
            Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            df_growth_1.loc[(df_growth_1.year == year) & (df_growth_1.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                                                S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                                                inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']
    
    # Use the data for the cumulative welfare growth of Black and White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == 2) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the cumulative welfare growth of Black and White Americans
    df_growth_2 = expand({'year': years[1:], 'race': [1, 2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [2], 'description': ['growth']})
    for race in [1, 2]:
        for year in years[1:]:
            S_i = df_survival.loc[(df_survival.year == years[years.index(year) - 1]) & (df_survival.race == race), 'S'].values
            S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == race), 'S'].values
            c_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar'].values
            c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar'].values
            ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'ell_bar'].values
            ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'ell_bar'].values
            T = year - years[years.index(year) - 1]
            S_intercept = dignity_u_bar.loc[:, 'S'].values
            c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
            ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
            c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar_nd'].values
            c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar_nd'].values
            Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            df_growth_2.loc[(df_growth_2.year == year) & (df_growth_2.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                                                S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                                                inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']
    
    # Cumulate the growth rates
    df_growth_1.loc[:, 'log_lambda'] = df_growth_1.groupby('race', as_index=False).log_lambda.transform(lambda x: np.exp(np.cumsum(x))).log_lambda.values
    df_growth_2.loc[:, 'log_lambda'] = df_growth_2.groupby('race', as_index=False).log_lambda.transform(lambda x: np.exp(np.cumsum(x))).log_lambda.values

    # Append and save the data frames
    df = df_1.append(df_2, ignore_index=True)
    df = df.append(df_black_1, ignore_index=True)
    df = df.append(df_black_2, ignore_index=True)
    df = df.append(df_latin_1, ignore_index=True)
    df = df.append(df_latin_2, ignore_index=True)
    df = df.append(df_growth_1, ignore_index=True)
    df = df.append(df_growth_2, ignore_index=True)
    df.to_csv(os.path.join(data, 'cew_bootstrap_ ' + str(b) + '.csv'), index=False)

# Calculate the consumption-equivalent welfare statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(bootstrap_statistics)(b) for b in range(1000))

# Append all bootstrap samples in a single data frame
cew_bootstrap = pd.DataFrame()
for b in range(1000):
    df = pd.read_csv(os.path.join(data, 'cew_bootstrap_ ' + str(b) + '.csv'))
    cew_bootstrap = cew_bootstrap.append(df, ignore_index=True)
    del df
cew_bootstrap.to_csv(os.path.join(data, 'cew_bootstrap.csv'), index=False)