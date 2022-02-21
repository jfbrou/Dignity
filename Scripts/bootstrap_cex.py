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

# Define a function to calculate CEX consumption statistics across bootstrap samples
def bootstrap(n):
    # Load the CEX data
    cex = pd.read_csv(os.path.join(data, 'cex.csv'))
    cex = cex.loc[cex.year.isin(range(1984, 2020 + 1)), :]

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
    del cex
    
    # Normalize consumption
    for column in ['consumption', 'consumption_nd']:
        if m == 1:
            df_b.loc[:, column + '_simple'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1), column])
            df_b.loc[:, column + '_simple_latin'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1) & (df_b.latin == 0), column])
            df_b.loc[:, column] = df_b.loc[:, column] / np.average(df_b.loc[df_b.year == 2019, column])
            
        else:
            df_b.loc[:, column + '_simple'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1), column], weights=df_b.loc[(df_b.year == 2019) & (df_b.race == 1), 'weight'])
            df_b.loc[:, column + '_simple_latin'] = df_b.loc[:, column] / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1) & (df_b.latin == 0), column], weights=df_b.loc[(df_b.year == 2019) & (df_b.race == 1) & (df_b.latin == 0), 'weight'])
            df_b.loc[:, column] = df_b.loc[:, column] / np.average(df_b.loc[df_b.year == 2019, column], weights=df_b.loc[df_b.year == 2019, 'weight'])

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

    # Keep the imputation sample
    df_b = df_b.loc[df_b.complete == 1, :]

    # Create race binary variables
    df_b = pd.concat([df_b, pd.get_dummies(df_b.race.astype('int'), prefix='race')], axis=1)

    # Create education binary variables
    df_b.loc[df_b.education.isna() | (df_b.age < 30), 'education'] = 4
    df_b = pd.concat([df_b, pd.get_dummies(df_b.education.astype('int'), prefix='education')], axis=1)

    # Recode the gender variable
    df_b.loc[:, 'gender'] = df_b.gender.replace({1: 1, 2: 0})

    # Define a function to calculate average consumption, income and demographics by year
    if m == 1:
        def f(x):
            d = {}
            columns = ['consumption_cex', 'earnings', 'salary'] + ['race_' + str(i) for i in range(1, 4 + 1)] \
                                                                + ['education_' + str(i) for i in range(1, 4 + 1)] \
                                                                + ['family_size', 'latin', 'gender', 'age']
            for column in columns:
                d[column + '_average'] = np.average(x.loc[:, column])
            return pd.Series(d, index=[key for key, value in d.items()])
    else:
        def f(x):
            d = {}
            columns = ['consumption_cex', 'earnings', 'salary'] + ['race_' + str(i) for i in range(1, 4 + 1)] \
                                                                + ['education_' + str(i) for i in range(1, 4 + 1)] \
                                                                + ['family_size', 'latin', 'gender', 'age']
            for column in columns:
                d[column + '_average'] = np.average(x.loc[:, column], weights=x.weight)
            return pd.Series(d, index=[key for key, value in d.items()])

    # Calculate average consumption, income and demographics by year
    df_b = pd.merge(df_b, df_b.groupby('year', as_index=False).apply(f), how='left')

    # Calculate the percentage deviation of consumption, income and demographics from their annual average
    columns = ['consumption_cex', 'earnings', 'salary'] + ['race_' + str(i) for i in range(1, 4 + 1)] \
                                                        + ['education_' + str(i) for i in range(1, 4 + 1)] \
                                                        + ['family_size', 'latin', 'gender', 'age']
    for column in columns:
        df_b.loc[:, column + '_deviation'] = (df_b.loc[:, column] - df_b.loc[:, column + '_average']) / df_b.loc[:, column + '_average']

    # Fit and save the OLS models for consumption
    if m == 1:
        earnings_model = sm.WLS(df_b.consumption_cex_deviation.to_numpy(), df_b.loc[:, [column for column in df_b.columns if column.endswith('deviation') and not column.startswith('consumption') and not column.startswith('salary')]].to_numpy()).fit()
        salary_model = sm.WLS(df_b.consumption_cex_deviation.to_numpy(), df_b.loc[:, [column for column in df_b.columns if column.endswith('deviation') and not column.startswith('consumption') and not column.startswith('earnings')]].to_numpy()).fit()
    else:
        earnings_model = sm.WLS(df_b.consumption_cex_deviation.to_numpy(), df_b.loc[:, [column for column in df_b.columns if column.endswith('deviation') and not column.startswith('consumption') and not column.startswith('salary')]].to_numpy(), weights=df_b.weight.to_numpy()).fit()
        salary_model = sm.WLS(df_b.consumption_cex_deviation.to_numpy(), df_b.loc[:, [column for column in df_b.columns if column.endswith('deviation') and not column.startswith('consumption') and not column.startswith('earnings')]].to_numpy(), weights=df_b.weight.to_numpy()).fit()
    earnings_model.save(os.path.join(data, 'earnings_bootstrap_' + str(b) + '_method_' + str(m) + '.pickle'))
    salary_model.save(os.path.join(data, 'salary_bootstrap_' + str(b) + '_method_' + str(m) + '.pickle'))

    # Save the data frame
    df_cex.to_csv(os.path.join(data, 'dignity_cex_bootstrap_' + str(b) + '_method_' + str(m) + '.csv'), index=False)
    del df_b, df_cex, df

# Calculate CEX consumption statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(bootstrap)(n) for n in range(2000))