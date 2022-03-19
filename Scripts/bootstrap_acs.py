# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from joblib import Parallel, delayed
import statsmodels.api as sm
import os

# Find the number of available CPUs
n_cpu = os.cpu_count()

# Import functions
from functions import *

# Set the Sherlock data directory
data = '/scratch/users/jfbrou/Dignity'

# Define a function to read the data by year
def year_chunk(file, chunksize=1e6):
    iterator = pd.read_csv(file, iterator=True, chunksize=chunksize)
    chunk = pd.DataFrame()
    for df in iterator:
        unique_years = np.sort(df.year.unique())
        if len(unique_years) == 1:
            chunk = chunk.append(df, ignore_index=True)
        else:
            chunk = chunk.append(df.loc[df.year == unique_years[0], :], ignore_index=True)
            yield chunk
            chunk = pd.DataFrame()
            chunk = chunk.append(df.loc[df.year == unique_years[1], :], ignore_index=True)
    yield chunk
chunks = year_chunk(os.path.join(data, 'bootstrap_acs.csv'), chunksize=1e6)

# Define a function to calculate ACS consumption and leisure statistics across bootstrap samples
def bootstrap(b):
    # Load and process the ACS data year by year
    df_b = pd.DataFrame()
    for chunk in chunks:
        # Sample from the data
        chunk = chunk.sample(frac=1, replace=True, random_state=b)

        # Impute consumption
        if int(chunk.year.unique()) == 1940:
            model = pd.read_csv(os.path.join(data, 'salary_bootstrap_' + str(b) + '.csv'))
            model.loc[model.variable == 'salary_deviation', 'variable'] = 'earnings_deviation'
        else:
            model = pd.read_csv(os.path.join(data, 'earnings_bootstrap_' + str(b) + '.csv'))
        chunk.loc[:, 'consumption'] = float(model.loc[model.variable == 'Intercept', 'coefficient'].values) + chunk.earnings_deviation * float(model.loc[model.variable == 'earnings_deviation', 'coefficient'].values) \
                                                                                                            + chunk.race_1_deviation * float(model.loc[model.variable == 'race_1_deviation', 'coefficient'].values) \
                                                                                                            + chunk.race_2_deviation * float(model.loc[model.variable == 'race_2_deviation', 'coefficient'].values) \
                                                                                                            + chunk.race_3_deviation * float(model.loc[model.variable == 'race_3_deviation', 'coefficient'].values) \
                                                                                                            + chunk.race_4_deviation * float(model.loc[model.variable == 'race_4_deviation', 'coefficient'].values) \
                                                                                                            + chunk.education_1_deviation * float(model.loc[model.variable == 'education_1_deviation', 'coefficient'].values) \
                                                                                                            + chunk.education_2_deviation * float(model.loc[model.variable == 'education_2_deviation', 'coefficient'].values) \
                                                                                                            + chunk.education_3_deviation * float(model.loc[model.variable == 'education_3_deviation', 'coefficient'].values) \
                                                                                                            + chunk.education_4_deviation * float(model.loc[model.variable == 'education_4_deviation', 'coefficient'].values) \
                                                                                                            + chunk.family_size_deviation * float(model.loc[model.variable == 'family_size_deviation', 'coefficient'].values) \
                                                                                                            + chunk.latin_deviation * float(model.loc[model.variable == 'latin_deviation', 'coefficient'].values) \
                                                                                                            + chunk.gender_deviation * float(model.loc[model.variable == 'gender_deviation', 'coefficient'].values) \
                                                                                                            + chunk.age_deviation * float(model.loc[model.variable == 'age_deviation', 'coefficient'].values)

        # Re-scale consumption expenditures such that it aggregates to the NIPA values
        chunk.loc[:, 'consumption'] = chunk.consumption_nipa + chunk.consumption_nipa * chunk.consumption

        # Calculate weighted averages of each variable
        if chunk.year.unique() == 1950:
            def f(x):
                d = {}
                d['consumption'] = np.average(x.consumption, weights=x.weight)
                d['weight'] = np.sum(x.weight)
                return pd.Series(d, index=[key for key, value in d.items()])
            def f_leisure(x):
                d = {}
                d['leisure_1'] = np.average(x.leisure_1, weights=x.SLWT)
                d['leisure_2'] = np.average(x.leisure_2, weights=x.SLWT)
                d['leisure_3'] = np.average(x.leisure_3, weights=x.SLWT)
                d['leisure_4'] = np.average(x.leisure_4, weights=x.SLWT)
                d['leisure_weight'] = np.sum(x.SLWT)
                return pd.Series(d, index=[key for key, value in d.items()])
            chunk_leisure = chunk.loc[chunk.SLWT != 0, :].groupby(['year', 'race', 'age'], as_index=False).apply(f_leisure)
            chunk = chunk.groupby(['year', 'race', 'age'], as_index=False).apply(f)
            chunk = pd.merge(chunk, chunk_leisure, how='left')
        else:
            def f(x):
                d = {}
                d['leisure_1'] = np.average(x.leisure_1, weights=x.weight)
                d['leisure_2'] = np.average(x.leisure_2, weights=x.weight)
                d['leisure_3'] = np.average(x.leisure_3, weights=x.weight)
                d['leisure_4'] = np.average(x.leisure_4, weights=x.weight)
                d['consumption'] = np.average(x.consumption, weights=x.weight)
                d['leisure_weight'] = np.sum(x.weight)
                d['weight'] = np.sum(x.weight)
                return pd.Series(d, index=[key for key, value in d.items()])
            chunk = chunk.groupby(['year', 'race', 'age'], as_index=False).apply(f)

        # Append the data frames for all chunks
        df_b = df_b.append(chunk, ignore_index=True)

    # Calculate the ratio of the average of the first leisure variable to the average of the other leisure variables in 1980 and 1990
    sample = ((df_b.year == 1980) | (df_b.year == 1990))
    scale_2 = np.average(df_b.loc[sample, 'leisure_1'], weights=df_b.loc[sample, 'leisure_weight']) / np.average(df_b.loc[sample, 'leisure_2'], weights=df_b.loc[sample, 'leisure_weight'])
    scale_3 = np.average(df_b.loc[sample, 'leisure_1'], weights=df_b.loc[sample, 'leisure_weight']) / np.average(df_b.loc[sample, 'leisure_3'], weights=df_b.loc[sample, 'leisure_weight'])
    scale_4 = np.average(df_b.loc[sample, 'leisure_1'], weights=df_b.loc[sample, 'leisure_weight']) / np.average(df_b.loc[sample, 'leisure_4'], weights=df_b.loc[sample, 'leisure_weight'])

    # Rescale the leisure variables
    df_b.loc[:, 'leisure_2'] = scale_2 * df_b.leisure_2
    df_b.loc[:, 'leisure_3'] = scale_3 * df_b.leisure_3
    df_b.loc[:, 'leisure_4'] = scale_4 * df_b.leisure_4

    # Create a unique leisure variable
    df_b.loc[df_b.leisure_1.notna(), 'leisure'] = df_b.leisure_1
    df_b.loc[df_b.leisure_1.isna() & df_b.leisure_2.notna(), 'leisure'] = df_b.leisure_2
    df_b.loc[df_b.leisure_1.isna() & df_b.leisure_2.isna() & df_b.leisure_3.notna(), 'leisure'] = df_b.leisure_3
    df_b.loc[df_b.leisure_1.isna() & df_b.leisure_2.isna() & df_b.leisure_3.isna() & df_b.leisure_4.notna(), 'leisure'] = df_b.leisure_4
    df_b = df_b.drop(['leisure_1', 'leisure_2', 'leisure_3', 'leisure_4'], axis=1)

    # Normalize consumption
    df_b.loc[:, 'consumption_simple'] = df_b.consumption / np.average(df_b.loc[(df_b.year == 2019) & (df_b.race == 1), 'consumption'], weights=df_b.loc[(df_b.year == 2019) & (df_b.race == 1), 'weight'])
    df_b.loc[:, 'consumption'] = df_b.consumption / np.average(df_b.loc[df_b.year == 2019, 'consumption'], weights=df_b.loc[df_b.year == 2019, 'weight'])

    # Instantiate empty data frames
    df_acs_consumption = pd.DataFrame()
    df_acs_leisure = pd.DataFrame()

    # Calculate ACS consumption statistics by year and age in the current bootstrap sample
    df = df_b.groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'c_bar': np.average(x.consumption, weights=x.weight)}))
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'bootstrap': [b], 'simple': [False]}), df, how='left')
    df.loc[:, 'c_bar'] = df.groupby('year', as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
    df_acs_consumption = df_acs_consumption.append(df, ignore_index=True)

    # Calculate ACS consumption statistics by year, race and age in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'c_bar': np.average(x.consumption, weights=x.weight)}))
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'bootstrap': [b], 'simple': [False]}), df, how='left')
    df.loc[:, 'c_bar'] = df.groupby(['year', 'race'], as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
    df_acs_consumption = df_acs_consumption.append(df, ignore_index=True)

    # Calculate ACS consumption statistics by year and race in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption_average': np.log(np.average(x.consumption_simple, weights=x.weight))}))
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'bootstrap': [b], 'simple': [True]}), df, how='left')
    df_acs_consumption = df_acs_consumption.append(df, ignore_index=True)

    # Calculate ACS leisure statistics by year and age in the current bootstrap sample
    df = df_b.groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.leisure, weights=x.weight)}))
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'bootstrap': [b], 'simple': [False]}), df, how='left')
    df.loc[:, 'ell_bar'] = df.groupby('year', as_index=False)['ell_bar'].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_acs_leisure = df_acs_leisure.append(df, ignore_index=True)

    # Calculate ACS leisure statistics by year, race and age in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.leisure, weights=x.weight)}))
    df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'bootstrap': [b], 'simple': [False]}), df, how='left')
    df.loc[:, 'ell_bar'] = df.groupby(['year', 'race'], as_index=False)['ell_bar'].transform(lambda x: filter(x, 100)).values
    df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_acs_leisure = df_acs_leisure.append(df, ignore_index=True)

    # Calculate ACS leisure statistics by year and race in the current bootstrap sample
    df = df_b.loc[df_b.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'leisure_average': np.average(x.leisure, weights=x.weight)}))
    df = pd.merge(expand({'year': df.year.unique(), 'age': [np.nan], 'race': [1, 2], 'bootstrap': [b], 'simple': [True]}), df, how='left')
    df_acs_leisure = df_acs_leisure.append(df, ignore_index=True)

    # Merge and save the two data frames
    df_acs = pd.merge(df_acs_consumption, df_acs_leisure, how='left')
    df_acs.to_csv(os.path.join(data, 'dignity_acs_bootstrap_' + str(b) + '.csv'), index=False)
    del df_b, df_acs, df_acs_consumption, df_acs_leisure, df

# Calculate ACS consumption and leisure statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(bootstrap)(b) for b in range(5))