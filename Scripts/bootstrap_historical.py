# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.api as sm
import itertools
from joblib import Parallel, delayed
import os

# Find the number of available CPUs
n_cpu = os.cpu_count()

# Set the data directory
data = os.path.join('/scratch/users/jfbrou/Dignity', 'Data')

# Define a weighted average function
def weighted_average(x, data=None, weights=None):
    if np.sum(data.loc[x[x.notna()].index, weights]) == 0:
        return np.nan
    else:
        return np.average(x[x.notna()], weights=data.loc[x[x.notna()].index, weights])

# Define a function to create a data frame of the right form
def expand(dictionary):
    rows = itertools.product(*dictionary.values())
    return pd.DataFrame.from_records(rows, columns=dictionary.keys())

# Define the consumption and leisure interpolation/extrapolation function
def filter(x, penalty):
    # Linearly interpolate the missing values coming before the oldest available age
    x = x.interpolate(limit_direction='backward')

    # HP-filter the resulting series
    x[x.notna()] = sm.tsa.filters.hpfilter(x[x.notna()], penalty)[1]

    # Keep the resulting series constant for missing ages coming after the oldest available age
    return x.interpolate(method='ffill', limit_direction='forward')

################################################################################
#                                                                              #
# This section of the script produces the bootstrap consumption and leisure    #
# statistics between 1940 and 2020.                                            #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(data, 'cex.csv'))

# Keep the imputation sample
cex = cex.loc[cex.complete == 1, :]

# Create race binary variables
cex = pd.concat([cex, pd.get_dummies(cex.race.astype('int'), prefix='race')], axis=1)

# Create education binary variables
cex.loc[cex.education.isna() | (cex.age < 30), 'education'] = 4
cex = pd.concat([cex, pd.get_dummies(cex.education.astype('int'), prefix='education')], axis=1)

# Recode the gender variable
cex.loc[:, 'gender'] = cex.gender.replace({1: 1, 2: 0})

# Define a function to read the data by year
def year_chunk(file, chunksize=1e6):
    reader = pd.read_csv(file, iterator=True, chunksize=chunksize)
    chunk = pd.DataFrame()
    for i in reader:
        unique_years = np.sort(i.year.unique())
        if len(unique_years) == 1:
            chunk = chunk.append(i, ignore_index=True)
        else:
            chunk = chunk.append(i.loc[i.year == unique_years[0], :], ignore_index=True)
            yield chunk
            chunk = pd.DataFrame()
            chunk = chunk.append(i.loc[i.year == unique_years[1], :], ignore_index=True)
    yield chunk
chunks = year_chunk(os.path.join(data, 'acs_bootstrap.csv'), chunksize=1e6)

# Define a function to impute consumption from income by year across bootstrap samples
def acs_bootstrap(sample):
    # Sample from the CEX data
    df = pd.DataFrame()
    for cex_year in cex.year.unique():
        df = df.append(cex.loc[cex.year == cex_year, :].sample(n=cex.loc[cex.year == cex_year, :].shape[0], replace=True, weights='weight', random_state=sample), ignore_index=True)

    # Calculate average consumption, income and demographics by year in the current bootstrap sample
    columns = ['consumption_cex', 'earnings', 'salary'] + ['race_' + str(i) for i in range(1, 4 + 1)] \
                                                        + ['education_' + str(i) for i in range(1, 4 + 1)] \
                                                        + ['family_size', 'latin', 'gender', 'age']
    functions = [lambda x: weighted_average(x, data=df, weights='weight')] * len(columns)
    names = [column + '_average' for column in columns]
    d_functions = dict(zip(columns, functions))
    d_names = dict(zip(columns, names))
    df = pd.merge(df, df.groupby('year', as_index=False).agg(d_functions).rename(columns=d_names), how='left')

    # Calculate the percentage deviation of consumption, income and demographics from their annual average
    for column in columns:
        df.loc[:, column + '_Δ'] = (df.loc[:, column] - df.loc[:, column + '_average']) / df.loc[:, column + '_average']

    # Fit the OLS models for consumption
    earnings_model = sm.WLS(df.consumption_cex_Δ.to_numpy(), df.loc[:, [column for column in df.columns if column.endswith('Δ') and not column.startswith('consumption') and not column.startswith('earnings')]].to_numpy(), weights=df.weight.to_numpy()).fit()
    salary_model = sm.WLS(df.consumption_cex_Δ.to_numpy(), df.loc[:, [column for column in df.columns if column.endswith('Δ') and not column.startswith('consumption') and not column.startswith('salary')]].to_numpy(), weights=df.weight.to_numpy()).fit()

    # Load and process the ACS data
    df = pd.DataFrame()
    for chunk in chunks:
        chunk = chunk.sample(n=chunk.shape[0], replace=True, weights='weight', random_state=sample)

        # Create race binary variables
        chunk = pd.concat([chunk, pd.get_dummies(chunk.race.astype('int'), prefix='race')], axis=1)

        # Create education binary variables
        chunk = pd.concat([chunk, pd.get_dummies(chunk.education.astype('int'), prefix='education')], axis=1)

        # Calculate the percentage deviation of each imputation variable from their average
        chunk.loc[:, 'earnings_Δ'] = (chunk.earnings - np.average(chunk.earnings, weights=chunk.weight)) / np.average(chunk.earnings, weights=chunk.weight)
        chunk.loc[:, 'race_1_Δ'] = (chunk.race_1 - np.average(chunk.race_1, weights=chunk.weight)) / np.average(chunk.race_1, weights=chunk.weight)
        chunk.loc[:, 'race_2_Δ'] = (chunk.race_2 - np.average(chunk.race_2, weights=chunk.weight)) / np.average(chunk.race_2, weights=chunk.weight)
        chunk.loc[:, 'race_3_Δ'] = (chunk.race_3 - np.average(chunk.race_3, weights=chunk.weight)) / np.average(chunk.race_3, weights=chunk.weight)
        chunk.loc[:, 'race_4_Δ'] = (chunk.race_4 - np.average(chunk.race_4, weights=chunk.weight)) / np.average(chunk.race_4, weights=chunk.weight)
        chunk.loc[:, 'education_1_Δ'] = (chunk.education_1 - np.average(chunk.education_1, weights=chunk.weight)) / np.average(chunk.education_1, weights=chunk.weight)
        chunk.loc[:, 'education_2_Δ'] = (chunk.education_2 - np.average(chunk.education_2, weights=chunk.weight)) / np.average(chunk.education_2, weights=chunk.weight)
        chunk.loc[:, 'education_3_Δ'] = (chunk.education_3 - np.average(chunk.education_3, weights=chunk.weight)) / np.average(chunk.education_3, weights=chunk.weight)
        chunk.loc[:, 'education_4_Δ'] = (chunk.education_4 - np.average(chunk.education_4, weights=chunk.weight)) / np.average(chunk.education_4, weights=chunk.weight)
        chunk.loc[:, 'family_size_Δ'] = (chunk.family_size - np.average(chunk.family_size, weights=chunk.weight)) / np.average(chunk.family_size, weights=chunk.weight)
        chunk.loc[:, 'latin_Δ'] = (chunk.latin - np.average(chunk.latin, weights=chunk.weight)) / np.average(chunk.latin, weights=chunk.weight)
        chunk.loc[:, 'gender_Δ'] = (chunk.gender.map({1: 1, 2: 0}) - np.average(chunk.gender.map({1: 1, 2: 0}), weights=chunk.weight)) / np.average(chunk.gender.map({1: 1, 2: 0}), weights=chunk.weight)
        chunk.loc[:, 'age_Δ'] = (chunk.age - np.average(chunk.age, weights=chunk.weight)) / np.average(chunk.age, weights=chunk.weight)

        # Impute consumption
        if chunk.year.unique() == 1940:
            chunk.loc[:, 'consumption'] = salary_model.predict(chunk.loc[:, [column for column in chunk.columns if column.endswith('Δ')]])
        else:
            chunk.loc[:, 'consumption'] = earnings_model.predict(chunk.loc[:, [column for column in chunk.columns if column.endswith('Δ')]])
        chunk = chunk.drop([column for column in chunk.columns if column.endswith('Δ')], axis=1)

        # Merge with the BEA data
        bea = pd.read_csv(os.path.join(data, 'bea.csv'))
        chunk = pd.merge(chunk, bea, how='left')

        # Re-scale personal earnings and consumption expenditures such that it aggregates to the NIPA values
        chunk.loc[:, 'earnings'] = chunk.earnings_nipa + chunk.earnings_nipa * (chunk.earnings - np.average(chunk.earnings, weights=chunk.weight)) / np.average(chunk.earnings, weights=chunk.weight)
        chunk.loc[chunk.missing_earnings == True, 'earnings'] = np.nan
        chunk.loc[:, 'consumption'] = chunk.consumption_nipa + chunk.consumption_nipa * chunk.consumption
        chunk = chunk.drop(['earnings_nipa', 'consumption_nipa'], axis=1)

        # Append the data frames for all chunks
        df = df.append(chunk, ignore_index=True)

    # Create the leisure weight variable
    df.loc[df.year != 1950, 'leisure_weight'] = df.weight
    df.loc[df.year == 1950, 'leisure_weight'] = df.SLWT
    df = df.drop('SLWT', axis=1)

    # Calculate the ratio of the average of the first leisure variable to the average of the other leisure variables in 1980 and 1990
    sample = ((df.year == 1980) | (df.year == 1990))
    scale_2 = weighted_average(df.loc[sample, 'leisure_1'], data=df.loc[sample, :], weights='leisure_weight') / weighted_average(df.loc[sample, 'leisure_2'], data=df.loc[sample, :], weights='leisure_weight')
    scale_3 = weighted_average(df.loc[sample, 'leisure_1'], data=df.loc[sample, :], weights='leisure_weight') / weighted_average(df.loc[sample, 'leisure_3'], data=df.loc[sample, :], weights='leisure_weight')
    scale_4 = weighted_average(df.loc[sample, 'leisure_1'], data=df.loc[sample, :], weights='leisure_weight') / weighted_average(df.loc[sample, 'leisure_4'], data=df.loc[sample, :], weights='leisure_weight')

    # Rescale the leisure variables
    df.loc[:, 'leisure_2'] = scale_2 * df.leisure_2
    df.loc[:, 'leisure_3'] = scale_3 * df.leisure_3
    df.loc[:, 'leisure_4'] = scale_4 * df.leisure_4

    # Create a unique leisure variable
    df.loc[df.leisure_1.notna(), 'leisure'] = df.leisure_1
    df.loc[df.leisure_1.isna() & df.leisure_2.notna(), 'leisure'] = df.leisure_2
    df.loc[df.leisure_1.isna() & df.leisure_2.isna() & df.leisure_3.notna(), 'leisure'] = df.leisure_3
    df.loc[df.leisure_1.isna() & df.leisure_2.isna() & df.leisure_3.isna() & df.leisure_4.notna(), 'leisure'] = df.leisure_4
    df = df.drop(['leisure_1', 'leisure_2', 'leisure_3', 'leisure_4'], axis=1)

    # Create a data frame with all levels of all variables
    df = df.loc[df.race.isin([1, 2]), :]
    df_levels = expand({'year': df.year.unique(), 'race': df.race.unique(), 'age': df.age.unique()})

    # Calculate weighted averages of each variable
    df = df.groupby(['year', 'race', 'age'], as_index=False).agg({'leisure':        lambda x: weighted_average(x, data=df, weights='leisure_weight'),
    				   								              'consumption':    lambda x: weighted_average(x, data=df, weights='weight'),
			                                                      'leisure_weight': 'sum',
				                                                  'weight':         'sum'})

    # Merge the data frames
    df = pd.merge(df_levels, df, how='left')
    df.loc[df.leisure_weight.isna(), 'leisure_weight'] = 0
    df.loc[df.weight.isna(), 'weight'] = 0

    # Normalize consumption
    df.loc[:, 'consumption'] = df.consumption / weighted_average(df.loc[df.year == 2019, 'consumption'], data=df, weights='weight')

    # Calculate ACS consumption statistics by year, race and age
    df_consumption = df.groupby(['year', 'race', 'age'], as_index=False).agg({'consumption': lambda x: weighted_average(x, data=df, weights='weight')}).rename(columns={'consumption': 'c_bar'})
    df_consumption = pd.merge(expand({'year': df_consumption.year.unique(), 'age': range(101), 'race': [1, 2], 'sample': [sample]}), df_consumption, how='left')
    df_consumption.loc[:, 'c_bar'] = df_consumption.groupby(['year', 'race'], as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values

    # Calculate ACS leisure statistics by year, race and age
    df_leisure = df.groupby(['year', 'race', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=df, weights='leisure_weight')}).rename(columns={'leisure': 'ℓ_bar'})
    df_leisure = pd.merge(expand({'year': df_leisure.year.unique(), 'age': range(101), 'race': [1, 2], 'sample': [sample]}), df_leisure, how='left')
    df_leisure.loc[:, 'ℓ_bar'] = df_leisure.groupby(['year', 'race'], as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
    df_leisure.loc[df_leisure.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1

    # Merge and return the data frames
    df = pd.merge(df_consumption, df_leisure, how='left')
    return df

# Calculate ACS consumption and leisure statistics across bootstrap samples
results = Parallel(n_jobs=n_cpu)(delayed(acs_bootstrap)(sample) for sample in range(1000))
df_acs_bootstrap = pd.DataFrame()
for sample in range(1000):
    df_acs_bootstrap = df_acs_bootstrap.append(results[sample], ignore_index=True)
del cex

# Load the survival rates data
survival = pd.read_csv(os.path.join(data, 'survival.csv'))
survival = survival.loc[survival.year.isin(range(1940, 2020 + 1)) & survival.race.isin([1, 2]) & (survival.latin == -1) & (survival.gender == -1), :].drop(['latin', 'gender'], axis=1)

# Merge and save the bootstrap data frames
dignity_bootstrap = pd.merge(df_acs_bootstrap, survival, how='left')
dignity_bootstrap.to_csv(os.path.join(data, 'dignity_bootstrap_historical.csv'), index=False)
