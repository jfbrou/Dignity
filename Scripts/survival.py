# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
load_dotenv()
import os

# Import functions and directories
from functions import *
from directories import *

# Define the Gompertz function
def gompertz(x, data=None):
    # Estimate the model
    model = LinearRegression().fit(np.array(range(65, 84 + 1)).reshape(-1, 1), np.log(x.iloc[:20].to_numpy()))

    # Return the mortality rates for ages 85 to 99
    return np.append(x.iloc[:20], np.exp(model.intercept_ + model.coef_ * range(85, 100)))

# Create a cdc data frame
cdc_df = pd.DataFrame()

################################################################################
#                                                                              #
# This section of the script computes survival rates for all individuals.      #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))

# Aggregate mortality by year and age
cdc = cdc.groupby(['year', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]), :]

# Aggregate population at risk by year and age
population = population.groupby(['year', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby('year', as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby('year', as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [-1], 'latin': [-1], 'gender': [-1]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby('year', as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby('year', as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by gender.                #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))

# Aggregate mortality by year, gender and age
cdc = cdc.groupby(['year', 'gender', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]), :]

# Aggregate population at risk by year, gender and age
population = population.groupby(['year', 'gender', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby(['year', 'gender'], as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby(['year', 'gender'], as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [-1], 'latin': [-1], 'gender': [1, 2]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(['year', 'gender'], as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby(['year', 'gender'], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race.                  #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))

# Aggregate mortality by year, race and age
cdc = cdc.groupby(['year', 'race', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]) & ((population.race == 1) | (population.race == 2)), :]

# Aggregate population at risk by year, race and age
population = population.groupby(['year', 'race', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby(['year', 'race'], as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby(['year', 'race'], as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [1, 2], 'latin': [-1], 'gender': [-1]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(['year', 'race'], as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby(['year', 'race'], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race and gender.       #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))

# Aggregate mortality by year, race, gender and age
cdc = cdc.groupby(['year', 'race', 'gender', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]) & ((population.race == 1) | (population.race == 2)), :]

# Aggregate population at risk by year, race, gender and age
population = population.groupby(['year', 'race', 'gender', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby(['year', 'race', 'gender'], as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby(['year', 'race', 'gender'], as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [1, 2], 'latin': [-1], 'gender': [1, 2]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(['year', 'race', 'gender'], as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby(['year', 'race', 'gender'], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for Latinos.              #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))
cdc = cdc.loc[cdc.latin == 1, :]

# Aggregate mortality by year and age
cdc = cdc.groupby(['year', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]) &  (population.latin == 1), :]

# Aggregate population at risk by year and age
population = population.groupby(['year', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby('year', as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby('year', as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [-1], 'latin': [1], 'gender': [-1]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby('year', as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby('year', as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by gender for Latinos.    #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))
cdc = cdc.loc[cdc.latin == 1, :]

# Aggregate mortality by year, gender and age
cdc = cdc.groupby(['year', 'gender', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]) &  (population.latin == 1), :]

# Aggregate population at risk by year, gender and age
population = population.groupby(['year', 'gender', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby(['year', 'gender'], as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby(['year', 'gender'], as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [-1], 'latin': [1], 'gender': [1, 2]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(['year', 'gender'], as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby(['year', 'gender'], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for non-Latinos by race.  #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))
cdc = cdc.loc[cdc.latin == 0, :]

# Aggregate mortality by year, race and age
cdc = cdc.groupby(['year', 'race', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]) & ((population.race == 1) | (population.race == 2)) & (population.latin == 0), :]

# Aggregate population at risk by year, race and age
population = population.groupby(['year', 'race', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby(['year', 'race'], as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby(['year', 'race'], as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [1, 2], 'latin': [0], 'gender': [-1]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(['year', 'race'], as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby(['year', 'race'], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for non-Latinos by race   #
# and gender.                                                                  #
#                                                                              #
################################################################################

# Load the cdc data from 2017 to 2020
cdc = pd.read_csv(os.path.join(cdc_f_data, 'cdc.csv'))
cdc = cdc.loc[cdc.latin == 0, :]

# Aggregate mortality by year, race, gender and age
cdc = cdc.groupby(['year', 'race', 'gender', 'age'], as_index=False).agg({'deaths': 'sum'})

# Load the population data from 2017 to 2020
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[population.year.isin([2017, 2018, 2019, 2020]) & ((population.race == 1) | (population.race == 2)) & (population.latin == 0), :]

# Aggregate population at risk by year, race, gender and age
population = population.groupby(['year', 'race', 'gender', 'age'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df = pd.merge(population, cdc, how='left')

# Smooth the population at risk and deaths with the Beer functions
df.loc[:, 'population'] = df.groupby(['year', 'race', 'gender'], as_index=False).population.transform(lambda x: beerpopulation(x))
df.loc[:, 'deaths'] = df.groupby(['year', 'race', 'gender'], as_index=False).deaths.transform(lambda x: beerdeath(x))

# Keep ages below 85
df = df.loc[df.age < 85, :]

# For observations with more deaths than population, set the population to a missing value to be interpolated
df.loc[df.population < df.deaths, 'population'] = np.nan

# Create a data frame with all levels of all variables
df = pd.merge(expand({'year': [2017, 2018, 2019, 2020], 'age': range(101), 'race': [1, 2], 'latin': [0], 'gender': [1, 2]}), df, how='left')

# Compute mortality rates
df.loc[df.deaths.isna() & (df.age < 85), 'deaths'] = 0
df.loc[:, 'M'] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(['year', 'race', 'gender'], as_index=False).M.transform(gompertz).M.values
df.loc[~df.age.isin(range(65, 100)), 'S'] = 1 - df.loc[~df.age.isin(range(65, 100)), 'M']
df.loc[:, 'S'] = df.groupby(['year', 'race', 'gender'], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S.values

# Append the CDC data frame
df = df.drop(['deaths', 'population', 'M'], axis=1)
cdc_df = cdc_df.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script processes the life tables.                        #
#                                                                              #
################################################################################

# Load the life tables data
lt = pd.read_csv(os.path.join(cdc_r_data, 'lifetables.csv'))

# Compute the mortality rates
lt.loc[:, 'M'] = lt.groupby(['year', 'race', 'latin', 'gender'], as_index=False).S.transform(lambda x: 1 - x.shift(-1) / x).values

# Extrapolate the survival rates
lt.loc[lt.age.isin(range(65, 100)), 'gompertz'] = 1 - lt.loc[lt.age.isin(range(65, 100)), :].groupby(['year', 'race', 'latin', 'gender'], as_index=False).M.transform(gompertz).M.values
lt.loc[~lt.age.isin(range(65, 100)), 'gompertz'] = 1 - lt.loc[~lt.age.isin(range(65, 100)), 'M']
lt.loc[:, 'gompertz'] = lt.groupby(['year', 'race', 'latin', 'gender'], as_index=False).gompertz.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).gompertz.values
lt.loc[lt.S.isna(), 'S'] = lt.gompertz
lt = lt.drop(['M', 'gompertz'], axis=1)

# Adjust the 1950 and 1960 survival rates for Black Americans
adjustment = lt.loc[(lt.year == 1970) & (lt.race == 2), 'S'].values / lt.loc[(lt.year == 1970) & (lt.race == 5), 'S'].values
lt = lt.loc[(lt.year != 1970) | ((lt.year == 1970) & (lt.race != 5)), :]
lt.loc[(lt.year == 1950) & (lt.race == 5), 'S'] = lt.loc[(lt.year == 1950) & (lt.race == 5), 'S'].values * adjustment
lt.loc[(lt.year == 1960) & (lt.race == 5), 'S'] = lt.loc[(lt.year == 1960) & (lt.race == 5), 'S'].values * adjustment
lt.loc[lt.race == 5, 'race'] = 2

# Append the life tables with the above data frame
lt.loc[:, 'lifetable'] = True
lt = lt.append(cdc_df, ignore_index=True)
lt.loc[lt.lifetable.isna(), 'lifetable'] = False

# Adjust the survival rates in 2018 and 2019
for race in [1, 2]:
    for latin in [-1, 0]:
        for gender in [-1, 1, 2]:
            adjustment = lt.loc[(lt.year == 2017) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == True), 'S'].values / lt.loc[(lt.year == 2017) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == False), 'S'].values
            lt.loc[(lt.year == 2018) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == False), 'S'] = lt.loc[(lt.year == 2018) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == False), 'S'] * adjustment
            lt.loc[(lt.year == 2019) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == False), 'S'] = lt.loc[(lt.year == 2019) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == False), 'S'] * adjustment
            lt.loc[(lt.year == 2020) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == False), 'S'] = lt.loc[(lt.year == 2020) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.lifetable == False), 'S'] * adjustment
for gender in [-1, 1, 2]:
    adjustment = lt.loc[(lt.year == 2017) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == True), 'S'].values / lt.loc[(lt.year == 2017) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == False), 'S'].values
    lt.loc[(lt.year == 2018) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == False), 'S'] = lt.loc[(lt.year == 2018) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == False), 'S'] * adjustment
    lt.loc[(lt.year == 2019) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == False), 'S'] = lt.loc[(lt.year == 2019) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == False), 'S'] * adjustment
    lt.loc[(lt.year == 2020) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == False), 'S'] = lt.loc[(lt.year == 2020) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.lifetable == False), 'S'] * adjustment

# Drop the unused tables
sample = (lt.lifetable == False) & (((lt.year == 2018) & ((lt.race == -1) | (lt.latin != -1))) | (lt.year == 2017))
lt = lt.loc[~sample, :].drop('lifetable', axis=1)

# Save the data
lt.to_csv(os.path.join(cdc_f_data, 'survival.csv'), index=False)
