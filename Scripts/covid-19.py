# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sodapy import Socrata
import scipy.interpolate as interpolate
from dotenv import load_dotenv
load_dotenv()
import os

# Import functions and directories
from functions import *
from directories import *

################################################################################
#                                                                              #
# This script calculates the COVID-19 mortality rates by race and age since    #
# April 2020.                                                                  #
#                                                                              #
################################################################################

# Retrieve the COVID-19 mortality data
client = Socrata('data.cdc.gov', os.getenv('cdc_api_key'))
covid = pd.DataFrame.from_records(client.get('m74n-4hbs', limit=500000)).loc[:, ['mmwryear', 'mmwrweek', 'raceethnicity', 'sex', 'agegroup', 'covid19_weighted']]

# Find the first day of each week
covid.loc[:, 'day'] = 7 * covid.mmwrweek.astype('int') - 6

# Find the date of each observation
covid.loc[:, 'date'] = pd.to_datetime(covid.mmwryear.astype(str), format='%Y') + pd.to_timedelta(covid.day.astype(str) + 'days')

# Keep the covid-19 death counts from April 2020 to April 2021
start_date = pd.to_datetime('2020-04-01')
factor = 365 / (covid.date.max() - start_date).days
covid = covid.loc[covid.date >= start_date, :].drop(['mmwryear', 'mmwrweek', 'day', 'date'], axis=1)

# Recode the race variable
covid = covid.loc[covid.raceethnicity.isin(['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'All Race/Ethnicity Groups']), :]
covid.loc[:, 'latin'] = covid.raceethnicity.map({'Non-Hispanic White': 0, 'Non-Hispanic Black': 0, 'Hispanic': 1, 'All Race/Ethnicity Groups': -1})
covid.loc[:, 'raceethnicity'] = covid.raceethnicity.map({'Non-Hispanic White': 1, 'Non-Hispanic Black': 2, 'Hispanic': -1, 'All Race/Ethnicity Groups': -1})

# Drop the gender breakdown
covid = covid.loc[covid.sex == 'All Sexes', :].drop('sex', axis=1)

# Drop the aggregate and missing age categories and recode the age variable
covid = covid.loc[~covid.agegroup.isin(['All Ages', 'Not stated']), :]
covid.loc[:, 'agegroup'] = covid.agegroup.map({'0-14 Years':  1,
                                               '15-19 Years': 2,
                                               '20-24 Years': 3,
                                               '25-29 Years': 4,
                                               '30-34 Years': 5,
                                               '35-39 Years': 6,
                                               '40-44 Years': 7,
                                               '45-49 Years': 8,
                                               '50-54 Years': 9,
                                               '55-59 Years': 10,
                                               '60-64 Years': 11,
                                               '65-69 Years': 12,
                                               '70-74 Years': 13,
                                               '75-79 Years': 14,
                                               '80-84 Years': 15,
                                               '85+':         16})

# Aggregate COVID-19 deaths by race and age group
covid = covid.astype({'covid19_weighted': 'int'})
covid = covid.groupby(['raceethnicity', 'latin', 'agegroup'], as_index=False).agg({'covid19_weighted': 'sum'})
covid.loc[:, 'covid19_weighted'] = covid.covid19_weighted * factor

# Rename variables
covid = covid.rename(columns={'raceethnicity': 'race', 'covid19_weighted': 'deaths'})

# Calculate the 2020 U.S. population by race and age
df = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
df_black_white = df.loc[(df.year == 2020) & df.race.isin([1, 2]) & (df.latin == 0), :].groupby(['race', 'latin', 'age'], as_index=False).agg({'population': 'sum'})
df_latin = df.loc[(df.year == 2020) & (df.latin == 1), :].groupby(['latin', 'age'], as_index=False).agg({'population': 'sum'})
df_latin.loc[:, 'race'] = -1
df_all = df.loc[df.year == 2020, :].groupby('age', as_index=False).agg({'population': 'sum'})
df_all.loc[:, 'race'] = -1
df_all.loc[:, 'latin'] = -1
df = df_black_white.append(df_latin.append(df_all, ignore_index=True), ignore_index=True)

# Group the age variable
df_group = df.copy(deep=True)
df_group.loc[df_group.age < 15, 'agegroup'] = 1
for g in range(14):
    df_group.loc[(df_group.age >= 15 + 5 * g) & (df_group.age < 15 + 5 * (g + 1)), 'agegroup'] = g + 2
df_group.loc[df_group.age >= 85, 'agegroup'] = 16
df_group = df_group.groupby(['race', 'latin', 'agegroup'], as_index=False).agg({'population': 'sum'})

# Merge the two data frames
df_group = pd.merge(df_group, covid, how='left')

# Compute the COVID-19 mortality rate by race and age group
df_group.loc[:, 'M'] = df_group.deaths / df_group.population

# Create a data frame with all levels of all variables
df_covid = expand({'race': df_group.race.unique(), 'latin': df_group.latin.unique(), 'age': range(100)})
df_covid = df_covid.loc[(df_covid.race.isin([1, 2]) & (df_covid.latin == 0)) | ((df_covid.race == -1) & df_covid.latin.isin([-1, 1])), :]

# Define a function to approximate mortality rates at every age between 0 and 99
def mortality(x):
    # Define the age midpoints
    midpoints = np.asarray([0, 7, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 93, 99])

    # Log-linearly approximate the mortality rates from age 0 to 100
    approximation = interpolate.interp1d(midpoints, np.append(np.append(np.log(x).iloc[0] - (midpoints[1] - midpoints[0]) * 0.112, np.log(x)), np.log(x).iloc[-1] + (midpoints[-1] - midpoints[-2]) * 0.112))
    M = np.exp(approximation(np.linspace(0, 99, 100)))
    M[0] = 0
    M[M > 1] = 1

    # Return the mortality rates
    return M

# Approximate the COVID-19 mortality rate by race and single years of age
df_covid.loc[(df_covid.race == 1) & (df_covid.latin == 0), 'M'] = mortality(df_group.loc[(df_group.race == 1) & (df_group.latin == 0), 'M'])
df_covid.loc[(df_covid.race == 2) & (df_covid.latin == 0), 'M'] = mortality(df_group.loc[(df_group.race == 2) & (df_group.latin == 0), 'M'])
df_covid.loc[(df_covid.race == -1) & (df_covid.latin == 1), 'M'] = mortality(df_group.loc[(df_group.race == -1) & (df_group.latin == 1), 'M'])
df_covid.loc[(df_covid.race == -1) & (df_covid.latin == -1), 'M'] = mortality(df_group.loc[(df_group.race == -1) & (df_group.latin == -1), 'M'])

# Compute the approximated number of COVID-19 deaths by race and single years of age
df_covid = pd.merge(df_covid, df, how='left')
df_covid.loc[:, 'deaths_approx'] = df_covid.M * df_covid.population

# Group the age variable
df_covid.loc[df_covid.age < 15, 'agegroup'] = 1
for g in range(14):
    df_covid.loc[(df_covid.age >= 15 + 5 * g) & (df_covid.age < 15 + 5 * (g + 1)), 'agegroup'] = g + 2
df_covid.loc[df_covid.age >= 85, 'agegroup'] = 16

# Merge with the actual number of COVID-19 deaths by race and age group
df_covid = df_covid.astype({'race':          'int',
                            'latin':         'int',
                            'agegroup':      'int',
                            'deaths_approx': 'float'})
covid = pd.merge(df_covid, covid, how='left')

# Re-scale the approximated number of COVID-19 deaths by race and age group
covid.loc[:, 'rescale'] = covid.deaths.values / covid.groupby(['race', 'latin', 'agegroup'], as_index=False).deaths_approx.transform(lambda x: x.sum()).values.squeeze()
covid = covid.drop(['deaths', 'agegroup', 'M'], axis=1)
covid.loc[:, 'deaths'] = covid.deaths_approx * covid.rescale
covid = covid.drop(['deaths_approx', 'rescale'], axis=1)

# Save the data
covid.to_csv(os.path.join(cdc_f_data, 'covid-19.csv'), index=False)