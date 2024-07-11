# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Define variable columns
columns = ['RPTYEAR',
           'SEX',
           'RACE',
           'AGEYREND']

# Define variable types
types = {'RPTYEAR':  'int',
         'SEX':      'int',
         'RACE':     'int',
         'AGEYREND': 'int'}

# Load the data
ncr = pd.read_csv(os.path.join(ncr_r_data, 'ncr.tsv'), delimiter='\t', header=0, usecols=columns, dtype=types)

# Recode the race variables
ncr.loc[:, 'RACE'] = ncr.RACE.map({1: 1, 2: 2, 3: -1, 4: 0, 9: 0})
ncr = ncr.loc[ncr.RACE.isin([1, 2]), :]

# Calculate the number of inmates by year, race and age group
ncr.loc[:, 'incarcerated'] = 1
ncr = ncr.groupby(['RPTYEAR', 'RACE', 'AGEYREND'], as_index=False).agg({'incarcerated': 'sum'})

# Rename variables
ncr = ncr.rename(columns={'RPTYEAR': 'year', 'RACE': 'race', 'AGEYREND': 'agegroup'})

# Load the population data
df = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
df = df.loc[df.year.isin(ncr.year.unique()) & df.race.isin([1, 2]) & (df.latin == 0), :]
df = df.groupby(['year', 'race', 'age'], as_index=False).agg({'population': 'sum'})

# Create the age group variable
df.loc[df.age < 18, 'agegroup'] = 0
df.loc[(df.age >= 18) & (df.age < 25), 'agegroup'] = 1
df.loc[(df.age >= 25) & (df.age < 35), 'agegroup'] = 2
df.loc[(df.age >= 35) & (df.age < 45), 'agegroup'] = 3
df.loc[(df.age >= 45) & (df.age < 55), 'agegroup'] = 4
df.loc[df.age >= 55, 'agegroup'] = 5

# Compute the population age distribution within age groups
df.loc[:, 'share'] = df.groupby(['year', 'race', 'agegroup'], as_index=False).population.transform(lambda x: x / x.sum()).values

# Merge the two data frames
ncr = pd.merge(df, ncr, how='left')

# Calculate the age distribution of incarcerated individuals
ncr.loc[ncr.incarcerated.isna(), 'incarcerated'] = 0
ncr.loc[:, 'incarcerated'] = ncr.share * ncr.incarcerated
ncr = ncr.drop(['agegroup', 'share'], axis=1)
ncr.loc[:, 'incarcerated'] = ncr.groupby(['year', 'race'], as_index=False).incarcerated.transform(lambda x: x / x.sum()).values

# Load and process the NPS data
nps = pd.read_csv(os.path.join(nps_r_data, 'nps.tsv'), delimiter='\t', usecols=['YEAR', 'STATE', 'WHITEM', 'WHITEF', 'BLACKM', 'BLACKF'])
nps = nps.loc[(nps.STATE == 'US') & nps.YEAR.isin(ncr.year.unique()), :].drop('STATE', axis=1)
nps.loc[:, 'WHITE'] = nps.WHITEM + nps.WHITEF
nps.loc[:, 'BLACK'] = nps.BLACKM + nps.BLACKF
nps_white = nps.loc[:, ['YEAR', 'WHITE']].rename(columns={'WHITE': 'total'})
nps_white.loc[:, 'race'] = 1
nps_black = nps.loc[:, ['YEAR', 'BLACK']].rename(columns={'BLACK': 'total'})
nps_black.loc[:, 'race'] = 2
nps = nps_white.append(nps_black, ignore_index=True).rename(columns={'YEAR': 'year'})

# Calculate the number of inmates at every age
ncr = pd.merge(ncr, nps, how='left')
ncr.loc[:, 'incarcerated'] = ncr.incarcerated * ncr.total
ncr = ncr.drop('total', axis=1)

# Save the data
ncr.to_csv(os.path.join(ncr_f_data, 'ncr.csv'), index=False)
