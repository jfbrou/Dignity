# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import os
import calendar

# Import functions and directories
from functions import *
from directories import *

# Start the BEA client
bea = beapy.BEA(key=bea_api_key)

# Define variable columns
columns = [
    'YEAR',
	'SERIAL',
	'ASECWT',
	'AGE',
	'SEX',
	'RACE',
	'FAMSIZE',
	'FAMUNIT',
	'HISPAN',
	'EMPSTAT',
	'LABFORCE',
	'AHRSWORKT',
	'WKSTAT',
	'EDUC',
	'WKSWORK1',
	'UHRSWORKLY',
	'WANTJOB',
	'INCTOT',
	'INCWAGE',
	'INCBUS',
	'INCFARM',
	'FEDTAX',
	'STATETAX',
	'REGION'
]

# Define variable types
types = {
    'YEAR':       'int',
	'SERIAL':     'int',
	'ASECWT':     'float',
	'AGE': 	      'int',
	'SEX': 	      'int',
	'RACE':       'int',
	'FAMSIZE':    'float',
	'FAMUNIT':    'int',
	'HISPAN':     'float',
	'EMPSTAT':    'int',
	'LABFORCE':   'int',
	'AHRSWORKT':  'float',
	'WKSTAT':     'int',
	'EDUC':       'float',
	'WKSWORK1':   'float',
	'UHRSWORKLY': 'float',
	'WANTJOB':    'int',
	'INCTOT':     'float',
	'INCWAGE':    'float',
	'INCBUS':     'float',
	'INCFARM':    'float',
	'FEDTAX':     'float',
	'STATETAX':   'float',
	'REGION':     'int'
}

# Define a race encoding
race_map = {
    100: 1,      # White
	200: 2,      # Black/Negro
	300: 3,      # American Indian/Aleut/Eskimo
	650: 4,      # Asian or Pacific Islander
	651: 4,      # Asian only
	652: 4,      # Hawaiian/Pacific Islander only
	700: np.nan, # Other (single) race, n.e.c.
	801: 12,     # White and Black
	802: 13,     # White and American Indian
	803: 14,     # White and Asian
	804: 14,     # White and Hawaiian/Pacific Islander
	805: 23,     # Black and American Indian
	806: 24,     # Black and Asian
	807: 24,     # Black and Hawaiian/Pacific Islander
	808: 34,     # American Indian and Asian
	809: 4,      # Asian and Hawaiian/Pacific Islander
	810: 123,    # White, Black and American Indian
	811: 124,    # White, Black and Asian
	812: 134,    # White, American Indian and Asian
	813: 14,     # White, Asian and Hawaiian/Pacific Islander
	814: 1234,   # White, Black, American Indian and Asian
	815: 34,     # American Indian and Hawaiian/Pacific Islander
	816: 124,    # White, Black and Hawaiian/Pacific Islander
	817: 134,    # White, American Indian and Hawaiian/Pacific Islander
	818: 234,    # Black, American Indian and Asian
	819: 134,    # White, American Indian, Asian and Hawaiian/Pacific Islander
	820: np.nan, # Two or three races (unspecified)
	830: np.nan, # Four or five races (unspecified)
	999: np.nan  # Unknown
}

# Define a latin encoding
latin_map = {
    0:   0,	     # Not Hispanic
	100: 1,      # Mexican
	102: 1,      # Mexican American
	103: 1,      # Mexicano/Mexicana
	104: 1,      # Chicano/Chicana
	108: 1,      # Mexican
	109: 1,      # Mexicano/Chicano
	200: 1,      # Puerto Rican
	300: 1,      # Cuban
	400: 1,      # Dominican
	500: 1,      # Salvadoran
	600: 1,      # Other Hispanic
	610: 1,      # Central or South American
	611: 1,      # Central American (excluding Salvadoran)
	612: 1,      # South American
	901: np.nan, # Do not know
	902: np.nan  # Unknown
} 

# Define an education encoding
education_map = {
    0:  1, # Not in universe or no schooling
	1:  1, # Grades 1 to 4
	2:  1, # Grades 5 to 6
	3:  1, # Grades 7 to 8
	4:  1, # Grade 9
	5:  1, # Grade 10
	6:  1, # Grade 11
	7:  1, # Grade 12
	8:  2, # 1 year of college
	9:  2, # 2 years of college
	10: 2, # 3 years of college
	11: 3, # 4 years of college
	12: 3  # 5+ years of college
}

# Load the ASEC CPS data
cps = pd.read_csv(os.path.join(cps_r_data, 'cps.csv'), header=0, usecols=columns, dtype=types)

# Define a list of years
years = cps.YEAR.unique().tolist()

# Load the NIPA personal earnings and income data from the BEA
meta = bea.data('nipa', tablename='t20100', frequency='a', year=years).metadata
earnings_series = list(meta.loc[meta.LineDescription.isin(['Wages and salaries', "Proprietors' earnings with inventory valuation and capital consumption adjustments"]), 'SeriesCode'])
income_series = meta.loc[meta.LineDescription == 'Equals: Disposable personal income', 'SeriesCode'].values[0]
earnings = 1e6 * bea.data('nipa', tablename='t20100', frequency='a', year=years).data.loc[:, earnings_series].sum(axis=1).values.squeeze()
income = 1e6 * bea.data('nipa', tablename='t20100', frequency='a', year=years).data[income_series]
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=years).data.B230RC.values.squeeze()
deflator = 1e2 / bea.data("nipa", tablename="t10104", frequency="a", year=years).data.DPCERG.values.squeeze()
deflator = deflator / deflator[years.index(2012)]
earnings_per_capita = deflator * earnings / population
income_per_capita = deflator * income / population

# Recode the race variable
cps.loc[:, 'RACE'] = cps.RACE.map(race_map)

# Split the White and Black observations in each category
second = cps.loc[cps.RACE == 12, :].copy(deep=True)
second.loc[:, 'RACE'] = 2
second.loc[:, 'ASECWT'] = second.ASECWT / 2
cps.loc[cps.RACE == 12, 'ASECWT'] = cps.ASECWT / 2
cps.loc[cps.RACE == 12, 'RACE'] = 1
cps = pd.concat([cps, second], ignore_index=True)

# Split the White and Native American observations in each category
second = cps.loc[cps.RACE == 13, :].copy(deep=True)
second.loc[:, 'RACE'] = 3
second.loc[:, 'ASECWT'] = second.ASECWT / 2
cps.loc[cps.RACE == 13, 'ASECWT'] = cps.ASECWT / 2
cps.loc[cps.RACE == 13, 'RACE'] = 1
cps = pd.concat([cps, second], ignore_index=True)

# Split the White and Asian or Pacific Islander observations in each category
second = cps.loc[cps.RACE == 14, :].copy(deep=True)
second.loc[:, 'RACE'] = 4
second.loc[:, 'ASECWT'] = second.ASECWT / 2
cps.loc[cps.RACE == 14, 'ASECWT'] = cps.ASECWT / 2
cps.loc[cps.RACE == 14, 'RACE'] = 1
cps = pd.concat([cps, second], ignore_index=True)

# Split the Black and Native American observations in each category
second = cps.loc[cps.RACE == 23, :].copy(deep=True)
second.loc[:, 'RACE'] = 3
second.loc[:, 'ASECWT'] = second.ASECWT / 2
cps.loc[cps.RACE == 23, 'ASECWT'] = cps.ASECWT / 2
cps.loc[cps.RACE == 23, 'RACE'] = 2
cps = pd.concat([cps, second], ignore_index=True)

# Split the Black and Asian or Pacific Islander observations in each category
second = cps.loc[cps.RACE == 24, :].copy(deep=True)
second.loc[:, 'RACE'] = 4
second.loc[:, 'ASECWT'] = second.ASECWT / 2
cps.loc[cps.RACE == 24, 'ASECWT'] = cps.ASECWT / 2
cps.loc[cps.RACE == 24, 'RACE'] = 2
cps = pd.concat([cps, second], ignore_index=True)

# Split the Native American and Asian or Pacific Islander observations in each category
second = cps.loc[cps.RACE == 34, :].copy(deep=True)
second.loc[:, 'RACE'] = 4
second.loc[:, 'ASECWT'] = second.ASECWT / 2
cps.loc[cps.RACE == 34, 'ASECWT'] = cps.ASECWT / 2
cps.loc[cps.RACE == 34, 'RACE'] = 3
cps = pd.concat([cps, second], ignore_index=True)

# Split the White, Black and Native American observations in each category
second = cps.loc[cps.RACE == 123, :].copy(deep=True)
second.loc[:, 'RACE'] = 2
second.loc[:, 'ASECWT'] = second.ASECWT / 3
third = cps.loc[cps.RACE == 123, :].copy(deep=True)
third.loc[:, 'RACE'] = 3
third.loc[:, 'ASECWT'] = third.ASECWT / 3
cps.loc[cps.RACE == 123, 'ASECWT'] = cps.ASECWT / 3
cps.loc[cps.RACE == 123, 'RACE'] = 1
cps = pd.concat([cps, second, third], ignore_index=True)

# Split the White, Black and Asian or Pacific Islander observations in each category
second = cps.loc[cps.RACE == 124, :].copy(deep=True)
second.loc[:, 'RACE'] = 2
second.loc[:, 'ASECWT'] = second.ASECWT / 3
third = cps.loc[cps.RACE == 124, :].copy(deep=True)
third.loc[:, 'RACE'] = 4
third.loc[:, 'ASECWT'] = third.ASECWT / 3
cps.loc[cps.RACE == 124, 'ASECWT'] = cps.ASECWT / 3
cps.loc[cps.RACE == 124, 'RACE'] = 1
cps = pd.concat([cps, second, third], ignore_index=True)

# Split the White, Native American and Asian or Pacific Islander observations in each category
second = cps.loc[cps.RACE == 134, :].copy(deep=True)
second.loc[:, 'RACE'] = 3
second.loc[:, 'ASECWT'] = second.ASECWT / 3
third = cps.loc[cps.RACE == 134, :].copy(deep=True)
third.loc[:, 'RACE'] = 4
third.loc[:, 'ASECWT'] = third.ASECWT / 3
cps.loc[cps.RACE == 134, 'ASECWT'] = cps.ASECWT / 3
cps.loc[cps.RACE == 134, 'RACE'] = 1
cps = pd.concat([cps, second, third], ignore_index=True)

# Split the Black, Native American and Asian or Pacific Islander observations in each category
second = cps.loc[cps.RACE == 234, :].copy(deep=True)
second.loc[:, 'RACE'] = 3
second.loc[:, 'ASECWT'] = second.ASECWT / 3
third = cps.loc[cps.RACE == 234, :].copy(deep=True)
third.loc[:, 'RACE'] = 4
third.loc[:, 'ASECWT'] = third.ASECWT / 3
cps.loc[cps.RACE == 234, 'ASECWT'] = cps.ASECWT / 3
cps.loc[cps.RACE == 234, 'RACE'] = 2
cps = pd.concat([cps, second, third], ignore_index=True)

# Split the White, Black, Native American and Asian or Pacific Islander observations in each category
second = cps.loc[cps.RACE == 1234, :].copy(deep=True)
second.loc[:, 'RACE'] = 2
second.loc[:, 'ASECWT'] = second.ASECWT / 4
third = cps.loc[cps.RACE == 1234, :].copy(deep=True)
third.loc[:, 'RACE'] = 3
third.loc[:, 'ASECWT'] = third.ASECWT / 4
fourth = cps.loc[cps.RACE == 1234, :].copy(deep=True)
fourth.loc[:, 'RACE'] = 4
fourth.loc[:, 'ASECWT'] = fourth.ASECWT / 4
cps.loc[cps.RACE == 1234, 'ASECWT'] = cps.ASECWT / 4
cps.loc[cps.RACE == 1234, 'RACE'] = 1
cps = pd.concat([cps, second, third, fourth], ignore_index=True)

# Drop the unknown race observations
cps = cps.loc[cps.RACE.notna(), :]

# Recode the latin origin variable
cps.loc[:, 'HISPAN'] = cps.HISPAN.map(latin_map)

# Recode the education variable
cps.loc[cps.EDUC == 999, 'EDUC'] = np.nan
cps.loc[:, 'EDUC'] = np.round_(cps.EDUC / 10).map(education_map)

# Create a family identifier
cps.loc[:, 'SERIAL'] = cps.SERIAL.astype('str') + cps.FAMUNIT.astype('str')
cps = cps.drop('FAMUNIT', axis=1)

# Recode the earnings variables
cps.loc[cps.INCWAGE == 99999999, 'INCWAGE'] = 0
cps.loc[cps.INCWAGE == 99999998, 'INCWAGE'] = np.nan
cps.loc[cps.INCBUS == 99999999, 'INCBUS'] = 0
cps.loc[cps.INCBUS == 99999998, 'INCBUS'] = np.nan
cps.loc[cps.INCFARM == 99999999, 'INCFARM'] = 0
cps.loc[cps.INCFARM == 99999998, 'INCFARM'] = np.nan
cps.loc[cps.INCTOT == 999999999, 'INCTOT'] = 0
cps.loc[cps.FEDTAX == 99999999, 'FEDTAX'] = 0
cps.loc[cps.STATETAX == 9999999, 'STATETAX'] = 0

# Compute total earnings and income
cps.loc[:, 'earnings'] = cps.INCWAGE.fillna(value=0) + cps.INCBUS.fillna(value=0) + cps.INCFARM.fillna(value=0)
cps.loc[:, 'income'] = cps.INCTOT.fillna(value=0) - cps.FEDTAX.fillna(value=0) - cps.STATETAX.fillna(value=0)

# Compute family earnings and income as the sum of the family members' earnings and income
cps = pd.merge(cps, cps.groupby(['YEAR', 'SERIAL'], as_index=False).agg({'earnings': 'sum', 'income': 'sum'}), how='left')
cps = cps.drop(['INCWAGE', 'INCBUS', 'INCFARM', 'INCTOT', 'FEDTAX', 'STATETAX'], axis=1)

# Divide family earnings and income evenly among family members
cps.loc[:, 'earnings'] = cps.earnings / cps.FAMSIZE
cps.loc[:, 'income'] = cps.income / cps.FAMSIZE
cps = cps.drop('FAMSIZE', axis=1)

# Rescale earnings such that it aggregates to the NIPA personal earnings
cps = pd.merge(cps, cps.groupby('YEAR', as_index=False).apply(lambda x: pd.Series({'earnings_average': np.average(x.earnings, weights=x.ASECWT)})), how='left')
cps = pd.merge(cps, pd.DataFrame({'YEAR': years, 'earnings_per_capita': earnings_per_capita}), how='left')
cps.loc[:, 'earnings'] = cps.earnings_per_capita + cps.earnings_per_capita * (cps.earnings - cps.earnings_average) / cps.earnings_average
cps = cps.drop(['earnings_average', 'earnings_per_capita'], axis=1)

# Rescale income such that it aggregates to the NIPA personal income
cps = pd.merge(cps, cps.groupby('YEAR', as_index=False).apply(lambda x: pd.Series({'income_average': np.average(x.income, weights=x.ASECWT)})), how='left')
cps = pd.merge(cps, pd.DataFrame({'YEAR': years, 'income_per_capita': income_per_capita}), how='left')
cps.loc[:, 'income'] = cps.income_per_capita + cps.income_per_capita * (cps.income - cps.income_average) / cps.income_average
cps = cps.drop(['income_average', 'income_per_capita'], axis=1)
cps.loc[cps.YEAR <= 1991, 'income'] = np.nan

# Recode the hours worked per week variables
cps.loc[cps.UHRSWORKLY == 999, 'UHRSWORKLY'] = 0
cps.loc[cps.AHRSWORKT == 999, 'AHRSWORKT'] = 0

# Create the hours worked per year variable
cps.loc[:, 'hours'] = cps.UHRSWORKLY * cps.WKSWORK1
cps = cps.drop('UHRSWORKLY', axis=1)

# Split hours worked per year evenly among family members between 25 and 64
cps = pd.merge(cps, cps.loc[(cps.AGE >= 25) & (cps.AGE <= 64), :].groupby(['YEAR', 'SERIAL'], as_index=False).agg({'hours': 'mean', 'AHRSWORKT': 'mean'}).rename(columns={'hours': 'split', 'AHRSWORKT': 'AHRSWORKT_split'}), how='left')
cps.loc[(cps.AGE >= 25) & (cps.AGE < 65), 'hours'] = cps.split
cps.loc[(cps.AGE >= 25) & (cps.AGE < 65), 'AHRSWORKT'] = cps.AHRSWORKT_split
cps = cps.drop(['SERIAL', 'split', 'AHRSWORKT_split'], axis=1)

# Create the leisure variables
leap_years = cps.YEAR.apply(calendar.isleap)
cps.loc[leap_years == True, 'leisure'] = (16 * 366 - cps.hours) / (16 * 366)
cps.loc[leap_years == False, 'leisure'] = (16 * 365 - cps.hours) / (16 * 365)
cps.loc[:, 'weekly_leisure'] = (16 * 7 - cps.AHRSWORKT) / (16 * 7)
cps = cps.drop(['hours', 'AHRSWORKT'], axis=1)

# Identify the employment status of each individual
cps.loc[cps.EMPSTAT.isin([10, 12]) & ~cps.WKSTAT.isin([20, 21]), 'status'] = 'employed'
cps.loc[cps.EMPSTAT.isin([20, 21, 22]) | (cps.WANTJOB == 2) | cps.WKSTAT.isin([20, 21]), 'status'] = 'unemployed'
cps = cps.drop(['EMPSTAT', 'WKSTAT', 'WANTJOB'], axis=1)

# Recode the labor force variable
cps.loc[:, 'LABFORCE'] = cps.LABFORCE.map({0: np.nan, 1: 0, 2: 1})

# Calculate average leisure for the employed
cps = pd.merge(cps, cps.loc[cps.status == 'employed', :].groupby(['YEAR', 'RACE', 'HISPAN'], as_index=False).apply(lambda x: pd.Series({'employed_leisure': np.average(x.weekly_leisure, weights=x.ASECWT)})), how='left')

# Calculate the leisure difference for the unemployed
cps.loc[cps.status == 'unemployed', 'Δ_leisure'] = cps.weekly_leisure - cps.employed_leisure
cps.loc[(cps.Δ_leisure < 0) | cps.Δ_leisure.isna(), 'Δ_leisure'] = 0
cps = cps.drop('employed_leisure', axis=1)

# Adjust the leisure variable for unemployment
cps.loc[:, 'leisure_half'] = cps.leisure - 0.5 * cps.Δ_leisure
cps.loc[:, 'leisure'] = cps.leisure - cps.Δ_leisure

# Only keep the first digit of the REGION variable and recode it
cps.loc[:, 'REGION'] = cps.REGION.apply(lambda x: int(str(x)[0]))
cps.loc[:, 'REGION'] = cps.REGION.map({1: 1, 2: 1, 3: 2, 4: 1})

# Rename variables
cps = cps.rename(columns={
    'YEAR':     'year',
	'SEX':      'gender',
	'RACE':     'race',
	'HISPAN':   'latin',
	'EDUC':     'education',
	'AGE':      'age',
	'LABFORCE': 'laborforce',
	'WKSWORK1': 'weeks_worked',
	'REGION':   'region',
	'ASECWT':   'weight'
})

# Define the variable types
cps = cps.astype({
    'year':           'int',
	'gender':         'int',
	'race':           'int',
	'latin':          'float32',
	'education':      'float32',
	'age':            'int',
	'leisure':        'float',
	'leisure_half':   'float',
	'weeks_worked':   'float',
	'weekly_leisure': 'float',
	'Δ_leisure':      'float',
	'earnings':       'float',
	'income':         'float',
	'status':         'str',
	'laborforce':     'float',
	'region':         'int',
	'weight':         'float'
})

# Sort the data frame
cps = cps.sort_values(by='year').reset_index(drop=True)

# Save the data
cps.to_csv(os.path.join(cps_f_data, 'cps.csv'), index=False)