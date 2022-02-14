# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import statsmodels.api as sm
from dotenv import load_dotenv
load_dotenv()
import os
import calendar

# Import functions and directories
from functions import *
from directories import *

# Start the BEA client
bea = beapy.BEA(key=os.getenv('bea_api_key'))

# Define a list of years
years = list(range(1940, 1990 + 1, 10)) + list(range(2000, 2020 + 1))

# Define variable columns
columns = ['YEAR',
		   'SERIAL',
           'PERWT',
		   'FOODSTMP',
		   'SLWT',
		   'FAMUNIT',
		   'FAMSIZE',
           'SEX',
           'AGE',
           'RACE',
           'RACWHT',
           'RACBLK',
           'RACAMIND',
           'RACASIAN',
           'RACPACIS',
           'HISPAN',
		   'HINSCAID',
		   'HINSCARE',
           'EDUC',
           'WKSWORK1',
		   'WKSWORK2',
           'HRSWORK1',
		   'HRSWORK2',
		   'UHRSWORK',
           'INCWAGE',
		   'INCBUSFM',
		   'INCBUS',
		   'INCBUS00',
		   'INCFARM']

# Define variable types
types = {'YEAR':     'int',
	     'SERIAL':   'uint64',
         'PERWT':    'float',
		 'FOODSTMP': 'float',
		 'SLWT':     'float',
		 'FAMUNIT':  'int',
         'FAMSIZE':  'float',
         'SEX':      'int',
         'AGE':      'int',
         'RACE':     'int',
         'RACWHT':   'float',
         'RACBLK':   'float',
         'RACAMIND': 'float',
         'RACASIAN': 'float',
         'RACPACIS': 'float',
         'HISPAN':   'int',
		 'HINSCAID': 'float',
		 'HINSCARE': 'float',
         'EDUC':     'int',
         'WKSWORK1': 'float',
		 'WKSWORK2': 'float',
         'HRSWORK1': 'float',
		 'HRSWORK2': 'float',
		 'UHRSWORK': 'float',
         'INCWAGE':  'float',
		 'INCBUSFM': 'float',
		 'INCBUS':   'float',
		 'INCBUS00': 'float',
		 'INCFARM':  'float'}

# Define a race mapping
race_map = {1: 1,      # White
            2: 2,      # Black
            3: 3,      # Native American
            4: 4,      # Chinese
            5: 4,      # Japanese
            6: 4,      # Other Asian or Pacific Islander
            7: np.nan, # Other
            8: 5,      # Two major races
            9: 5}      # Three or more major races

# Define a latin origin mapping
latin_map = {0: 0,  # Not latin
             1: 1,  # Mexican
             2: 1,  # Puerto Rican
             3: 1,  # Cuban
             4: 1,  # Other
             9: -1} # Unknown

# Define an education mapping
education_map = {0:  1, # Not in universe or no schooling
                 1:  1, # Grades 1 to 4
                 2:  1, # Grades 5 to 8
                 3:  1, # Grade 9
                 4:  1, # Grade 10
                 5:  1, # Grade 11
                 6:  1, # Grade 12
                 7:  2, # 1 year of college
                 8:  2, # 2 years of college
                 9:  2, # 3 years of college
                 10: 3, # 4 years of college
                 11: 3} # 5+ years of college

# Define a weeks worked mapping
weeks_map = {0: 0,
             1: 7,
             2: 20,
             3: 33,
             4: 43.5,
             5: 48.5,
             6: 51}

# Define a hours worked mapping
hours_map = {0: 0,
             1: 7.5,
             2: 22,
             3: 32,
             4: 37,
             5: 40,
             6: 44.5,
			 7: 54,
			 8: 79.5}

# Load the imputation models
earnings_model = sm.load(os.path.join(cex_f_data, 'earnings.pickle'))
salary_model = sm.load(os.path.join(cex_f_data, 'salary.pickle'))

# Load the NIPA personal earnings and PCE data from the BEA
meta = bea.data('nipa', tablename='t20100', frequency='a', year=years).metadata
income_series = list(meta.loc[meta.LineDescription.isin(['Wages and salaries', "Proprietors' income with inventory valuation and capital consumption adjustments"]), 'SeriesCode'])
earnings = 1e6 * bea.data('nipa', tablename='t20100', frequency='a', year=years).data.loc[:, income_series].sum(axis=1).values.squeeze()
pce = 1e6 * (bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DPCERC.values.squeeze() - bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DINSRC.values.squeeze())
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=years).data.B230RC.values.squeeze()
deflator = 1e2 / bea.data('nipa', tablename='t10104', frequency='a', year=years).data.DPCERG.values.squeeze()
earnings = deflator * earnings / population
pce = deflator * pce / population

# Store the above four series in a data frame
BEA = pd.DataFrame(data={'YEAR':             years,
						 'earnings_nipa':    earnings,
						 'consumption_nipa': pce})

# Define a function to read the data by year
def year_chunk(file, chunksize=1e6):
    iterator = pd.read_csv(file, compression='gzip', header=0, usecols=columns, dtype=types, iterator=True, chunksize=chunksize)
    chunk = pd.DataFrame()
    for df in iterator:
        unique_years = np.sort(df.YEAR.unique())
        if len(unique_years) == 1:
            chunk = chunk.append(df, ignore_index=True)
        else:
            chunk = chunk.append(df.loc[df.YEAR == unique_years[0], :], ignore_index=True)
            yield chunk
            chunk = pd.DataFrame()
            chunk = chunk.append(df.loc[df.YEAR == unique_years[1], :], ignore_index=True)
    yield chunk
chunks = year_chunk(os.path.join(acs_r_data, 'acs.csv.gz'), chunksize=1e6)

# Initialize a data frame
acs = pd.DataFrame()

# Load the U.S. censuses and ACS data
for chunk in chunks:
	# Recode the race and latin origin variables
	chunk.loc[:, 'RACE'] = chunk.RACE.map(race_map)
	chunk.loc[:, 'HISPAN'] = chunk.HISPAN.map(latin_map)
	chunk.loc[:, 'RACWHT'] = chunk.RACWHT.map({1: np.nan, 2: 1})
	chunk.loc[:, 'RACBLK'] = chunk.RACBLK.map({1: np.nan, 2: 1})
	chunk.loc[:, 'RACAMIND'] = chunk.RACAMIND.map({1: np.nan, 2: 1})
	chunk.loc[:, 'RACASIAN'] = chunk.RACASIAN.map({1: np.nan, 2: 1})
	chunk.loc[:, 'RACPACIS'] = chunk.RACPACIS.map({1: np.nan, 2: 1})

	# Recode the White observations
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.isna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'RACE'] = 1

	# Recode the Black observations
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'RACE'] = 2

	# Recode the Native American observations
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'RACE'] = 3

	# Recode the Asian or Pacific Islander observations
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.isna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 4

	# Split the White and Black observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), :].copy(deep=True)
	second.loc[:, 'RACE'] = 2
	second.loc[:, 'PERWT'] = second.PERWT / 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'RACE'] = 1
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'PERWT'] = chunk.PERWT / 2
	chunk = chunk.append(second, ignore_index=True)

	# Split the White and Native American observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), :].copy(deep=True)
	second.loc[:, 'RACE'] = 3
	second.loc[:, 'PERWT'] = second.PERWT / 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'RACE'] = 1
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'PERWT'] = chunk.PERWT / 2
	chunk = chunk.append(second, ignore_index=True)

	# Split the White and Asian or Pacific Islander observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	second.loc[:, 'RACE'] = 4
	second.loc[:, 'PERWT'] = second.PERWT / 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 1
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'PERWT'] = chunk.PERWT / 2
	chunk = chunk.append(second, ignore_index=True)

	# Split the Black and Native American observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), :].copy(deep=True)
	second.loc[:, 'RACE'] = 3
	second.loc[:, 'PERWT'] = second.PERWT / 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'RACE'] = 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'PERWT'] = chunk.PERWT / 2
	chunk = chunk.append(second, ignore_index=True)

	# Split the Black and Asian or Pacific Islander observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	second.loc[:, 'RACE'] = 4
	second.loc[:, 'PERWT'] = second.PERWT / 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'PERWT'] = chunk.PERWT / 2
	chunk = chunk.append(second, ignore_index=True)

	# Split the Native American and Asian or Pacific Islander observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	second.loc[:, 'RACE'] = 4
	second.loc[:, 'PERWT'] = second.PERWT / 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 3
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'PERWT'] = chunk.PERWT / 2
	chunk = chunk.append(second, ignore_index=True)

	# Split the White, Black and Native American observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), :].copy(deep=True)
	second.loc[:, 'RACE'] = 2
	second.loc[:, 'PERWT'] = second.PERWT / 3
	third = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), :].copy(deep=True)
	third.loc[:, 'RACE'] = 3
	third.loc[:, 'PERWT'] = third.PERWT / 3
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'RACE'] = 1
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & chunk.RACASIAN.isna() & chunk.RACPACIS.isna(), 'PERWT'] = chunk.PERWT / 3
	chunk = chunk.append(second.append(third, ignore_index=True), ignore_index=True)

	# Split the White, Black and Asian or Pacific Islander observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	second.loc[:, 'RACE'] = 2
	second.loc[:, 'PERWT'] = second.PERWT / 3
	third = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	third.loc[:, 'RACE'] = 4
	third.loc[:, 'PERWT'] = third.PERWT / 3
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 1
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.isna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'PERWT'] = chunk.PERWT / 3
	chunk = chunk.append(second.append(third, ignore_index=True), ignore_index=True)

	# Split the White, Native American and Asian or Pacific Islander observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	second.loc[:, 'RACE'] = 3
	second.loc[:, 'PERWT'] = second.PERWT / 3
	third = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	third.loc[:, 'RACE'] = 4
	third.loc[:, 'PERWT'] = third.PERWT / 3
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 1
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.isna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'PERWT'] = chunk.PERWT / 3
	chunk = chunk.append(second.append(third, ignore_index=True), ignore_index=True)

	# Split the Black, Native American and Asian or Pacific Islander observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	second.loc[:, 'RACE'] = 3
	second.loc[:, 'PERWT'] = second.PERWT / 3
	third = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	third.loc[:, 'RACE'] = 4
	third.loc[:, 'PERWT'] = third.PERWT / 3
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 2
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.isna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'PERWT'] = chunk.PERWT / 3
	chunk = chunk.append(second.append(third, ignore_index=True), ignore_index=True)

	# Split the White, Black, Native American and Asian or Pacific Islander observations in each category
	second = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	second.loc[:, 'RACE'] = 2
	second.loc[:, 'PERWT'] = second.PERWT / 4
	third = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	third.loc[:, 'RACE'] = 3
	third.loc[:, 'PERWT'] = third.PERWT / 4
	fourth = chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), :].copy(deep=True)
	fourth.loc[:, 'RACE'] = 4
	fourth.loc[:, 'PERWT'] = fourth.PERWT / 4
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'RACE'] = 1
	chunk.loc[(chunk.RACE == 5) & chunk.RACWHT.notna() & chunk.RACBLK.notna() & chunk.RACAMIND.notna() & (chunk.RACASIAN.notna() | chunk.RACPACIS.notna()), 'PERWT'] = chunk.PERWT / 4
	chunk = chunk.append(second.append(third.append(fourth, ignore_index=True), ignore_index=True), ignore_index=True)

	# Drop the unknown race observations and multi-race variables
	chunk = chunk.loc[chunk.RACE.notna() & (chunk.RACE != 5), :]
	chunk = chunk.drop(['RACWHT', 'RACBLK', 'RACAMIND', 'RACASIAN', 'RACPACIS'], axis=1)

	# Create a food stamps, medicaid and medicare identifiers
	chunk.loc[:, 'FOODSTMP'] = chunk.FOODSTMP.map({0: 0, 1: 0, 2: 1, np.nan: np.nan})
	chunk.loc[:, 'HINSCAID'] = chunk.HINSCAID.map({1: 0, 2: 1, np.nan: np.nan})
	chunk.loc[:, 'HINSCARE'] = chunk.HINSCARE.map({1: 0, 2: 1, np.nan: np.nan})
	chunk = chunk.rename(columns={'FOODSTMP': 'foodstamps', 'HINSCAID': 'medicaid', 'HINSCARE': 'medicare'})

	# Recode the education variable
	chunk.loc[:, 'EDUC'] = chunk.EDUC.map(education_map)
	chunk.loc[chunk.EDUC.isna() | (chunk.AGE < 30), 'EDUC'] = 4

	# Recode the earnings variables
	chunk.loc[:, 'missing_earnings'] = ((chunk.INCWAGE.isna() | (chunk.INCWAGE == 999998)) &
                                    	(chunk.INCBUSFM.isna() | (chunk.INCBUSFM == 99999)) &
                                    	(chunk.INCBUS.isna() | (chunk.INCBUS == 999999)) &
                                    	(chunk.INCBUS00.isna() | (chunk.INCBUS00 == 999999)) &
                                    	(chunk.INCFARM.isna() | (chunk.INCFARM == 999999)))
	chunk.loc[chunk.INCWAGE.isna() | (chunk.INCWAGE == 999999) | (chunk.INCWAGE == 999998), 'INCWAGE'] = 0
	chunk.loc[chunk.INCBUSFM.isna() | (chunk.INCBUSFM == 99999), 'INCBUSFM'] = 0
	chunk.loc[chunk.INCBUS.isna() | (chunk.INCBUS == 999999), 'INCBUS'] = 0
	chunk.loc[chunk.INCBUS00.isna() | (chunk.INCBUS00 == 999999), 'INCBUS00'] = 0
	chunk.loc[chunk.INCFARM.isna() | (chunk.INCFARM == 999999), 'INCFARM'] = 0
	chunk.loc[:, 'earnings'] = chunk.loc[:, [col for col in chunk.columns if col.startswith('INC')]].sum(axis=1).values
	chunk = chunk.drop([col for col in chunk.columns if col.startswith('INC')], axis=1)

	# Create a family identifier
	chunk.loc[:, 'SERIAL'] = chunk.SERIAL.astype('str') + chunk.FAMUNIT.astype('str')
	chunk = chunk.drop('FAMUNIT', axis=1)

	# Calculate family earnings as the sum of the family members' personal earnings
	chunk = pd.merge(chunk, chunk.groupby('SERIAL', as_index=False).agg({'earnings': 'sum'}).rename(columns={'earnings': 'family_earnings'}), how='left')

	# Divide family earnings evenly across family members
	chunk.loc[:, 'earnings'] = chunk.family_earnings / chunk.FAMSIZE
	chunk = chunk.drop('family_earnings', axis=1)

	# Create race binary variables
	chunk = pd.concat([chunk, pd.get_dummies(chunk.RACE.astype('int'), prefix='race')], axis=1)

	# Create education binary variables
	chunk = pd.concat([chunk, pd.get_dummies(chunk.EDUC.astype('int'), prefix='education')], axis=1)

	# Calculate the percentage deviation of each imputation variable from their average
	chunk.loc[:, 'earnings_Δ'] = (chunk.earnings - np.average(chunk.earnings, weights=chunk.PERWT)) / np.average(chunk.earnings, weights=chunk.PERWT)
	chunk.loc[:, 'race_1_Δ'] = (chunk.race_1 - np.average(chunk.race_1, weights=chunk.PERWT)) / np.average(chunk.race_1, weights=chunk.PERWT)
	chunk.loc[:, 'race_2_Δ'] = (chunk.race_2 - np.average(chunk.race_2, weights=chunk.PERWT)) / np.average(chunk.race_2, weights=chunk.PERWT)
	chunk.loc[:, 'race_3_Δ'] = (chunk.race_3 - np.average(chunk.race_3, weights=chunk.PERWT)) / np.average(chunk.race_3, weights=chunk.PERWT)
	chunk.loc[:, 'race_4_Δ'] = (chunk.race_4 - np.average(chunk.race_4, weights=chunk.PERWT)) / np.average(chunk.race_4, weights=chunk.PERWT)
	chunk.loc[:, 'education_1_Δ'] = (chunk.education_1 - np.average(chunk.education_1, weights=chunk.PERWT)) / np.average(chunk.education_1, weights=chunk.PERWT)
	chunk.loc[:, 'education_2_Δ'] = (chunk.education_2 - np.average(chunk.education_2, weights=chunk.PERWT)) / np.average(chunk.education_2, weights=chunk.PERWT)
	chunk.loc[:, 'education_3_Δ'] = (chunk.education_3 - np.average(chunk.education_3, weights=chunk.PERWT)) / np.average(chunk.education_3, weights=chunk.PERWT)
	chunk.loc[:, 'education_4_Δ'] = (chunk.education_4 - np.average(chunk.education_4, weights=chunk.PERWT)) / np.average(chunk.education_4, weights=chunk.PERWT)
	chunk.loc[:, 'family_size_Δ'] = (chunk.FAMSIZE - np.average(chunk.FAMSIZE, weights=chunk.PERWT)) / np.average(chunk.FAMSIZE, weights=chunk.PERWT)
	chunk.loc[:, 'latin_Δ'] = (chunk.HISPAN - np.average(chunk.HISPAN, weights=chunk.PERWT)) / np.average(chunk.HISPAN, weights=chunk.PERWT)
	chunk.loc[:, 'gender_Δ'] = (chunk.SEX.map({1: 1, 2: 0}) - np.average(chunk.SEX.map({1: 1, 2: 0}), weights=chunk.PERWT)) / np.average(chunk.SEX.map({1: 1, 2: 0}), weights=chunk.PERWT)
	chunk.loc[:, 'age_Δ'] = (chunk.AGE - np.average(chunk.AGE, weights=chunk.PERWT)) / np.average(chunk.AGE, weights=chunk.PERWT)

	# Impute consumption
	if int(chunk.YEAR.unique()) == 1940:
		chunk.loc[:, 'consumption'] = salary_model.predict(chunk.loc[:, [col for col in chunk.columns if col.endswith('Δ')]])
	else:
		chunk.loc[:, 'consumption'] = earnings_model.predict(chunk.loc[:, [col for col in chunk.columns if col.endswith('Δ')]])
	chunk = chunk.drop([col for col in chunk.columns if col.endswith('Δ')], axis=1)

	# Merge with the BEA data
	chunk = pd.merge(chunk, BEA, how='left')

	# Re-scale personal earnings and consumption expenditures such that it aggregates to the NIPA values
	chunk.loc[:, 'earnings'] = chunk.earnings_nipa + chunk.earnings_nipa * (chunk.earnings - np.average(chunk.earnings, weights=chunk.PERWT)) / np.average(chunk.earnings, weights=chunk.PERWT)
	chunk.loc[chunk.missing_earnings == True, 'earnings'] = np.nan
	chunk.loc[:, 'consumption'] = chunk.consumption_nipa + chunk.consumption_nipa * chunk.consumption
	chunk = chunk.drop(['earnings_nipa', 'consumption_nipa'], axis=1)

	# Create the different definitions of hours worked per year variable
	chunk.loc[:, 'hours_1'] = chunk.UHRSWORK * chunk.WKSWORK1
	chunk.loc[:, 'hours_2'] = chunk.UHRSWORK * chunk.WKSWORK2.map(weeks_map)
	chunk.loc[:, 'hours_3'] = chunk.HRSWORK1 * chunk.WKSWORK1
	chunk.loc[:, 'hours_4'] = chunk.HRSWORK2.map(hours_map) * chunk.WKSWORK2.map(weeks_map)
	chunk = chunk.drop(['HRSWORK1', 'HRSWORK2', 'WKSWORK1', 'WKSWORK2', 'UHRSWORK'], axis=1)

	# Split hours worked per year evenly among family members between 25 and 64 for years 1940 and 1950
	chunk = pd.merge(chunk, chunk.loc[(chunk.AGE >= 25) & (chunk.AGE <= 64), :].groupby('SERIAL', as_index=False).agg(dict(zip(['hours_' + str(i + 1) for i in range(4)], ['mean'] * 4))).rename(columns=dict(zip(['hours_' + str(i + 1) for i in range(4)], ['split_' + str(i + 1) for i in range(4)]))), how='left')
	for i in range(4):
		chunk.loc[(chunk.AGE >= 25) & (chunk.AGE < 65), 'hours_' + str(i + 1)] = chunk.loc[:, 'split_' + str(i + 1)]
	chunk = chunk.drop(['split_1', 'split_2', 'split_3', 'split_4', 'SERIAL'], axis=1)

	# Create the leisure variables
	for i in range(4):
	    if bool(calendar.isleap(chunk.YEAR.unique())):
	        chunk.loc[:, 'leisure_' + str(i + 1)] = (16 * 366 - chunk.loc[:, 'hours_' + str(i + 1)]) / (16 * 366)
	    else:
	        chunk.loc[:, 'leisure_' + str(i + 1)] = (16 * 365 - chunk.loc[:, 'hours_' + str(i + 1)]) / (16 * 365)
	chunk = chunk.drop(['hours_1', 'hours_2', 'hours_3', 'hours_4'], axis=1)

	# Append the data frames for all chunks
	acs = acs.append(chunk, ignore_index=True)

# Calculate food stamps, medicaid and medicare usage by race in 2019
d_bea = dict(zip(['foodstamps', 'medicaid', 'medicare'], ['TRP600', 'W729RC', 'W824RC']))
deflator = 1e2 / bea.data('nipa', tablename='t10104', frequency='a', year=2019).data.DPCERG
welfare = pd.DataFrame({'race': acs.RACE.unique()})
for i in ['foodstamps', 'medicaid', 'medicare']:
    df = acs.loc[(acs.YEAR == 2019) & (acs.loc[:, i] == 1), :].groupby('RACE', as_index=False).agg({'PERWT': 'sum'})
    df.loc[:, 'PERWT'] = df.PERWT / df.PERWT.sum()
    df.loc[:, 'expenditures'] = 1e6 * deflator * bea.data('nipa', tablename='t31200', frequency='a', year=2019).data[d_bea[i]]
    df.loc[:, i] = df.loc[:, 'PERWT'] * df.loc[:, 'expenditures']
    df = df.drop(['PERWT', 'expenditures'], axis=1).rename(columns={'RACE': 'race'})
    welfare = pd.merge(welfare, df)
welfare.to_csv(os.path.join(acs_f_data, 'welfare.csv'), index=False)

# Create the leisure weight variable
acs.loc[acs.YEAR != 1950, 'leisure_weight'] = acs.PERWT
acs.loc[acs.YEAR == 1950, 'leisure_weight'] = acs.SLWT
acs = acs.drop('SLWT', axis=1)

# Calculate the ratio of the average of the first leisure variable to the average of the other leisure variables in 1980 and 1990
sample = ((acs.YEAR == 1980) | (acs.YEAR == 1990))
scale_2 = np.average(acs.loc[sample, 'leisure_1'], weights=acs.loc[sample, 'leisure_weight']) / np.average(acs.loc[sample, 'leisure_2'], weights=acs.loc[sample, 'leisure_weight'])
scale_3 = np.average(acs.loc[sample, 'leisure_1'], weights=acs.loc[sample, 'leisure_weight']) / np.average(acs.loc[sample, 'leisure_3'], weights=acs.loc[sample, 'leisure_weight'])
scale_4 = np.average(acs.loc[sample, 'leisure_1'], weights=acs.loc[sample, 'leisure_weight']) / np.average(acs.loc[sample, 'leisure_4'], weights=acs.loc[sample, 'leisure_weight'])

# Rescale the leisure variables
acs.loc[:, 'leisure_2'] = scale_2 * acs.leisure_2
acs.loc[:, 'leisure_3'] = scale_3 * acs.leisure_3
acs.loc[:, 'leisure_4'] = scale_4 * acs.leisure_4

# Create a unique leisure variable
acs.loc[acs.leisure_1.notna(), 'leisure'] = acs.leisure_1
acs.loc[acs.leisure_1.isna() & acs.leisure_2.notna(), 'leisure'] = acs.leisure_2
acs.loc[acs.leisure_1.isna() & acs.leisure_2.isna() & acs.leisure_3.notna(), 'leisure'] = acs.leisure_3
acs.loc[acs.leisure_1.isna() & acs.leisure_2.isna() & acs.leisure_3.isna() & acs.leisure_4.notna(), 'leisure'] = acs.leisure_4
acs = acs.drop(['leisure_1', 'leisure_2', 'leisure_3', 'leisure_4'], axis=1)

# Create a data frame with all levels of all variables
df = expand({'YEAR':   acs.YEAR.unique(),
			 'RACE':   acs.RACE.unique(),
			 'HISPAN': acs.HISPAN.unique(),
			 'SEX':    acs.SEX.unique(),
			 'EDUC':   acs.EDUC.unique(),
			 'AGE':    acs.AGE.unique()})

# Calculate weighted averages of each variable
def f(x):
    d = {}
    d['leisure'] = np.average(x.leisure, weights=x.leisure_weight)
    d['consumption'] = np.average(x.consumption, weights=x.PERWT)
    d['earnings'] = np.average(x.earnings, weights=x.PERWT)
    d['leisure_weight'] = np.sum(x.leisure_weight)
    d['PERWT'] = np.sum(x.PERWT)
    return pd.Series(d, index=[key for key, value in d.items()])
acs = acs.groupby(['YEAR', 'RACE', 'HISPAN', 'SEX', 'EDUC', 'AGE'], as_index=False).apply(f)

# Merge the data frames
acs = pd.merge(df, acs, how='left')
acs.loc[acs.leisure_weight.isna(), 'leisure_weight'] = 0
acs.loc[acs.PERWT.isna(), 'PERWT'] = 0

# Recode the education variable
acs.loc[acs.EDUC == 4, 'EDUC'] = np.nan

# Rename variables
acs = acs.rename(columns={'YEAR':   'year',
						  'SEX':    'gender',
						  'RACE':   'race',
						  'HISPAN': 'latin',
						  'EDUC':   'education',
						  'AGE':    'age',
						  'PERWT':  'weight'})

# Define the variable types
acs = acs.astype({'year':           'int',
				  'gender':         'int',
				  'race':           'int',
				  'latin':          'int',
				  'education':      'float',
				  'age':            'int',
				  'leisure':        'float',
				  'consumption':    'float',
				  'earnings':       'float',
				  'weight':         'float',
				  'leisure_weight': 'float'})

# Save the data
acs.to_csv(os.path.join(acs_f_data, 'acs.csv'), index=False)