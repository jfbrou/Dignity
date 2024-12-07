# Import libraries
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import ipumspy
import sys

# Import functions and directories
from functions import *
from directories import *

# Start the IPUMS API
ipums = ipumspy.IpumsApiClient(ipums_api_key)

# Define variable columns
variables = [
    'YEAR',
    'PERWEIGHT',
    'AGE',
    'SEX',
    'RACEA',
    'HISPETH',
    'EDUCREC1',
    'HEALTH',
    'LANY',
    'LADL',
    'LAIADL',
    'LANOWORK',
    'LAMTWRK'
]

# Define a race encoding
race_map = {
    100: 1,       # White
    200: 2,       # Black
    300: 3,       # Aleut, Alaskan Native, or Native American
    310: 3,       # Alaskan Native or Native American
    411: 4,       # Chinese
    412: 4,       # Filipino
    416: 4,       # Asian Indian
    433: 4,       # Other Asian or Pacific Islander (1997-1998)
    434: 4,       # Other Asian (1999 forward)
    560: np.nan,  # Other Race (1997-1998)
    570: np.nan,  # Other Race (1999-2002)
    580: np.nan,  # Primary Race not releasable
    600: np.nan,  # Multiple Race
    970: np.nan,  # Refused
    980: np.nan,  # Not ascertained
    990: np.nan   # Unknown
}

# Define a Latin origin encoding
latin_map = {
    10: 0, # Not Latin
    20: 1, # Mexican
    21: 1, # Mexicano
    23: 1, # Mexican-American
    30: 1, # Puerto Rican
    40: 1, # Cuban
    50: 1, # Dominican
    61: 1, # Central or South American
    62: 1, # Other Latin
    63: 1, # Other Spanish
    64: 1, # Latin non-specific type
    65: 1, # Latin type refused
    66: 1, # Latin type not ascertained
    67: 1, # Latin type unknown
    70: 1  # Multiple Latin origins
}

# Define an education encoding
education_map = {
     0: 1,       # NIU
     1: 1,       # Never attended or kindergarten only
     2: 1,       # Grade 1
     3: 1,       # Grade 2
     4: 1,       # Grade 3
     5: 1,       # Grade 4
     6: 1,       # Grade 5
     7: 1,       # Grade 6
     8: 1,       # Grade 7
     9: 1,       # Grade 8
    10: 1,       # Grade 9
    11: 1,       # Grade 10
    12: 1,       # Grade 11
    13: 1,       # Grade 12
    14: 2,       # 1 to 3 years of college
    15: 3,       # 4 years college
    16: 3,       # 5+ years of college
    97: np.nan,  # Refused
    98: np.nan,  # Not ascertained
    99: np.nan   # Unknown
}

# Define health and limitation encodings
health_map = {
    0: np.nan,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    7: np.nan,
    8: np.nan,
    9: np.nan
}

ADL_map = {
    0: 0,
    1: 0,
    2: 1,
    7: np.nan,
    8: np.nan,
    9: np.nan
}

IADL_map = {
    0: 0,
    1: 0,
    2: 1,
    7: np.nan,
    8: np.nan,
    9: np.nan
}

unable_work_map = {
    0: 0,
    1: 0,
    2: 1,
    7: np.nan,
    8: np.nan,
    9: np.nan
}

limited_work_map = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    7: np.nan,
    8: np.nan,
    9: np.nan
}

limited_any_map = {
    10: 1,
    20: 0,
    21: 0,
    22: np.nan
}

# Submit and download the IPUMS NHIS extract
if not any([file.endswith('.csv.gz') for file in os.listdir(nhis_r_data)]):
	samples = ['ih' + str(year) for year in range(1997, 2022 + 1, 1)]
	extract = ipumspy.MicrodataExtract(samples=samples, variables=variables, collection="nhis", data_format="csv")
	ipums.submit_extract(extract)
	ipums.extract_status(extract)
	ipums.wait_for_extract(extract)
	print(f"{extract.collection} number {extract.extract_id} is complete!")
	ipums.download_extract(extract, download_dir=nhis_r_data)

# Load the NHIS data
file_name = [file for file in os.listdir(nhis_r_data) if file.endswith('.csv.gz')][0]
nhis = pd.read_csv(os.path.join(nhis_r_data, file_name), header=0, usecols=variables, compression='gzip')

# Drop observations with not sampling weights
nhis = nhis.dropna(subset=['PERWEIGHT'])

# Recode the race and latin origin variables
nhis.loc[:, 'RACEA'] = nhis.RACEA.map(race_map)
nhis.loc[:, 'HISPETH'] = nhis.HISPETH.map(latin_map)
nhis = nhis.loc[nhis.RACEA.notna(), :]

# Recode the education variable
nhis.loc[:, 'EDUCREC1'] = nhis.EDUCREC1.map(education_map)

# Recode the health and limitation variables
nhis.loc[:, 'HEALTH'] = nhis.HEALTH.map(health_map)
nhis.loc[:, 'LADL'] = nhis.LADL.map(ADL_map)
nhis.loc[:, 'LAIADL'] = nhis.LAIADL.map(IADL_map)
nhis.loc[:, 'LANOWORK'] = nhis.LANOWORK.map(unable_work_map)
nhis.loc[:, 'LAMTWRK'] = nhis.LAMTWRK.map(limited_work_map)
nhis.loc[:, 'LANY'] = nhis.LANY.map(limited_any_map)
nhis.loc[:, 'LNONE'] = (((nhis.LANY == 0) | nhis.LANY.isna()) & ((nhis.LAMTWRK == 0) | nhis.LAMTWRK.isna())
                                                              & ((nhis.LANOWORK == 0) | nhis.LANOWORK.isna())
                                                              & ((nhis.LAIADL == 0) | nhis.LAIADL.isna())
                                                              & ((nhis.LADL == 0) | nhis.LADL.isna()))
nhis.loc[:, 'LANY'] = ((nhis.LANY == 1) & ((nhis.LAMTWRK == 0) | nhis.LAMTWRK.isna())
                                        & ((nhis.LANOWORK == 0) | nhis.LANOWORK.isna())
                                        & ((nhis.LAIADL == 0) | nhis.LAIADL.isna())
                                        & ((nhis.LADL == 0) | nhis.LADL.isna()))
nhis.loc[:, 'LAMTWRK'] = ((nhis.LAMTWRK == 1) & ((nhis.LANOWORK == 0) | nhis.LANOWORK.isna())
                                              & ((nhis.LAIADL == 0) | nhis.LAIADL.isna())
                                              & ((nhis.LADL == 0) | nhis.LADL.isna()))
nhis.loc[:, 'LANOWORK'] = ((nhis.LANOWORK == 1) & ((nhis.LAIADL == 0) | nhis.LAIADL.isna())
                                                & ((nhis.LADL == 0) | nhis.LADL.isna()))
nhis.loc[:, 'LAIADL'] = ((nhis.LAIADL == 1) & ((nhis.LADL == 0) | nhis.LADL.isna()))
nhis.loc[:, 'LADL'] = (nhis.LADL == 1)

# Calculate the single attribute scores as in Erickson, Wilson and Shannon (1995)
nhis.loc[:, 'HEALTH'] = nhis.HEALTH.map({1: 1, 2: 0.85, 3: 0.7, 4: 0.3, 5: 0})
nhis.loc[nhis.LNONE, 'limitation'] = 1
nhis.loc[nhis.LANY, 'limitation'] = 0.75
nhis.loc[nhis.LAMTWRK, 'limitation'] = 0.65
nhis.loc[nhis.LANOWORK, 'limitation'] = 0.4
nhis.loc[nhis.LAIADL, 'limitation'] = 0.2
nhis.loc[nhis.LADL, 'limitation'] = 0
nhis = nhis.drop(['LADL', 'LAIADL', 'LANOWORK', 'LAMTWRK', 'LANY', 'LNONE'], axis=1)

# Calculate the M_ij matrix as in Erickson, Wilson and Shannon (1995)
nhis.loc[:, 'halex'] = 0.41 * (nhis.HEALTH + nhis.limitation) + 0.18 * nhis.HEALTH * nhis.limitation
nhis = nhis.drop(['HEALTH', 'limitation'], axis=1)

# Load the population data and calculate the average age above 85 years old
df = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
df = df.loc[df.year.isin(nhis.YEAR.unique()) & (df.age >= 85), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'age': np.average(x.age, weights=x.population)})).rename(columns={'year': 'YEAR', 'age': 'age_85', 'race': 'RACEA'})
df.loc[:, 'age_85'] = np.round_(df.age_85).astype('int')

# Merge the population data with the NHIS data and set the age of 85 year olds to the average age above 85 years old
nhis = pd.merge(nhis, df, how='left')
nhis.loc[nhis.AGE == 85, 'AGE'] = nhis.age_85
nhis = nhis.drop('age_85', axis=1)

# Rename variables
nhis = nhis.rename(columns={'YEAR':      'year',
							'SEX':       'gender',
							'RACEA':     'race',
							'HISPETH':   'latin',
							'EDUCREC1':  'education',
							'AGE':       'age',
							'HEALTH':    'health',
							'PERWEIGHT': 'weight'})

# Define the variable types
nhis = nhis.astype({'year':      'int',
					'gender':    'int',
					'race':      'int',
					'latin':     'float',
					'education': 'float',
					'age':       'int',
					'halex':     'float',
					'weight':    'float'})

# Save the data
nhis.to_csv(os.path.join(nhis_f_data, 'nhis.csv'), index=False)
