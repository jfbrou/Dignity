# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Map states to regions
state_to_region = {
    'AL': 'South',
    'AK': 'West',
    'AZ': 'West',
    'AR': 'South',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'South',
    'FL': 'South',
    'GA': 'South',
    'HI': 'West',
    'ID': 'West',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'South',
    'LA': 'South',
    'ME': 'Northeast',
    'MD': 'South',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'South',
    'MO': 'Midwest',
    'MT': 'West',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Northeast',
    'NM': 'West',
    'NY': 'Northeast',
    'NC': 'South',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'South',
    'OR': 'West',
    'PA': 'Northeast',
    'RI': 'Northeast',
    'SC': 'South',
    'SD': 'Midwest',
    'TN': 'South',
    'TX': 'South',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'South',
    'WA': 'West',
    'WV': 'South',
    'WI': 'Midwest',
    'WY': 'West'
}
fips_to_state = {
    1: 'AL',
    2: 'AK',
    4: 'AZ',
    5: 'AR',
    6: 'CA',
    8: 'CO',
    9: 'CT',
    10: 'DE',
    11: 'DC',
    12: 'FL',
    13: 'GA',
    15: 'HI',
    16: 'ID',
    17: 'IL',
    18: 'IN',
    19: 'IA',
    20: 'KS',
    21: 'KY',
    22: 'LA',
    23: 'ME',
    24: 'MD',
    25: 'MA',
    26: 'MI',
    27: 'MN',
    28: 'MS',
    29: 'MO',
    30: 'MT',
    31: 'NE',
    32: 'NV',
    33: 'NH',
    34: 'NJ',
    35: 'NM',
    36: 'NY',
    37: 'NC',
    38: 'ND',
    39: 'OH',
    40: 'OK',
    41: 'OR',
    42: 'PA',
    44: 'RI',
    45: 'SC',
    46: 'SD',
    47: 'TN',
    48: 'TX',
    49: 'UT',
    50: 'VT',
    51: 'VA',
    53: 'WA',
    54: 'WV',
    55: 'WI',
    56: 'WY',
    60: 'AS', 
    66: 'GU',  
    69: 'MP',  
    72: 'PR',  
    78: 'VI'   
}

# Define the column names and their widths for the 1984 to 1989 data
# URL: https://www.census.gov/data/datasets/time-series/demo/popest/1980s-state.html
columns = [
    (0, 2),     # State FIPS code
    (2, 3),     # Year
    (3, 4),     # Race
    (5, 12),    # 0-4 year olds
    (12, 19),   # 5-9 year olds
    (19, 26),   # 10-14 year olds
    (26, 33),   # 15-19 year olds
    (33, 40),   # 20-24 year olds
    (40, 47),   # 25-29 year olds
    (47, 54),   # 30-34 year olds
    (54, 61),   # 35-39 year olds
    (61, 68),   # 40-44 year olds
    (68, 75),   # 45-49 year olds
    (75, 82),   # 50-54 year olds
    (82, 89),   # 55-59 year olds
    (89, 96),   # 60-64 year olds
    (96, 103),  # 65-69 year olds
    (103, 110), # 70-74 year olds
    (110, 117), # 75-79 year olds
    (117, 124), # 80-84 year olds
    (124, 131)  # 85+ year olds
]
names = [
    'state',
    'year',
    'race',
    '0-4',
    '5-9',
    '10-14',
    '15-19',
    '20-24',
    '25-29',
    '30-34',
    '35-39',
    '40-44',
    '45-49',
    '50-54',
    '55-59',
    '60-64',
    '65-69',
    '70-74',
    '75-79',
    '80-84',
    '85+'
]

# Process the data from 1984 to 1989 data
df_1984_1989 = pd.read_fwf(os.path.join(pop_r_data, 'st_int_asrh.txt'), colspecs=columns, names=names)
df_1984_1989.loc[:, 'population'] = df_1984_1989.loc[:, '0-4':'85+'].sum(axis=1)
df_1984_1989 = df_1984_1989.drop(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+'], axis=1)
df_1984_1989.loc[:, 'year'] = df_1984_1989.year + 1980
df_1984_1989 = df_1984_1989.loc[df_1984_1989.race.isin([1, 2]) & df_1984_1989.year.isin(range(1984, 1989 + 1, 1)), :]
df_1984_1989.loc[:, 'region'] = df_1984_1989.state.map(fips_to_state).map(state_to_region).map({'Northeast': 1, 'Midwest': 1, 'South': 2, 'West': 1})
df_1984_1989 = df_1984_1989.drop('state', axis=1)
df_1984_1989 = df_1984_1989.groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'})

# Process the data from 1990 to 2020
df_1990_2020 = pd.read_csv(os.path.join(pop_r_data, 'population_1990_2020.txt'), sep='\t')
df_1990_2020 = df_1990_2020[['Race', 'Region', 'Yearly July 1st Estimates', 'Population']].dropna()
df_1990_2020 = df_1990_2020.groupby(['Yearly July 1st Estimates', 'Race', 'Region']).agg({'Population': 'sum'}).reset_index().rename(columns={'Yearly July 1st Estimates': 'year', 'Race': 'race', 'Region': 'region', 'Population': 'population'})
df_1990_2020.loc[:, 'race'] = df_1990_2020.race.map({'White': 1, 'Black or African American': 2})
df_1990_2020.loc[:, 'region'] = df_1990_2020.region.map({'Census Region 1: Northeast': 1, 'Census Region 2: Midwest': 1, 'Census Region 3: South': 2, 'Census Region 4: West': 1})
df_1990_2020 = df_1990_2020.groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'})

# Process the data from 2018 to 2022
df_2021_2022 = pd.read_csv(os.path.join(pop_r_data, 'population_2021_2022.txt'), sep='\t')
df_2021_2022 = df_2021_2022[['Single Race 6', 'Census Region', 'Year', 'Population']].dropna()
df_2021_2022 = df_2021_2022.groupby(['Year', 'Single Race 6', 'Census Region']).agg({'Population': 'sum'}).reset_index().rename(columns={'Year': 'year', 'Single Race 6': 'race', 'Census Region': 'region', 'Population': 'population'})
df_2021_2022.loc[:, 'race'] = df_2021_2022.race.map({'White': 1, 'Black or African American': 2})
df_2021_2022.loc[:, 'region'] = df_2021_2022.region.map({'Census Region 1: Northeast': 1, 'Census Region 2: Midwest': 1, 'Census Region 3: South': 2, 'Census Region 4: West': 1})
df_2021_2022 = df_2021_2022.groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'})

# Adjust the data from 2021 to 2022
white_adjustment = np.mean(df_1990_2020.loc[df_1990_2020.year.isin(range(2018, 2021)) & (df_1990_2020.race == 1), 'population'].values / df_2021_2022.loc[df_2021_2022.year.isin(range(2018, 2021)) & (df_2021_2022.race == 1), 'population'].values)
black_adjustment = np.mean(df_1990_2020.loc[df_1990_2020.year.isin(range(2018, 2021)) & (df_1990_2020.race == 2), 'population'].values / df_2021_2022.loc[df_2021_2022.year.isin(range(2018, 2021)) & (df_2021_2022.race == 2), 'population'].values)
df_2021_2022.loc[df_2021_2022.race == 1, 'population'] = df_2021_2022.loc[df_2021_2022.race == 1, 'population'] * white_adjustment
df_2021_2022.loc[df_2021_2022.race == 2, 'population'] = df_2021_2022.loc[df_2021_2022.race == 2, 'population'] * black_adjustment
df_2021_2022 = df_2021_2022.loc[df_2021_2022.year.isin(range(2021, 2023)), :]

# Combine and save the data
df = pd.concat([df_1984_1989, df_1990_2020, df_2021_2022], ignore_index=True)
df.to_csv(os.path.join(pop_f_data, 'population.csv'), index=False)