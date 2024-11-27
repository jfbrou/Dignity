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
age_to_adult = {
    '< 1 year':    0,
    '1-4 years':   0,
    '5-9 years':   0,
    '10-14 years': 0,
    '15-19 years': 0,
    '20-24 years': 1,
    '25-29 years': 1,
    '30-34 years': 1,
    '35-39 years': 1,
    '40-44 years': 1,
    '45-49 years': 1,
    '50-54 years': 1,
    '55-59 years': 1,
    '60-64 years': 1,
    '65-69 years': 1,
    '70-74 years': 1,
    '75-79 years': 1,
    '80-84 years': 1,
    '85+ years':   1,
    '85-89 years': 1, 
    '90-94 years': 1, 
    '95-99 years': 1,
    '100+ years':  1
}

# Process the data from 1980 to 1989 data
# URL: https://www.census.gov/data/datasets/time-series/demo/popest/1980s-state.html
columns_1984_1989 = [
    (0, 2),     # State FIPS code
    (2, 3),     # Year
    (3, 4),     # Race
    (4, 5),     # Gender
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
names_1984_1989 = [
    'state',
    'year',
    'race',
    'gender',
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
df_1984_1989 = pd.read_fwf(os.path.join(pop_r_data, 'st_int_asrh.txt'), colspecs=columns_1984_1989, names=names_1984_1989)
df_1984_1989.loc[:, 'total_population'] = df_1984_1989.loc[:, '0-4':'85+'].sum(axis=1)
df_1984_1989.loc[:, 'adult_population'] = df_1984_1989.loc[:, '20-24':'85+'].sum(axis=1)
df_1984_1989 = df_1984_1989.drop(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+'], axis=1)
df_1984_1989.loc[:, 'year'] = df_1984_1989.year + 1980
df_1984_1989 = df_1984_1989.loc[df_1984_1989.race.isin([1, 2]) & df_1984_1989.year.isin(range(1984, 1989 + 1, 1)), :]
df_1984_1989.loc[:, 'region'] = df_1984_1989.state.map(fips_to_state).map(state_to_region).map({'Northeast': 1, 'Midwest': 1, 'South': 2, 'West': 1})
df_1984_1989 = df_1984_1989.drop('state', axis=1)
df_1984_1989 = df_1984_1989.groupby(['year', 'region', 'race'], as_index=False).agg({'total_population': 'sum', 'adult_population': 'sum'})

# Process the census data from 1990 to 1999
# URL: https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-1990-2000-state-and-county-characteristics.html
columns_1990_1999 = [
    (0, 2),   # Year
    (4, 6),   # State FIPS code
    (10, 12), # Age group
    (13, 14), # Race and gender
    (15, 16), # Latin origin
    (16, 23)  # Population
]
names_1990_1999 = [
    'year',
    'state',
    'age',
    'race',
    'latin',
    'population'
]
df_1990_1999 = pd.DataFrame()
for year in range(1990, 2000):
    df_year = pd.read_fwf(os.path.join(pop_r_data, 'stch-icen' + str(year) + '.txt'), colspecs=columns_1990_1999, names=names_1990_1999)
    df_year.loc[:, 'adult'] = (df_year.age >= 5)
    df_year.loc[:, 'year'] = df_year.year + 1900
    df_year.loc[:, 'region'] = df_year.state.map(fips_to_state).map(state_to_region).map({'Northeast': 1, 'Midwest': 1, 'South': 2, 'West': 1})
    df_year = df_year.loc[(df_year.race <= 4) & (df_year.latin == 1), :]
    df_year.loc[:, 'race'] = df_year.race.map({1: 1, 2: 1, 3: 2, 4: 2})
    df_year_adult = df_year.loc[df_year.adult == True, :].groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'}).rename(columns={'population': 'adult_population'})
    df_year_total = df_year.groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'}).rename(columns={'population': 'total_population'})
    df_year = pd.merge(df_year_total, df_year_adult)
    df_1990_1999 = pd.concat([df_1990_1999, df_year], ignore_index=True)

# Process the data from 1990 to 2020
df_1990_2020 = pd.read_csv(os.path.join(pop_r_data, 'population_1990_2020.txt'), sep='\t')
df_1990_2020 = df_1990_2020[['Age Group', 'Race', 'Region', 'Yearly July 1st Estimates', 'Population']].dropna()
df_1990_2020.loc[:, 'adult'] = df_1990_2020['Age Group'].map(age_to_adult)
df_1990_2020 = df_1990_2020.groupby(['Yearly July 1st Estimates', 'Race', 'Region', 'adult']).agg({'Population': 'sum'}).reset_index().rename(columns={'Yearly July 1st Estimates': 'year', 'Race': 'race', 'Region': 'region', 'Population': 'population'})
df_1990_2020.loc[:, 'race'] = df_1990_2020.race.map({'White': 1, 'Black or African American': 2})
df_1990_2020.loc[:, 'region'] = df_1990_2020.region.map({'Census Region 1: Northeast': 1, 'Census Region 2: Midwest': 1, 'Census Region 3: South': 2, 'Census Region 4: West': 1})
df_1990_2020_adult = df_1990_2020.loc[df_1990_2020.adult == 1, :].groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'}).rename(columns={'population': 'adult_population'})
df_1990_2020_total = df_1990_2020.groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'}).rename(columns={'population': 'total_population'})
df_1990_2020 = pd.merge(df_1990_2020_total, df_1990_2020_adult)

# Process the data from 2018 to 2022
df_2021_2022 = pd.read_csv(os.path.join(pop_r_data, 'population_2021_2022.txt'), sep='\t')
df_2021_2022 = df_2021_2022[['Five-Year Age Groups', 'Single Race 6', 'Census Region', 'Year', 'Population']].dropna()
df_2021_2022 = df_2021_2022.loc[df_2021_2022['Five-Year Age Groups'].isin(age_to_adult.keys()), :]
df_2021_2022.loc[:, 'adult'] = df_2021_2022['Five-Year Age Groups'].map(age_to_adult)
df_2021_2022 = df_2021_2022.loc[df_2021_2022.Population != 'Not Applicable', :]
df_2021_2022 = df_2021_2022.astype({'Population': 'int32'})
df_2021_2022 = df_2021_2022.groupby(['Year', 'Single Race 6', 'Census Region', 'adult']).agg({'Population': 'sum'}).reset_index().rename(columns={'Year': 'year', 'Single Race 6': 'race', 'Census Region': 'region', 'Population': 'population'})
df_2021_2022.loc[:, 'race'] = df_2021_2022.race.map({'White': 1, 'Black or African American': 2})
df_2021_2022.loc[:, 'region'] = df_2021_2022.region.map({'Census Region 1: Northeast': 1, 'Census Region 2: Midwest': 1, 'Census Region 3: South': 2, 'Census Region 4: West': 1})
df_2021_2022_adult = df_2021_2022.loc[df_2021_2022.adult == 1, :].groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'}).rename(columns={'population': 'adult_population'})
df_2021_2022_total = df_2021_2022.groupby(['year', 'region', 'race'], as_index=False).agg({'population': 'sum'}).rename(columns={'population': 'total_population'})
df_2021_2022 = pd.merge(df_2021_2022_total, df_2021_2022_adult)

# Adjust the data from 1984 to 1989 using the data from 1990 to 1999
for race in [1, 2]:
    for region in [1, 2]:
        for age in ['total', 'adult']:
            adjustment = np.mean(df_1990_2020.loc[df_1990_2020.year.isin(range(1990, 2000)) & (df_1990_2020.race == race) & (df_1990_2020.region == region), age + '_population'].values / df_1990_1999.loc[df_1990_1999.year.isin(range(1990, 2000)) & (df_1990_1999.race == race) & (df_1990_1999.region == region), age + '_population'].values)
            df_1984_1989.loc[(df_1984_1989.race == race) & (df_1984_1989.region == region), age + '_population'] = df_1984_1989.loc[(df_1984_1989.race == race) & (df_1984_1989.region == region), age + '_population'] * adjustment

# Adjust the data from 2021 to 2022 using the data from 2018 to 2020
for race in [1, 2]:
    for region in [1, 2]:
        for age in ['total', 'adult']:
            adjustment = np.mean(df_1990_2020.loc[df_1990_2020.year.isin(range(2018, 2021)) & (df_1990_2020.race == race) & (df_1990_2020.region == region), age + '_population'].values / df_2021_2022.loc[df_2021_2022.year.isin(range(2018, 2021)) & (df_2021_2022.race == race) & (df_2021_2022.region == region), age + '_population'].values)
            df_2021_2022.loc[(df_2021_2022.race == race) & (df_2021_2022.region == region), age + '_population'] = df_2021_2022.loc[(df_2021_2022.race == race) & (df_2021_2022.region == region), age + '_population'] * adjustment
df_2021_2022 = df_2021_2022.loc[df_2021_2022.year.isin(range(2021, 2023)), :]

# Combine and save the data
df = pd.concat([df_1984_1989, df_1990_2020, df_2021_2022], ignore_index=True)
df.to_csv(os.path.join(pop_f_data, 'population.csv'), index=False)