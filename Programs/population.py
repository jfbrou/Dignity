# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Define the column names and their widths for the 1984 to 1989 data
columns = [
    (2, 4),     # Month
    (4, 6),     # Year
    (6, 9),     # Age
    (140, 150), # White, non-Hispanic male population
    (150, 160), # White, non-Hispanic female population
    (160, 170), # Black, non-Hispanic male population
    (170, 180), # Black, non-Hispanic female population
]
names = [
    "month", 
    "year", 
    "age", 
    "wm", 
    "wf", 
    "bm", 
    "bf", 
]

# Process the data from 1984 to 1989 data
df_1984_1989 = expand({'year': range(1984, 1990), 'race': [1, 2]})
df_1984_1989.loc[:, 'population'] = np.nan
for year in range(1984, 1990):
    df_year = pd.read_fwf(os.path.join(pop_r_data, 'E' + str(year)[2:] + str(year + 1)[2:] + 'RQI.TXT'), colspecs=columns, names=names)
    df_year = df_year.loc[(df_year.year == int(str(year)[2:])) & (df_year.month == 4) & df_year.age.isin(range(18, 85)), :]
    df_1984_1989.loc[(df_1984_1989.year == year) & (df_1984_1989.race == 1), 'population'] = df_year.wm.sum() + df_year.wf.sum()
    df_1984_1989.loc[(df_1984_1989.year == year) & (df_1984_1989.race == 2), 'population'] = df_year.bm.sum() + df_year.bf.sum()

# Process the data from 1990 to 2020
df_1990_2020 = pd.read_csv(os.path.join(pop_r_data, 'population.txt'), sep='\t')
df_1990_2020 = df_1990_2020[['Age', 'Race', 'Yearly July 1st Estimates', 'Population', 'Ethnicity']].dropna()
age_filter = ['18 years', '19 years', '20 years', '21 years', '22 years', '23 years', '24 years', 
              '25 years', '26 years', '27 years', '28 years', '29 years', '30 years', '31 years', 
              '32 years', '33 years', '34 years', '35 years', '36 years', '37 years', '38 years', 
              '39 years', '40 years', '41 years', '42 years', '43 years', '44 years', '45 years', 
              '46 years', '47 years', '48 years', '49 years', '50 years', '51 years', '52 years', 
              '53 years', '54 years', '55 years', '56 years', '57 years', '58 years', '59 years', 
              '60 years', '61 years', '62 years', '63 years', '64 years', '65 years', '66 years', 
              '67 years', '68 years', '69 years', '70 years', '71 years', '72 years', '73 years', 
              '74 years', '75 years', '76 years', '77 years', '78 years', '79 years', '80 years', 
              '81 years', '82 years', '83 years', '84 years']
df_1990_2020 = df_1990_2020[df_1990_2020.Age.isin(age_filter)]
df_1990_2020 = df_1990_2020[df_1990_2020.Race.isin(['Black or African American', 'White'])]
df_1990_2020 = df_1990_2020.loc[df_1990_2020.Ethnicity == 'Not Hispanic or Latino', :]
df_1990_2020 = df_1990_2020.groupby(['Yearly July 1st Estimates', 'Race']).agg({'Population': 'sum'}).reset_index().rename(columns={'Yearly July 1st Estimates': 'year', 'Race': 'race', 'Population': 'population'})
df_1990_2020.loc[df_1990_2020.race == 'White', 'race'] = 1
df_1990_2020.loc[df_1990_2020.race == 'Black or African American', 'race'] = 2

# Define variable columns for the 2021 to 2022 data
columns = ["Single-Year Ages Code", "Gender Code", "Hispanic Origin", "Single Race 6", "Year", "Deaths", "Population"]
names = ["age", "gender", "latin", "race", "year", "deaths", "population"]

# Peocess the data from 2021 to 2022
df_2021_2022 = pd.read_csv(os.path.join(cdc_r_data, "deaths.txt"), delimiter="\t", header=0, usecols=columns).rename(columns=dict(zip(columns, names)))
df_2021_2022 = df_2021_2022.loc[(df_2021_2022.age != "NS") & (df_2021_2022.population != "Not Applicable") & (df_2021_2022.latin != 'Hispanic or Latino') & df_2021_2022.race.isin(['White', 'Black or African American']) & pd.to_numeric(df_2021_2022.age, errors='coerce').isin(range(18, 85)), :]
df_2021_2022.loc[:, "race"] = df_2021_2022.race.map({"White": 1, "Black or African American": 2})
df_2021_2022.loc[:, "population"] = pd.to_numeric(df_2021_2022.population, errors='coerce')
df_2021_2022 = df_2021_2022.groupby(["year", 'race'], as_index=False).agg({'population': 'sum'})

# Adjust the data from 2021 to 2022
white_adjustment = np.mean(df_1990_2020.loc[df_1990_2020.year.isin(range(2018, 2021)) & (df_1990_2020.race == 1), 'population'].values / df_2021_2022.loc[df_2021_2022.year.isin(range(2018, 2021)) & (df_2021_2022.race == 1), 'population'].values)
black_adjustment = np.mean(df_1990_2020.loc[df_1990_2020.year.isin(range(2018, 2021)) & (df_1990_2020.race == 2), 'population'].values / df_2021_2022.loc[df_2021_2022.year.isin(range(2018, 2021)) & (df_2021_2022.race == 2), 'population'].values)
df_2021_2022.loc[df_2021_2022.race == 1, 'population'] = df_2021_2022.loc[df_2021_2022.race == 1, 'population'] * white_adjustment
df_2021_2022.loc[df_2021_2022.race == 2, 'population'] = df_2021_2022.loc[df_2021_2022.race == 2, 'population'] * black_adjustment
df_2021_2022 = df_2021_2022.loc[df_2021_2022.year.isin(range(2021, 2023)), :]

# Combine and save the data
df = pd.concat([df_1984_1989, df_1990_2020, df_2021_2022], ignore_index=True)
df.to_csv(os.path.join(pop_f_data, 'population.csv'), index=False)