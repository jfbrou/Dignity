# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Process the data from 1984 to 1989
files_1984_1989 = ['1984_data.TXT', '1985_data.TXT', '1986_data.TXT', '1987_data.TXT', '1988_data.TXT', '1989_data.TXT']
df_1984_1989 = expand({'year': range(1984, 1989 + 1, 1), 'race': {1, 2}})
for file in files_1984_1989:
    # Load the fixed-width file format using column positions based on the layout
    df_year = pd.read_fwf(os.path.join(pop_r_data, file), colspecs=[(0, 2), (2, 4), (4, 6), (6, 9), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)], 
                     names=['series', 'month', 'year', 'age', 'total', 'white_male', 'white_female', 'black_male', 'black_female'])

    # Only keep the relevant rows
    df_year = df_year.loc[(df_year.age >= 18) & (df_year.age < 85) & (df_year.year == int(file[2:4])) & (df_year.month == 7), :]
    
    # Sum White and Black populations
    total_white = df_year['white_male'].sum() + df_year['white_female'].sum()
    total_black = df_year['black_female'].sum() + df_year['black_female'].sum()
    
    # Extract the year from the file and append results
    df_1984_1989.loc[(df_1984_1989.year == int(file[:4])) & (df_1984_1989.race == 1), 'population'] = total_white
    df_1984_1989.loc[(df_1984_1989.year == int(file[:4])) & (df_1984_1989.race == 2), 'population'] = total_black

# Process the data from 1990 to 2020
df_1990_2020 = pd.read_csv(os.path.join(pop_r_data, '1990_2020_data.txt'), sep='\t')
df_1990_2020 = df_1990_2020[['Age', 'Race', 'Yearly July 1st Estimates', 'Population']].dropna()
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
df_1990_2020 = df_1990_2020[df_1990_2020['Age'].isin(age_filter)]
df_1990_2020 = df_1990_2020[df_1990_2020['Race'].isin(['Black or African American', 'White'])]
df_1990_2020 = df_1990_2020.groupby(['Yearly July 1st Estimates', 'Race']).agg({'Population': 'sum'}).reset_index().rename(columns={'Yearly July 1st Estimates': 'year', 'Race': 'race', 'Population': 'population'})
df_1990_2020.loc[df_1990_2020.race == 'White', 'race'] = 1
df_1990_2020.loc[df_1990_2020.race == 'Black or African American', 'race'] = 2

# Process the data from 2021 to 2022
files_2021_2022 = ['2021_data.csv', '2022_data.csv']
df_2021_2022 = expand({'year': range(2021, 2022 + 1, 1), 'race': {1, 2}})
for file in files_2021_2022:
    # Load the dataset
    df_year = pd.read_csv(os.path.join(pop_r_data, file))

    # Only keep the relevant rows
    df_year = df_year.loc[(df_year.AGE >= 18) & (df_year.AGE < 85), :]

    # Sum White and Black populations
    total_black = df_year['BA_MALE'].sum() + df_year['BA_FEMALE'].sum()
    total_white = df_year['WA_MALE'].sum() + df_year['WA_FEMALE'].sum()

    # Extract the year from the file and append results
    df_2021_2022.loc[(df_2021_2022.year == int(file[:4])) & (df_2021_2022.race == 1), 'population'] = total_white
    df_2021_2022.loc[(df_2021_2022.year == int(file[:4])) & (df_2021_2022.race == 2), 'population'] = total_black

# Combine and save the data
pd.concat([df_1984_1989, df_1990_2020, df_2021_2022], ignore_index=True).to_csv(os.path.join(pop_f_data, 'population.csv'), index=False)