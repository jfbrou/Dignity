# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Process the data from 1990 to 2020
df = pd.read_csv(os.path.join(pop_r_data, 'population.txt'), sep='\t')
df = df[['Age', 'Race', 'Yearly July 1st Estimates', 'Population', 'Ethnicity']].dropna()
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
df = df[df['Age'].isin(age_filter)]
df = df[df['Race'].isin(['Black or African American', 'White'])]
df = df.loc[df.Ethnicity == 'Not Hispanic or Latino', :]
df = df.groupby(['Yearly July 1st Estimates', 'Race']).agg({'Population': 'sum'}).reset_index().rename(columns={'Yearly July 1st Estimates': 'year', 'Race': 'race', 'Population': 'population'})
df.loc[df.race == 'White', 'race'] = 1
df.loc[df.race == 'Black or African American', 'race'] = 2

# Combine and save the data
df.to_csv(os.path.join(pop_f_data, 'population.csv'), index=False)