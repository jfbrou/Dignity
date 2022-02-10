# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Define variable columns
columns = ['Notes', 'Single-Year Ages', 'Gender Code', 'Hispanic Origin', 'Race', 'Year', 'Deaths']

# Define variable names
names = ['notes', 'age', 'gender', 'latin', 'race', 'year', 'deaths']

# Load the data
df = pd.read_csv(os.path.join(cdc_r_data, 'deaths.txt'), delimiter='\t', header=0, usecols=columns).rename(columns=dict(zip(columns, names)))

# Drop observations where age is missing
df = df.loc[df.notes.isna() & (df.age != 'Not Stated'), :].drop('notes', axis=1)

# Recode the age variable
df.loc[:, 'age'] = df.age.map(dict(zip(df.age.unique(), range(0, 100 + 1))))

# Recode the gender variable
df.loc[:, 'gender'] = df.gender.map({'M': 1, 'F': 2})

# Recode the latin origin variable
df.loc[:, 'latin'] = df.latin.map({'Hispanic or Latino': 1, 'Not Hispanic or Latino': 0, 'Not Stated': 0})

# Recode the race variable
df.loc[:, 'race'] = df.race.map({'White': 1, 'Black or African American': 2, 'American Indian or Alaska Native': 3, 'Asian or Pacific Islander': 4})

# Group and save the data
df = df.groupby(['year', 'race', 'gender', 'latin', 'age'], as_index=False).agg({'deaths': 'sum'}).astype({'year': 'int', 'deaths': 'int'})
df.to_csv(os.path.join(cdc_f_data, 'cdc.csv'), index=False)
