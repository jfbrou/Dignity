# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from zipfile import ZipFile
from dotenv import load_dotenv
load_dotenv()
import os

# Import functions and directories
from functions import *
from directories import *

################################################################################
#                                                                              #
# This section of the script computes the total U.S. population by gender and  #
# race for the 1970s.                                                          #
#                                                                              #
################################################################################

# Define variable names
names = [1, 2, 3, 4, 9, 10]

# Define variable types
types = dict(zip(names, ['int'] * 6))

# Define the columns to use
columns = [5, 6, 8, 9, 11, 12]

# Define the rows to skip
rows = [0, 1, 2, 3, 5, 6, 7]

# Define a data frame
df_1970 = pd.DataFrame(columns=['year', 'gender', 'race', 'latin', 'age', 'population'])

# Process data files for all years
for year in list(range(1970, 1979 + 1)):
    # Load the data
    df = pd.read_excel(os.path.join(population_r_data, 'brpe_1970-1979.xls'), header=0, sheet_name=str(year), skiprows=rows, skipfooter=15, usecols=columns, names=names, dtype=types)

    # Create the age variable
    df.loc[:, 'age'] = range(85 + 1)

    # Reshape the data
    df = pd.melt(df, id_vars='age', value_vars=df.columns[df.columns != 'age']).rename(columns={'variable': 'group', 'value': 'population'})

    # Create the gender variable
    df.loc[df.group.isin([1, 3, 9]), 'gender'] = 1
    df.loc[df.group.isin([2, 4, 10]), 'gender'] = 2

    # Create the race variable
    df.loc[df.group.isin([1, 2]), 'race'] = 1
    df.loc[df.group.isin([3, 4]), 'race'] = 2
    df.loc[df.group.isin([9, 10]), 'race'] = -1

    # Aggregate the population estimates by gender, race and age
    df = df.groupby(['gender', 'race', 'age'], as_index=False).agg({'population': 'sum'})

    # Create the year variable and append the data frames for all years and redefine variable types
    df.loc[:, 'year'] = year
    df_1970 = df_1970.append(df, ignore_index=True)
    df_1970 = df_1970.astype({'year':       'int',
                              'gender':     'int',
                              'race':       'int8',
                              'age':        'int',
                              'population': 'int'})

# Recode the race missing values
df_1970.loc[df_1970.race == -1, 'race'] = np.nan

################################################################################
#                                                                              #
# This section of the script computes the total U.S. population by gender and  #
# race for the 1980s.                                                          #
#                                                                              #
################################################################################

# Define variable columns
columns = [(3, 4),     # Month
           (4, 6),     # Year
           (6, 9),     # Age
           (40, 50),   # White males
           (50, 60),   # White females
           (60, 70),   # Black males
           (70, 80),   # Black females
           (80, 90),   # Native American males
           (90, 100),  # Native American females
           (100, 110), # Asian and Pacific Islander males
           (110, 120), # Asian and Pacific Islander females
           (140, 150), # White non-Latin males
           (150, 160), # White non-Latin females
           (160, 170), # Black non-Latin males
           (170, 180), # Black non-Latin females
           (180, 190), # Native American non-Latin males
           (190, 200), # Native American non-Latin females
           (200, 210), # Asian and Pacific Islander non-Latin males
           (210, 220)] # Asian and Pacific Islander non-Latin females

# Define variable names
names = ['month', 'year', 'age'] + list(range(1, 8 + 1)) + list(range(10, 80 + 1, 10))

# Define variable types
types = dict(zip(names, ['int', 'int', 'int'] + ['int' for i in range(16)]))

# Define a data frame
df_1980 = pd.DataFrame(columns=['year', 'gender', 'race', 'latin', 'age', 'population'])

# Process data files for all years
for year in list(range(1980, 1989 + 1)):
    # Load the data
    z = ZipFile(os.path.join(population_r_data, 'brpe_' + str(year) + '.zip'))
    df = pd.read_fwf(z.open(z.namelist()[0]), colspecs=columns, names=names, dtype=types, skipfooter=1)

    # Drop the aggregate age category
    df = df.loc[df.age != 999, :]

    # Recode the year and month variables
    df.loc[:, 'year'] = df.year + 1900
    df = df.loc[df.month == 7, :]

    # Reshape the data
    df = pd.melt(df, id_vars=['year', 'age'], value_vars=df.columns[(df.columns != 'year')
                                                                    & (df.columns != 'month')
                                                                    & (df.columns != 'age')].tolist()).rename(columns={'variable': 'group', 'value': 'population'})

    # Create the gender variable
    df.loc[df.group.isin([1, 3, 5, 7, 10, 30, 50, 70]), 'gender'] = 1
    df.loc[df.group.isin([2, 4, 6, 8, 20, 40, 60, 80]), 'gender'] = 2

    # Create the race variable
    df.loc[df.group.isin([1, 2, 10, 20]), 'race'] = 1
    df.loc[df.group.isin([3, 4, 30, 40]), 'race'] = 2
    df.loc[df.group.isin([5, 6, 50, 60]), 'race'] = 3
    df.loc[df.group.isin([7, 8, 70, 80]), 'race'] = 4

    # Create the latin origin variable
    df_nonlatin = df.loc[df.group >= 10, :].groupby(['year', 'gender', 'race', 'age'], as_index=False).agg({'population': 'sum'})
    df_total = df.loc[df.group < 10, :].groupby(['year', 'gender', 'race', 'age'], as_index=False).agg({'population': 'sum'})
    df_latin = pd.merge(df_nonlatin, df_total.rename(columns={'population': 'total'}), how='left')
    df_latin.loc[:, 'population'] = df_latin.total - df_latin.population
    df_latin = df_latin.drop('total', axis=1)
    df_latin.loc[:, 'latin'] = 1
    df_nonlatin.loc[:, 'latin'] = 0
    df = df_nonlatin.append(df_latin, ignore_index=True)

    # Append the data frames for all years and redefine variable types
    df_1980 = df_1980.append(df, ignore_index=True)
    df_1980 = df_1980.astype({'year':       'int',
                              'gender':     'int',
                              'race':       'int',
                              'latin':      'int',
                              'age':        'int',
                              'population': 'int'})

################################################################################
#                                                                              #
# This section of the script computes the total U.S. population by gender and  #
# race for the 1990s.                                                          #
#                                                                              #
################################################################################

# Define variable columns
columns = ['Notes', 'Age Code', 'Ethnicity', 'Gender Code', 'Race', 'Yearly July 1st Estimates', 'Population']

# Define variable types
types = dict(zip(columns, ['str', 'str', 'str', 'str', 'str', 'float', 'float']))

# Load the data
df_1990 = pd.read_csv(os.path.join(population_r_data, 'brpe_1990-2020_young.txt'), delimiter='\t', header=0, usecols=columns, dtype=types)

# Rename variables
df_1990 = df_1990.rename(columns={'Notes':                     'notes',
                                  'Age Code':                  'age',
                                  'Ethnicity':                 'latin',
                                  'Gender Code':               'gender',
                                  'Race':                      'race',
                                  'Yearly July 1st Estimates': 'year',
                                  'Population':                'population'})

# Drop the aggregate age category
df_1990 = df_1990.loc[df_1990.notes.isna(), :].drop('notes', axis=1)

# Keep the 1990s years
df_1990 = df_1990.loc[(df_1990.year >= 1990) & (df_1990.year <= 1999), :]

# Recode the age variable
df_1990.loc[df_1990.age == '85+', 'age'] = 85
df_1990.loc[:, 'age'] = df_1990.age.astype('int')

# Recode the gender variable
df_1990.loc[:, 'gender'] = df_1990.gender.map({'M': 1, 'F': 2})

# Recode the race variable
df_1990.loc[:, 'race'] = df_1990.race.map({'White': 1, 'Black or African American': 2, 'American Indian or Alaska Native': 3, 'Asian or Pacific Islander': 4})

# Recode the latin origin variable
df_1990.loc[:, 'latin'] = df_1990.latin.map({'Not Hispanic or Latino': 0, 'Hispanic or Latino': 1})

# Redefine variable types
df_1990 = df_1990.astype({'year':       'int',
                          'gender':     'int',
                          'race':       'int',
                          'latin':      'int',
                          'age':        'int',
                          'population': 'int'})

################################################################################
#                                                                              #
# This section of the script computes the total U.S. population by gender and  #
# race for the 2000s.                                                          #
#                                                                              #
################################################################################

# Define variable columns
young_columns = ['Notes', 'Age Code', 'Ethnicity', 'Gender Code', 'Race', 'Yearly July 1st Estimates', 'Population']
old_columns = [(8, 12), (12, 13), (13, 16), (16, 17), (17, 18), (18, 26)]

# Define variable names
old_names = ['year', 'month', 'age', 'group', 'latin', 'population']

# Define variable types
young_types = dict(zip(columns, ['str', 'str', 'str', 'str', 'str', 'float', 'float']))
old_types = dict(zip(old_names, ['int', 'int', 'int', 'int', 'int', 'int']))

# Load the data
df_old = pd.read_fwf(os.path.join(population_r_data, 'brpe_2000-2009_old.txt'), colspecs=old_columns, names=old_names, dtype=old_types)
df_young = pd.read_csv(os.path.join(population_r_data, 'brpe_1990-2020_young.txt'), delimiter='\t', header=0, usecols=young_columns, dtype=young_types)

# Rename variables in the young data frame
df_young = df_young.rename(columns={'Notes':                     'notes',
                                    'Age Code':                  'age',
                                    'Ethnicity':                 'latin',
                                    'Gender Code':               'gender',
                                    'Race':                      'race',
                                    'Yearly July 1st Estimates': 'year',
                                    'Population':                'population'})

# Drop the aggregate age category
df_young = df_young.loc[df_young.notes.isna(), :].drop('notes', axis=1)

# Keep the 2000s years
df_young = df_young.loc[(df_young.year >= 2000) & (df_young.year <= 2009), :]

# Recode the age variable
df_young = df_young.loc[df_young.age != '85+', :]
df_young.loc[:, 'age'] = df_young.age.astype('int')

# Recode the gender variable
df_young.loc[:, 'gender'] = df_young.gender.map({'M': 1, 'F': 2})

# Recode the race variable
df_young.loc[:, 'race'] = df_young.race.map({'White': 1, 'Black or African American': 2, 'American Indian or Alaska Native': 3, 'Asian or Pacific Islander': 4})

# Recode the latin origin variable
df_young.loc[:, 'latin'] = df_young.latin.map({'Not Hispanic or Latino': 0, 'Hispanic or Latino': 1})

# Keep the July estimates in the old data frame
df_old = df_old.loc[df_old.month == 7, :].drop('month', axis=1)

# Create the gender variable in the old data frame
df_old.loc[df_old.group.isin([1, 3, 5, 7]), 'gender'] = 1
df_old.loc[df_old.group.isin([2, 4, 6, 8]), 'gender'] = 2

# Create the race variable in the old data frame
df_old.loc[df_old.group.isin([1, 2]), 'race'] = 1
df_old.loc[df_old.group.isin([3, 4]), 'race'] = 2
df_old.loc[df_old.group.isin([5, 6]), 'race'] = 3
df_old.loc[df_old.group.isin([7, 8]), 'race'] = 4

# Recode the latin origin variable in the old data frame
df_old.loc[:, 'latin'] = df_old.latin.map({1: 0, 2: 1})

# Append the young and old data frames
df_old = df_old.drop('group', axis=1)
df_2000 = df_young.append(df_old, ignore_index=True)

# Redefine variable types
df_2000 = df_2000.astype({'year':       'int',
                          'gender':     'int',
                          'race':       'int',
                          'latin':      'int',
                          'age':        'int',
                          'population': 'int'})

################################################################################
#                                                                              #
# This section of the script computes the total U.S. population by gender and  #
# race for the 2010s.                                                          #
#                                                                              #
################################################################################

# Define variable columns
young_columns = ['Notes', 'Age Code', 'Ethnicity', 'Gender Code', 'Race', 'Yearly July 1st Estimates', 'Population']
old_columns = [(4, 8), (8, 9), (9, 12), (12, 13), (13, 14), (14, 22)]

# Define variable names
old_names = ['year', 'month', 'age', 'group', 'latin', 'population']

# Define variable types
young_types = dict(zip(columns, ['str', 'str', 'str', 'str', 'str', 'float', 'float']))
old_types = dict(zip(old_names, ['int', 'int', 'int', 'int', 'int', 'int']))

# Load the data
df_old = pd.read_fwf(os.path.join(population_r_data, 'brpe_2010-2020_old.txt'), colspecs=old_columns, names=old_names, dtype=old_types)
df_young = pd.read_csv(os.path.join(population_r_data, 'brpe_1990-2020_young.txt'), delimiter='\t', header=0, usecols=young_columns, dtype=young_types)

# Rename variables in the young data frame
df_young = df_young.rename(columns={'Notes':                     'notes',
                                    'Age Code':                  'age',
                                    'Ethnicity':                 'latin',
                                    'Gender Code':               'gender',
                                    'Race':                      'race',
                                    'Yearly July 1st Estimates': 'year',
                                    'Population':                'population'})

# Drop the aggregate age category
df_young = df_young.loc[df_young.notes.isna(), :].drop('notes', axis=1)

# Keep the 2010s years
df_young = df_young.loc[(df_young.year >= 2010) & (df_young.year <= 2020), :]

# Recode the age variable
df_young = df_young.loc[df_young.age != '85+', :]
df_young.loc[:, 'age'] = df_young.age.astype('int')

# Recode the gender variable
df_young.loc[:, 'gender'] = df_young.gender.map({'M': 1, 'F': 2})

# Recode the race variable
df_young.loc[:, 'race'] = df_young.race.map({'White': 1, 'Black or African American': 2, 'American Indian or Alaska Native': 3, 'Asian or Pacific Islander': 4})

# Recode the latin origin variable
df_young.loc[:, 'latin'] = df_young.latin.map({'Not Hispanic or Latino': 0, 'Hispanic or Latino': 1})

# Keep the July estimates in the old data frame
df_old = df_old.loc[df_old.month == 7, :].drop('month', axis=1)

# Create the gender variable in the old data frame
df_old.loc[df_old.group.isin([1, 3, 5, 7]), 'gender'] = 1
df_old.loc[df_old.group.isin([2, 4, 6, 8]), 'gender'] = 2

# Create the race variable in the old data frame
df_old.loc[df_old.group.isin([1, 2]), 'race'] = 1
df_old.loc[df_old.group.isin([3, 4]), 'race'] = 2
df_old.loc[df_old.group.isin([5, 6]), 'race'] = 3
df_old.loc[df_old.group.isin([7, 8]), 'race'] = 4

# Recode the latin origin variable in the old data frame
df_old.loc[:, 'latin'] = df_old.latin.map({1: 0, 2: 1})

# Append the young and old data frames
df_old = df_old.drop('group', axis=1)
df_2010 = df_young.append(df_old, ignore_index=True)

# Redefine variable types
df_2010 = df_2010.astype({'year':       'int',
                          'gender':     'int',
                          'race':       'int',
                          'latin':      'int',
                          'age':        'int',
                          'population': 'int'})

################################################################################
#                                                                              #
# Interpolate the age distribution within the 85+ years old age group by       #
# gender and race for the 1970s.                                               #
#                                                                              #
################################################################################

# Compute the age distribution within the 85+ years old age group by gender and race for 1970
df_1 = acs.loc[(acs.year == 1970) & (acs.age >= 85) & ((acs.race == 1) | (acs.race == 2)), :].groupby(['gender', 'race', 'age'], as_index=False).agg({'weight': 'sum'})
df_2 = acs.loc[(acs.year == 1970) & (acs.age >= 85) & ((acs.race != 1) & (acs.race != 2)), :].groupby(['gender', 'age'], as_index=False).agg({'weight': 'sum'})
df_2.loc[:, 'race'] = -1
df_70 = df_1.append(df_2, ignore_index=True)
df_70.loc[:, 'share'] = df_70.groupby(['gender', 'race'], as_index=False).weight.transform(lambda x: x / x.sum()).values
df_70.loc[:, 'year'] = 1970
df_70 = df_70.drop('weight', axis=1)

# Compute the age distribution within the 85+ years old age group by gender and race for the first two years of the 1980s
df_1 = df_1980.loc[(df_1980.year <= 1981) & (df_1980.age >= 85) & ((df_1980.race == 1) | (df_1980.race == 2)), :].groupby(['gender', 'race', 'age'], as_index=False).agg({'population': 'sum'})
df_2 = df_1980.loc[(df_1980.year <= 1981) & (df_1980.age >= 85) & ((df_1980.race != 1) & (df_1980.race != 2)), :].groupby(['gender', 'age'], as_index=False).agg({'population': 'sum'})
df_2.loc[:, 'race'] = -1
df_80 = df_1.append(df_2, ignore_index=True)
df_80.loc[:, 'share'] = df_80.groupby(['gender', 'race'], as_index=False).population.transform(lambda x: x / x.sum()).values
df_80.loc[:, 'year'] = 1980
df_80 = df_80.drop('population', axis=1)

# Interpolate the age distribution within the 85+ years old age group by gender and race for the 1970s
df = pd.merge(expand({'year': list(range(1970, 1980 + 1)), 'gender': range(1, 2 + 1), 'race': [1, 2, -1], 'age': range(85, 100 + 1)}), df_70.append(df_80, ignore_index=True), how='left')
df.loc[:, 'share'] = df.groupby(['gender', 'race', 'age'], as_index=False).share.transform(lambda x: x.interpolate(limit_direction='both')).values
df = pd.merge(df.loc[(df.year >= 1970) & (df.year <= 1979), :], df_1970.loc[df_1970.age == 85, ['year', 'gender', 'race', 'population']].fillna(-1), how='left')
df.loc[:, 'population'] = df.population * df.share
df = df.drop('share', axis=1)
df_1970 = df_1970.loc[df_1970.age < 85, :].append(df, ignore_index=True)
df_1970.loc[df_1970.race == -1, 'race'] = np.nan

################################################################################
#                                                                              #
# Interpolate the age distribution within the 85+ years old age group by       #
# gender and race for the 1990s.                                               #
#                                                                              #
################################################################################

# Compute the age distribution within the 85+ years old age group by gender and race for the last two years of the 1980s
df_80 = df_1980.loc[(df_1980.year >= 1988) & (df_1980.age >= 85), :].groupby(['gender', 'race', 'latin', 'age'], as_index=False).agg({'population': 'sum'})
df_80.loc[:, 'share'] = df_80.groupby(['gender', 'race', 'latin'], as_index=False).population.transform(lambda x: x / x.sum()).values
df_80.loc[:, 'year'] = 1989
df_80 = df_80.drop('population', axis=1)

# Compute the age distribution within the 85+ years old age group by gender and race for the first two years of the 2000s
df_00 = df_2000.loc[(df_2000.year <= 2001) & (df_2000.age >= 85), :].groupby(['gender', 'race', 'latin', 'age'], as_index=False).agg({'population': 'sum'})
df_00.loc[:, 'share'] = df_00.groupby(['gender', 'race', 'latin'], as_index=False).population.transform(lambda x: x / x.sum()).values
df_00.loc[:, 'year'] = 2000
df_00 = df_00.drop('population', axis=1)

# Interpolate the age distribution within the 85+ years old age group by gender and race for the 1990s
df = pd.merge(expand({'year': range(1989, 2000 + 1), 'gender': range(1, 2 + 1), 'race': range(1, 4 + 1), 'latin': range(0, 1 + 1), 'age': range(85, 100 + 1)}), df_80.append(df_00, ignore_index=True), how='left')
df.loc[:, 'share'] = df.groupby(['gender', 'race', 'latin', 'age'], as_index=False).share.transform(lambda x: x.interpolate(limit_direction='both')).values
df = pd.merge(df.loc[(df.year >= 1990) & (df.year <= 1999), :], df_1990.loc[df_1990.age == 85, ['year', 'gender', 'race', 'latin', 'population']], how='left')
df.loc[:, 'population'] = df.population * df.share
df = df.drop('share', axis=1)
df_1990 = df_1990.loc[df_1990.age < 85, :].append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script appends all the above population data frames.     #
#                                                                              #
################################################################################

# Append the population estimates data frames for all years
population = df_1970.append(df_1980.append(df_1990.append(df_2000.append(df_2010, ignore_index=True), ignore_index=True), ignore_index=True), ignore_index=True)
population = population.sort_values(by=['year', 'gender', 'race', 'latin', 'age'])
population = population.astype({'year':       'int',
                                'gender':     'int',
                                'race':       'float',
                                'latin':      'float',
                                'age':        'int',
                                'population': 'float'})

# Save the data
population.to_csv(os.path.join(population_f_data, 'population.csv'), index=False)
