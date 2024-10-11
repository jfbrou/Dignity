# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Load and process the NPS data
nps = pd.read_csv(os.path.join(nps_r_data, 'nps.tsv'), delimiter='\t', usecols=['YEAR', 'STATE', 'WHITEM', 'WHITEF', 'BLACKM', 'BLACKF'])
nps = nps.loc[(nps.STATE == 'US') & nps.YEAR.isin(range(1984, 2022 + 1, 1)), :].drop('STATE', axis=1)
nps.loc[:, 'WHITE'] = nps.WHITEM + nps.WHITEF
nps.loc[:, 'BLACK'] = nps.BLACKM + nps.BLACKF
nps_white = nps.loc[:, ['YEAR', 'WHITE']].rename(columns={'WHITE': 'incarcerated_nps'})
nps_white.loc[:, 'race'] = 1
nps_black = nps.loc[:, ['YEAR', 'BLACK']].rename(columns={'BLACK': 'incarcerated_nps'})
nps_black.loc[:, 'race'] = 2
nps = pd.concat([nps_white, nps_black], ignore_index=True).rename(columns={'YEAR': 'year'})

# Load and process the ASJ data
asj = expand({'year': range(1984, 2022 + 1, 1), 'race': {1, 2}})
for year in list(range(1985, 1987 + 1, 1)) + list(range(1989, 1992 + 1, 1)) + list(range(1994, 1998 + 1, 1)) + list(range(2000, 2003 + 1, 1)) + [2004, 2006] + list(range(2008, 2019, 1)) + list(range(2020, 2022 + 1, 1)):
    if (year >= 2013) | ((year >= 2008) & (year <= 2009)):
        df = pd.read_csv(os.path.join(asj_r_data, str(year) + '.tsv'), sep='\t')
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'WHITE'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'BLACK'].sum()
    elif (year >= 2010) & (year <= 2012):
        df = pd.read_csv(os.path.join(asj_r_data, str(year) + '.tsv'), sep='\t')
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'white'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'black'].sum()
    elif (year >= 2003) & (year <= 2006):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'WHITE'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'BLACK'].sum()
    elif (year >= 2000) & (year <= 2002):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'V51'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'V53'].sum()
    elif year == 1998:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'V43'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'V45'].sum()
    elif (year >= 1996) & (year <= 1997):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[:, 'V42'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[:, 'V44'], errors='coerce').sum()
    elif year == 1995:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[:, 'V36'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[:, 'V38'], errors='coerce').sum()
    elif year == 1994:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[:, 'V26'], errors='coerce').sum().sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[:, 'V28'], errors='coerce').sum().sum()
    elif (year >= 1991) & (year <= 1992):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V65', 'V70'])
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'V65'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'V70'].sum()
    elif (year >= 1989) & (year <= 1990):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V63', 'V68'])
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'V63'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'V68'].sum()
    elif year == 1987:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V53', 'V58'])
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'V53'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'V58'].sum()
    elif year == 1986:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V50', 'V55'])
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'V50'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'V55'].sum()
    elif year == 1985:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V44', 'V49'])
        asj.loc[(asj.year == year) & (asj.race == 1), 'incarcerated_asj'] = df.loc[:, 'V44'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2), 'incarcerated_asj'] = df.loc[:, 'V49'].sum()

# Merge the NPS and ASJ data
df = pd.merge(nps, asj)
df = df.loc[df.year.isin(range(1984, 2022 + 1, 1)), :]

# Interpolate the missing values in the ASJ
df.loc[df.race == 1, 'incarcerated_asj'] = df.loc[df.race == 1, 'incarcerated_asj'].interpolate().values
df.loc[df.race == 2, 'incarcerated_asj'] = df.loc[df.race == 2, 'incarcerated_asj'].interpolate().values
df.loc[(df.year == 1984) & (df.race == 1), 'incarcerated_asj'] = df.loc[(df.year == 1985) & (df.race == 1), 'incarcerated_asj'].values * df.loc[(df.year == 1984) & (df.race == 1), 'incarcerated_nps'].values / df.loc[(df.year == 1985) & (df.race == 1), 'incarcerated_nps'].values
df.loc[(df.year == 1984) & (df.race == 2), 'incarcerated_asj'] = df.loc[(df.year == 1985) & (df.race == 1), 'incarcerated_asj'].values * df.loc[(df.year == 1984) & (df.race == 2), 'incarcerated_nps'].values / df.loc[(df.year == 1985) & (df.race == 2), 'incarcerated_nps'].values

# Calculate the total number of incarcerated individuals
df.loc[:, 'incarcerated'] = df.loc[:, 'incarcerated_nps'] + df.loc[:, 'incarcerated_asj']
df = df.drop(['incarcerated_nps', 'incarcerated_asj'], axis=1)

# Load the adult population data
pop = pd.read_csv(os.path.join(pop_f_data, 'population.csv'))

# Merge the data and calculate the incarceration rate
df = pd.merge(df, pop)
df = pd.concat([df, expand({'year': range(1984, 2022 + 1, 1), 'race': [-1]})], ignore_index=True)
df.loc[df.race == -1, 'incarcerated'] = df.loc[df.race != -1, :].groupby('year').apply(lambda x: x.incarcerated.sum()).values
df.loc[df.race == -1, 'population'] = df.loc[df.race != -1, :].groupby('year').apply(lambda x: x.population.sum()).values
df.loc[:, 'incarceration_rate'] = df.loc[:, 'incarcerated'] / df.loc[:, 'population']
df = df.drop(['incarcerated', 'population'], axis=1)
df = pd.merge(expand({'year': range(1984, 2022 + 1, 1), 'race': [-1, 1, 2], 'age': range(101)}), df).reset_index(drop=True)
df.loc[(df.age < 18) | (df.age >= 85), 'incarceration_rate'] = 0

# Save the data
df.to_csv(os.path.join(incarceration_f_data, 'incarceration.csv'), index=False)