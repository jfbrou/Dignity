# Import libraries
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Import functions and directories
from functions import *
from directories import *

# Load the survival rates data
survival = pd.read_csv(os.path.join(cdc_f_data, 'survival.csv'))
survival = survival.loc[survival.year.isin(range(1984, 2022 + 1)), :]

# Load the incarceration data
incarceration = pd.read_csv(os.path.join(incarceration_f_data, 'incarceration.csv')).rename(columns={'incarceration_rate': 'I'})

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2022 + 1)), :]
for column in [column for column in cex.columns if column.startswith('consumption')]:
    cex.loc[:, column] = cex.loc[:, column] / np.average(cex.loc[cex.year == 2022, column], weights=cex.loc[cex.year == 2022, 'weight'])

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

################################################################################
#                                                                              #
# This section of the script calculates the consumption-equivalent welfare     #
# consumption statistics for all groups.                                       #
#                                                                              #
################################################################################

# Create a consumption data frame
df_consumption = pd.DataFrame()

# Define a function to perform the CEX aggregation
def f_cex(x):
    d = {}
    columns = [column for column in x.columns if column.startswith('consumption')]
    for column in columns:
        d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]), weights=x.weight)
        d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column], weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Define a list of column names
columns = [column.replace('consumption', 'Elog_of_c') for column in cex.columns if column.startswith('consumption')] + \
          [column.replace('consumption', 'c_bar') for column in cex.columns if column.startswith('consumption')]

# Calculate CEX consumption statistics by year and age
df = cex.groupby(['year', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'region': [-1]}), df, how='left')
df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
df_consumption = pd.concat([df_consumption, df], ignore_index=True)

# Calculate CEX consumption statistics by year, race, and age
df = cex.loc[cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'region': [-1]}), df, how='left')
df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
df_consumption = pd.concat([df_consumption, df], ignore_index=True)

# Calculate CEX consumption statistics by year, race, region, and age
df = cex.loc[cex.race.isin([1, 2]) & cex.region.isin([1, 2]), :].groupby(['year', 'race', 'region', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'region': [1, 2]}), df, how='left')
df.loc[:, columns] = df.groupby(['year', 'race', 'region'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
df_consumption = pd.concat([df_consumption, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script calculates the consumption-equivalent welfare     #
# leisure statistics for all groups.                                           #
#                                                                              #
################################################################################

# Create a leisure data frame
df_leisure = pd.DataFrame()

# Define a function to perform the CPS aggregation
def f_cps(x):
    d = {}
    d['Ev_of_ell'] = np.average(v_of_ell(x.leisure), weights=x.weight)
    d['ell_bar'] = np.average(x.leisure, weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate CPS leisure statistics by year and age
df = cps.groupby(['year', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'region': [-1]}), df, how='left')
df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby('year', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = pd.concat([df_leisure, df], ignore_index=True)

# Calculate CPS leisure statistics by year, race, and age
df = cps.loc[cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'region': [-1]}), df, how='left')
df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = pd.concat([df_leisure, df], ignore_index=True)

# Calculate CPS leisure statistics by year, race, region, and age
df = cps.loc[cps.race.isin([1, 2]) & cps.region.isin([1, 2]), :].groupby(['year', 'race', 'region', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'region': [1, 2]}), df, how='left')
df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race', 'region'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = pd.concat([df_leisure, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script merges the above data frames.                     #
#                                                                              #
################################################################################

# Merge the data frames
dignity = pd.merge(survival, incarceration, how='left')
dignity = pd.merge(dignity, df_consumption, how='left')
dignity = pd.merge(dignity, df_leisure, how='left')
dignity = dignity.sort_values(by=['year', 'race', 'region', 'age'])

# Save the data
dignity.to_csv(os.path.join(f_data, 'dignity.csv'), index=False)