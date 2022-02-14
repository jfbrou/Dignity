# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Load the survival rates data
survival = pd.read_csv(os.path.join(cdc_f_data, 'survival.csv'))
survival = survival.loc[survival.gender == -1, :]

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2020 + 1)), :]
for column in [column for column in cex.columns if column.startswith('consumption') and not column.endswith('cex')]:
    cex.loc[:, column] = cex.loc[:, column] / np.average(cex.loc[cex.year == 2019, column], weights=cex.loc[cex.year == 2019, 'weight'])

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2021 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Load the census and ACS data
acs = pd.read_csv(os.path.join(acs_f_data, 'acs.csv'))
acs.loc[:, 'consumption'] = acs.consumption / np.average(acs.loc[acs.year == 2019, 'consumption'], weights=acs.loc[acs.year == 2019, 'weight'])

################################################################################
#                                                                              #
# This section of the script calculates the consumption-equivalent welfare     #
# survival rate statistics for all groups.                                     #
#                                                                              #
################################################################################

# Create a data frame with all variables for all groups in the survival rates data
survival_recent = survival.loc[survival.year.isin(range(1984, 2020 + 1)), :]
survival_recent.loc[:, 'historical'] = False
survival_historical = survival.loc[survival.year.isin(list(range(1940, 1990 + 1, 10)) + list(range(2000, 2020 + 1))), :]
survival = survival_recent.append(survival_historical, ignore_index=True)
survival.loc[survival.historical.isna(), 'historical'] = True

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
    columns = [column for column in x.columns if column.startswith('consumption') and not column.endswith('cex')]
    for column in columns:
        d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]), weights=x.weight)
        d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column], weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate CEX consumption statistics by year and age
df = cex.groupby(['year', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby('year', as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year, race and age
df = cex.loc[cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby(['year', 'race'], as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year and age for Latinos
df = cex.loc[(cex.latin == 1) & (cex.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby('year', as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year, race and age for non-Latinos
df = cex.loc[cex.race.isin([1, 2]) & (cex.latin == 0) & (cex.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby(['year', 'race'], as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year and age
df = acs.groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'c_bar': np.average(x.consumption, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'c_bar'] = df.groupby('year', as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year, race and age
df = acs.loc[acs.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'c_bar': np.average(x.consumption, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'c_bar'] = df.groupby(['year', 'race'], as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year and age for Latinos
df = acs.loc[(acs.latin == 1) & (acs.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'c_bar': np.average(x.consumption, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [True]}), df, how='left')
df.loc[:, 'c_bar'] = df.groupby('year', as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year, race and age for non-Latinos
df = acs.loc[acs.race.isin([1, 2]) & (acs.latin == 0) & (acs.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'c_bar': np.average(x.consumption, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [True]}), df, how='left')
df.loc[:, 'c_bar'] = df.groupby(['year', 'race'], as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

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
    d['Ev_of_ℓ'] = np.average(v_of_ℓ(x.leisure), weights=x.weight)
    d['ℓ_bar'] = np.average(x.leisure, weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate CPS leisure statistics by year and age
df = cps.groupby(['year', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby('year', as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year, race and age
df = cps.loc[cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year and age for Latinos
df = cps.loc[(cps.latin == 1) & (cps.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby('year', as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year, race and age for non-Latinos
df = cps.loc[cps.race.isin([1, 2]) & (cps.latin == 0) & (cps.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year and age
df = acs.groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'ℓ_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'ℓ_bar'] = df.groupby('year', as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year, race and age
df = acs.loc[acs.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'ℓ_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'ℓ_bar'] = df.groupby(['year', 'race'], as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year and age for Latinos
df = acs.loc[(acs.latin == 1) & (acs.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'ℓ_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [True]}), df, how='left')
df.loc[:, 'ℓ_bar'] = df.groupby('year', as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year, race and age for non-Latinos
df = acs.loc[acs.race.isin([1, 2]) & (acs.latin == 0) & (acs.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'ℓ_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [True]}), df, how='left')
df.loc[:, 'ℓ_bar'] = df.groupby(['year', 'race'], as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script merges the above data frames.                     #
#                                                                              #
################################################################################

# Merge the data frames
dignity = pd.merge(survival, df_consumption, how='left')
dignity = pd.merge(dignity, df_leisure, how='left')

# Save the data
dignity.to_csv(os.path.join(f_data, 'dignity.csv'), index=False)