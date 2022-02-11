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
    cex.loc[:, column] = cex.loc[:, column] / weighted_average(cex.loc[cex.year == 2019, column], data=cex, weights='weight')

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2021 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Load the census and ACS data
acs = pd.read_csv(os.path.join(acs_f_data, 'acs.csv'))
acs.loc[:, 'consumption'] = acs.consumption / weighted_average(acs.loc[acs.year == 2019, 'consumption'], data=acs, weights='weight')

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

# Define dictionaries
columns = [column for column in cex.columns if column.startswith('consumption') and not column.endswith('cex')]
functions_log = [lambda x: weighted_average(np.log(x), data=cex, weights='weight')] * len(columns)
functions = [lambda x: weighted_average(x, data=cex, weights='weight')] * len(columns)
names_log = [column.replace('consumption', 'Elog_of_c') for column in cex.columns if column.startswith('consumption') and not column.endswith('cex')]
names = [column.replace('consumption', 'c_bar') for column in cex.columns if column.startswith('consumption') and not column.endswith('cex')]
d_functions_log = dict(zip(columns, functions_log))
d_names_log = dict(zip(columns, names_log))
d_functions = dict(zip(columns, functions))
d_names = dict(zip(columns, names))

# Calculate CEX consumption statistics by year and age
df = pd.merge(cex.groupby(['year', 'age'], as_index=False).agg(d_functions_log).rename(columns=d_names_log),
              cex.groupby(['year', 'age'], as_index=False).agg(d_functions).rename(columns=d_names), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby('year', as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year, race and age
df = pd.merge(cex.loc[cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).agg(d_functions_log).rename(columns=d_names_log),
              cex.loc[cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).agg(d_functions).rename(columns=d_names), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby(['year', 'race'], as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year and age for Latinos
df = pd.merge(cex.loc[(cex.latin == 1) & (cex.year >= 2006), :].groupby(['year', 'age'], as_index=False).agg(d_functions_log).rename(columns=d_names_log),
              cex.loc[(cex.latin == 1) & (cex.year >= 2006), :].groupby(['year', 'age'], as_index=False).agg(d_functions).rename(columns=d_names), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby('year', as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year, race and age for non-Latinos
df = pd.merge(cex.loc[cex.race.isin([1, 2]) & (cex.latin == 0) & (cex.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).agg(d_functions_log).rename(columns=d_names_log),
              cex.loc[cex.race.isin([1, 2]) & (cex.latin == 0) & (cex.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).agg(d_functions).rename(columns=d_names), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [False]}), df, how='left')
df.loc[:, names_log + names] = df.groupby(['year', 'race'], as_index=False)[names_log + names].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year and age
df = acs.groupby(['year', 'age'], as_index=False).agg({'consumption': lambda x: weighted_average(x, data=acs, weights='weight')}).rename(columns={'consumption': 'c_bar'})
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'c_bar'] = df.groupby('year', as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year, race and age
df = acs.loc[acs.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).agg({'consumption': lambda x: weighted_average(x, data=acs, weights='weight')}).rename(columns={'consumption': 'c_bar'})
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'c_bar'] = df.groupby(['year', 'race'], as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year and age for Latinos
df = acs.loc[(acs.latin == 1) & (acs.year >= 2006), :].groupby(['year', 'age'], as_index=False).agg({'consumption': lambda x: weighted_average(x, data=acs, weights='weight')}).rename(columns={'consumption': 'c_bar'})
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [True]}), df, how='left')
df.loc[:, 'c_bar'] = df.groupby('year', as_index=False).c_bar.transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate ACS consumption statistics by year, race and age for non-Latinos
df = acs.loc[acs.race.isin([1, 2]) & (acs.latin == 0) & (acs.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).agg({'consumption': lambda x: weighted_average(x, data=acs, weights='weight')}).rename(columns={'consumption': 'c_bar'})
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

# Calculate CPS leisure statistics by year and age
df = pd.merge(cps.groupby(['year', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(v_of_ℓ(x), data=cps, weights='weight')}).rename(columns={'leisure': 'Ev_of_ℓ'}),
              cps.groupby(['year', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=cps, weights='weight')}).rename(columns={'leisure': 'ℓ_bar'}), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby('year', as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year, race and age
df = pd.merge(cps.loc[cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(v_of_ℓ(x), data=cps, weights='weight')}).rename(columns={'leisure': 'Ev_of_ℓ'}),
              cps.loc[cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=cps, weights='weight')}).rename(columns={'leisure': 'ℓ_bar'}), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year and age for Latinos
df = pd.merge(cps.loc[(cps.latin == 1) & (cps.year >= 2006), :].groupby(['year', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(v_of_ℓ(x), data=cps, weights='weight')}).rename(columns={'leisure': 'Ev_of_ℓ'}),
              cps.loc[(cps.latin == 1) & (cps.year >= 2006), :].groupby(['year', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=cps, weights='weight')}).rename(columns={'leisure': 'ℓ_bar'}), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby('year', as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year, race and age for non-Latinos
df = pd.merge(cps.loc[cps.race.isin([1, 2]) & (cps.latin == 0) & (cps.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(v_of_ℓ(x), data=cps, weights='weight')}).rename(columns={'leisure': 'Ev_of_ℓ'}),
              cps.loc[cps.race.isin([1, 2]) & (cps.latin == 0) & (cps.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=cps, weights='weight')}).rename(columns={'leisure': 'ℓ_bar'}), how='left')
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ℓ', 'ℓ_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ℓ', 'ℓ_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ℓ'] > 0, 'Ev_of_ℓ'] = 0
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year and age
df = acs.groupby(['year', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=acs, weights='leisure_weight')}).rename(columns={'leisure': 'ℓ_bar'})
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'ℓ_bar'] = df.groupby('year', as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year, race and age
df = acs.loc[acs.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=acs, weights='leisure_weight')}).rename(columns={'leisure': 'ℓ_bar'})
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'ℓ_bar'] = df.groupby(['year', 'race'], as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year and age for Latinos
df = acs.loc[(acs.latin == 1) & (acs.year >= 2006), :].groupby(['year', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=acs, weights='leisure_weight')}).rename(columns={'leisure': 'ℓ_bar'})
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [True]}), df, how='left')
df.loc[:, 'ℓ_bar'] = df.groupby('year', as_index=False)['ℓ_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ℓ_bar'] > 1, 'ℓ_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year, race and age for non-Latinos
df = acs.loc[acs.race.isin([1, 2]) & (acs.latin == 0) & (acs.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).agg({'leisure': lambda x: weighted_average(x, data=acs, weights='leisure_weight')}).rename(columns={'leisure': 'ℓ_bar'})
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
