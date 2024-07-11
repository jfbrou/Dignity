# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
from dotenv import load_dotenv
load_dotenv()
import os

# Import functions and directories
from functions import *
from directories import *

# Start the BEA client
bea = beapy.BEA(key=os.getenv('bea_api_key'))

# Load the survival rates data
survival = pd.read_csv(os.path.join(cdc_f_data, 'survival.csv'))
survival = survival.loc[survival.gender == -1, :].drop('gender', axis=1)

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2020 + 1)), :]
for column in [column for column in cex.columns if column.startswith('consumption')]:
    cex.loc[:, column] = cex.loc[:, column] / np.average(cex.loc[cex.year == 2019, column], weights=cex.loc[cex.year == 2019, 'weight'])

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2021 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Load the census and ACS data
acs = pd.read_csv(os.path.join(acs_f_data, 'acs.csv'))
acs.loc[:, 'consumption'] = acs.consumption / np.average(acs.loc[acs.year == 2019, 'consumption'], weights=acs.loc[acs.year == 2019, 'weight'])

# Calibrate the value of theta
#consumption = 1e5 * (bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC - bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DINSRC) / (bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC * bea.data('nipa', tablename='t20404', frequency='a', year=2006).data.DPCERG)
#cps_theta = cps.loc[(cps.year == 2006) & cps.age.isin(range(25, 55)), :]
#theta = (1 - 0.353) * np.average(cps_theta.earnings, weights=cps_theta.weight) / (consumption * (1 - np.average(cps_theta.leisure, weights=cps_theta.weight))**2)

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
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year, race and age
df = cex.loc[cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year and age for Latinos
df = cex.loc[(cex.latin == 1) & (cex.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [False]}), df, how='left')
df.loc[:, columns] = df.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
df_consumption = df_consumption.append(df, ignore_index=True)

# Calculate CEX consumption statistics by year, race and age for non-Latinos
df = cex.loc[cex.race.isin([1, 2]) & (cex.latin == 0) & (cex.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [False]}), df, how='left')
df.loc[:, columns] = df.groupby(['year', 'race'], as_index=False)[columns].transform(lambda x: filter(x, 1600)).values
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
    d['Ev_of_ell'] = np.average(v_of_ell(x.leisure), weights=x.weight)
    d['ell_bar'] = np.average(x.leisure, weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate CPS leisure statistics by year and age
df = cps.groupby(['year', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby('year', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year, race and age
df = cps.loc[cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year and age for Latinos
df = cps.loc[(cps.latin == 1) & (cps.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby('year', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate CPS leisure statistics by year, race and age for non-Latinos
df = cps.loc[cps.race.isin([1, 2]) & (cps.latin == 0) & (cps.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [False]}), df, how='left')
df.loc[:, ['Ev_of_ell', 'ell_bar']] = df.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year and age
df = acs.groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'ell_bar'] = df.groupby('year', as_index=False)['ell_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year, race and age
df = acs.loc[acs.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [-1], 'historical': [True]}), df, how='left')
df.loc[:, 'ell_bar'] = df.groupby(['year', 'race'], as_index=False)['ell_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year and age for Latinos
df = acs.loc[(acs.latin == 1) & (acs.year >= 2006), :].groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [-1], 'latin': [1], 'historical': [True]}), df, how='left')
df.loc[:, 'ell_bar'] = df.groupby('year', as_index=False)['ell_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_leisure = df_leisure.append(df, ignore_index=True)

# Calculate ACS leisure statistics by year, race and age for non-Latinos
df = acs.loc[acs.race.isin([1, 2]) & (acs.latin == 0) & (acs.year >= 2006), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.leisure, weights=x.weight)}))
df = pd.merge(expand({'year': df.year.unique(), 'age': range(101), 'race': [1, 2], 'latin': [0], 'historical': [True]}), df, how='left')
df.loc[:, 'ell_bar'] = df.groupby(['year', 'race'], as_index=False)['ell_bar'].transform(lambda x: filter(x, 100)).values
df.loc[df.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
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
