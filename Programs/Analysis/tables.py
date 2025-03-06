# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import os

# Import functions and directories
from functions import *
from directories import *

# Retrieve nominal consumption per capita in 2006
bea_20405 = pd.read_csv(os.path.join(bea_r_data, 'table_20405.csv'), skiprows=[0, 1, 2, 4], skipfooter=5, header=0, engine='python').rename(columns={'Unnamed: 1': 'series'}).iloc[:, 1:]
bea_20405['series'] = bea_20405['series'].str.strip()
bea_20405 = bea_20405.melt(id_vars='series', var_name='year', value_name='value').dropna()
bea_20405 = bea_20405[bea_20405['value'] != '---']
bea_20405['value'] = pd.to_numeric(bea_20405['value'])
bea_20405['year'] = pd.to_numeric(bea_20405['year'])
c_nominal = bea_20405.loc[(bea_20405['series'] == 'Personal consumption expenditures') & (bea_20405['year'] == 2006), 'value'].values
bea_20100 = pd.read_csv(os.path.join(bea_r_data, 'table_20100.csv'), skiprows=[0, 1, 2, 4], header=0).rename(columns={'Unnamed: 1': 'series'}).iloc[:, 1:]
bea_20100['series'] = bea_20100['series'].str.strip()
bea_20100 = bea_20100.melt(id_vars='series', var_name='year', value_name='value').dropna()
bea_20100['year'] = pd.to_numeric(bea_20100['year'])
population = 1e3 * bea_20100.loc[(bea_20100['series'] == 'Population (midperiod, thousands)6') & (bea_20100['year'] == 2006), 'value'].values
c_nominal = 1e6 * c_nominal / population

################################################################################
#                                                                              #
# This section of the script tabulates the consumption-equivalent welfare of   #
# Black relative to White Americans in 1984 and 2022 for different parameter   #
# values and assumptions.                                                      #
#                                                                              #
################################################################################

# Define a list of years
years = [1984, 2022]

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1), :]

# Instantiate an empty data frame
df = expand({'year': years, 'case': ['benchmark', 'beta_and_g', 'age_min_1', 'age_min_5', 'CV', 'EV', 'sqrt', 'high_vsl', 'low_vsl', 'gamma', 'high_frisch', 'low_frisch', 'incarceration', 'unemployment'], 'lambda': [np.nan]})

# Calculate the benchmark consumption-equivalent welfare of Black relative to White Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'benchmark'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans with discounting and growth
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'beta_and_g'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, beta=0.99, g=0.02,
                                                                                       S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                       inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans from age one
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'age_min_1'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, age_min=1,
                                                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans from age five
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'age_min_5'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, age_min=5,
                                                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the CV consumption-equivalent welfare of Black relative to White Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'CV'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                               S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda_CV'])

# Calculate the EV consumption-equivalent welfare of Black relative to White Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'EV'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                               S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda_EV'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans with household consumption divided by the square root of household size
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_sqrt'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_sqrt'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar_sqrt'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd_sqrt'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd_sqrt'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_sqrt'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_sqrt'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd_sqrt'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd_sqrt'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'sqrt'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                 S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                 inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans for a VSL of 10M USD
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'high_vsl'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                     S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal, vsl=10e6,
                                                                                     inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans for a VSL of 5M USD
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'low_vsl'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                    S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal, vsl=5e6,
                                                                                    inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
ell_bar = np.average(cex.loc[(cex.year == 2006) & cex.age.isin(range(25, 55)), 'leisure'], weights=cex.loc[(cex.year == 2006) & cex.age.isin(range(25, 55)), 'weight'])
cex = cex.loc[cex.year.isin(years), :]
gamma = 2.0
c_nominal_gamma = c_nominal * np.average(cex.loc[cex.year == 2022, 'consumption'], weights=cex.loc[cex.year == 2022, 'weight'])**((1 - gamma) / gamma)
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[cex.year == 2022, 'consumption'], weights=cex.loc[cex.year == 2022, 'weight'])

# Define the flow utility function from consumption and leisure
def u(c, ell, gamma=2, epsilon=1, theta=8.851015121158213):
    return c**(1 - gamma) * (1 + (gamma - 1) * theta * epsilon * (1 - ell)**((1 + epsilon) / epsilon) / (1 + epsilon))**gamma / (1 - gamma)

# Calculate CEX consumption statistics by year, race and age
df_gamma = cex.loc[cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'Eu_of_c_and_ell': np.average(u(x.consumption, x.leisure), weights=x.weight)}))
df_gamma = pd.merge(expand({'year': df_gamma.year.unique(), 'age': range(101), 'race': [1, 2]}), df_gamma, how='left')
df_gamma.loc[:, 'Eu_of_c_and_ell'] = df_gamma.groupby(['year', 'race'], as_index=False).Eu_of_c_and_ell.transform(lambda x: filter(x, 1600)).values

# Calculate the consumption-equivalent welfare of Black relative to White Americans for gamma equal to two
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    Eu_of_c_and_ell_i = df_gamma.loc[(df_gamma.year == year) & (df_gamma.race == 1), 'Eu_of_c_and_ell'].values
    Eu_of_c_and_ell_j = df_gamma.loc[(df_gamma.year == year) & (df_gamma.race == 2), 'Eu_of_c_and_ell'].values
    df.loc[(df.year == year) & (df.case == 'gamma'), 'lambda'] = cew_level_gamma(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, Eu_of_c_and_ell_i=Eu_of_c_and_ell_i, Eu_of_c_and_ell_j=Eu_of_c_and_ell_j,
                                                                                 S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal_gamma, ell_bar=ell_bar)['lambda_average']

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Define the leisure utility function
def v_of_ell(x, epsilon=1.0):
    theta = 1.2397887161513956 * (1 - 0.353) / (1 - 0.6989563723839205)**(1 / epsilon + 1)
    return -(theta * epsilon / (1 + epsilon)) * (1 - x)**((1 + epsilon) / epsilon)

# Calculate CPS leisure statistics by year, race and age for the high Frisch elasticity of labor supply
df_high_frisch = cps.loc[cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'Ev_of_ell': np.average(v_of_ell(x.leisure, epsilon=2), weights=x.weight)}))
df_high_frisch = pd.merge(expand({'year': df_high_frisch.year.unique(), 'age': range(101), 'race': [1, 2]}), df_high_frisch, how='left')
df_high_frisch.loc[:, 'Ev_of_ell'] = df_high_frisch.groupby(['year', 'race'], as_index=False).Ev_of_ell.transform(lambda x: filter(x, 100)).values
df_high_frisch.loc[df_high_frisch.Ev_of_ell > 0, 'Ev_of_ell'] = 0

# Calculate the consumption-equivalent welfare of Black relative to White Americans for the high Frisch elasticity of labor supply
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = df_high_frisch.loc[(df_high_frisch.year == year) & (df_high_frisch.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = df_high_frisch.loc[(df_high_frisch.year == year) & (df_high_frisch.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'high_frisch'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, frisch=2,
                                                                                        S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                        inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate CPS leisure statistics by year, race and age for the low Frisch elasticity of labor supply
df_low_frisch = cps.loc[cps.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'Ev_of_ell': np.average(v_of_ell(x.leisure, epsilon=0.5), weights=x.weight)}))
df_low_frisch = pd.merge(expand({'year': df_low_frisch.year.unique(), 'age': range(101), 'race': [1, 2]}), df_low_frisch, how='left')
df_low_frisch.loc[:, 'Ev_of_ell'] = df_low_frisch.groupby(['year', 'race'], as_index=False).Ev_of_ell.transform(lambda x: filter(x, 100)).values
df_low_frisch.loc[df_low_frisch.Ev_of_ell > 0, 'Ev_of_ell'] = 0

# Calculate the consumption-equivalent welfare of Black relative to White Americans for the low Frisch elasticity of labor supply
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = df_low_frisch.loc[(df_low_frisch.year == year) & (df_low_frisch.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = df_low_frisch.loc[(df_low_frisch.year == year) & (df_low_frisch.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'low_frisch'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, frisch=0.5,
                                                                                       S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                       inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2022 + 1)), :]
for column in [column for column in cex.columns if column.startswith('consumption')]:
    cex.loc[:, column] = cex.loc[:, column] / np.average(cex.loc[cex.year == 2022, column], weights=cex.loc[cex.year == 2022, 'weight'])
cex = cex.loc[cex.year.isin([1984, 2006, 2022]) & (cex.education == 1), :]

# Define a function to perform the CEX aggregation
def f_cex(x):
    d = {}
    columns = [column for column in x.columns if column.startswith('consumption')]
    for column in columns:
        d[column.replace('consumption', 'Elog_of_c_I')] = np.average(np.log(x.loc[:, column]), weights=x.weight)
        d[column.replace('consumption', 'c_bar_I')] = np.average(x.loc[:, column], weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Define a list of column names
columns = [column.replace('consumption', 'Elog_of_c_I') for column in cex.columns if column.startswith('consumption')] + \
          [column.replace('consumption', 'c_bar_I') for column in cex.columns if column.startswith('consumption')]

# Calculate CEX consumption statistics by age for individuals with a high school education or less
df_cex_intercept = cex.loc[cex.year == 2006, :].groupby('age', as_index=False).apply(f_cex)
df_cex_intercept = pd.merge(expand({'age': range(101)}), df_cex_intercept, how='left')
df_cex_intercept.loc[:, columns] = df_cex_intercept[columns].transform(lambda x: filter(x, 1600)).values
df_cex = cex.loc[cex.year.isin([1984, 2022]), :].groupby(['year', 'age'], as_index=False).apply(f_cex)
df_cex = pd.merge(expand({'year': [1984, 2022], 'age': range(101)}), df_cex, how='left')
df_cex.loc[:, columns] = df_cex.groupby('year', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin([1984, 2006, 2022]) & (cps.education == 1), :]

# Define a function to perform the CPS aggregation
def f_cps(x):
    d = {}
    d['Ev_of_ell_I'] = np.average(v_of_ell(x.leisure), weights=x.weight)
    d['ell_bar_I'] = np.average(x.leisure, weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate cps leisure statistics by age for individuals with a high school education or less
df_cps_intercept = cps.loc[cps.year == 2006, :].groupby('age', as_index=False).apply(f_cps)
df_cps_intercept = pd.merge(expand({'age': range(101)}), df_cps_intercept, how='left')
df_cps_intercept.loc[:, ['Ev_of_ell_I', 'ell_bar_I']] = df_cps_intercept[['Ev_of_ell_I', 'ell_bar_I']].transform(lambda x: filter(x, 100)).values
df_cps_intercept.loc[df_cps_intercept.loc[:, 'Ev_of_ell_I'] > 0, 'Ev_of_ell_I'] = 0
df_cps_intercept.loc[df_cps_intercept.loc[:, 'ell_bar_I'] > 1, 'ell_bar_I'] = 1
df_cps = cps.loc[cps.year.isin([1984, 2022]), :].groupby(['year', 'age'], as_index=False).apply(f_cps)
df_cps = pd.merge(expand({'year': [1984, 2022], 'age': range(101)}), df_cps, how='left')
df_cps.loc[:, ['Ev_of_ell_I', 'ell_bar_I']] = df_cps.groupby('year', as_index=False)[['Ev_of_ell_I', 'ell_bar_I']].transform(lambda x: filter(x, 100)).values
df_cps.loc[df_cps.loc[:, 'Ev_of_ell_I'] > 0, 'Ev_of_ell_I'] = 0
df_cps.loc[df_cps.loc[:, 'ell_bar_I'] > 1, 'ell_bar_I'] = 1

# Calculate the consumption-equivalent welfare of Black relative to White Americans when incarceration is less costly
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    c_intercept_I = df_cex_intercept.loc[:, 'c_bar_I'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    ell_intercept_I = df_cps_intercept.loc[:, 'ell_bar_I'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    Elog_of_c_I = df_cex.loc[df_cex.year == year, 'Elog_of_c_I'].values
    Ev_of_ell_I = df_cps.loc[df_cps.year == year, 'Ev_of_ell_I'].values
    df.loc[(df.year == year) & (df.case == 'incarceration'), 'lambda'] = np.exp(cew_level_incarceration(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, c_intercept_I=c_intercept_I, ell_intercept=ell_intercept, ell_intercept_I=ell_intercept_I, c_nominal=c_nominal,
                                                                                      c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j,
                                                                                      Elog_of_c_I=Elog_of_c_I, Ev_of_ell_I=Ev_of_ell_I, incarceration_parameter=0.5)['log_lambda'])

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin([1984, 2006, 2022]), :]

# Define a function to perform the CPS aggregation
def f_cps(x):
    d = {}
    d['Ev_of_ell'] = np.average(v_of_ell(x.leisure_half), weights=x.weight)
    d['ell_bar'] = np.average(x.leisure_half, weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate cps leisure statistics by age for individuals with a high school education or less
df_cps_intercept = cps.loc[cps.year == 2006, :].groupby('age', as_index=False).apply(f_cps)
df_cps_intercept = pd.merge(expand({'age': range(101)}), df_cps_intercept, how='left')
df_cps_intercept.loc[:, ['Ev_of_ell', 'ell_bar']] = df_cps_intercept[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df_cps_intercept.loc[df_cps_intercept.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df_cps_intercept.loc[df_cps_intercept.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_cps = cps.loc[cps.year.isin([1984, 2022]) & cps.race.isin([1 , 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(f_cps)
df_cps = pd.merge(expand({'year': [1984, 2022], 'age': range(101), 'race': [1, 2]}), df_cps, how='left')
df_cps.loc[:, ['Ev_of_ell', 'ell_bar']] = df_cps.groupby(['year', 'race'], as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df_cps.loc[df_cps.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df_cps.loc[df_cps.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1

# Calculate the consumption-equivalent welfare of Black relative to White Americans when some unemployment time is considered as leisure
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = df_cps.loc[(df_cps.year == year) & (df_cps.race == 1), 'ell_bar'].values
    ell_j_bar = df_cps.loc[(df_cps.year == year) & (df_cps.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = df_cps_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = df_cps.loc[(df_cps.year == year) & (df_cps.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = df_cps.loc[(df_cps.year == year) & (df_cps.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'unemployment'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Write a table with the robustness results
table = open(os.path.join(tables, 'Robustness.tex'), 'w')
lines = [r'\begin{table}[ht]',
         r'\centering',
         r'\caption{Robustness results}',
         r'\begin{threeparttable}',
         r'\begin{tabular}{l C{3.25cm} C{3.25cm}}',
         r'& \multicolumn{2}{c}{Consumption-equivalent welfare (\%)} \\',
         r'\cmidrule(lr){2-3}',
         r'& 1984 & 2022 \\',
         r'\hline',
         r'Benchmark case & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'benchmark') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'benchmark') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Equivalent variation & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'EV') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'EV') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Compensating variation & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'CV') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'CV') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'$\beta = 0.99$ and $g = 0.02$ & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'beta_and_g') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'beta_and_g') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Household size (square root) & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'sqrt') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'sqrt') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Ages 1 and above & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_1') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_1') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Ages 5 and above & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_5') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_5') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'$\gamma = 2$ & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'gamma') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'gamma') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Frisch elasticity = 0.5 & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_frisch') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_frisch') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Frisch elasticity = 2 & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_frisch') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_frisch') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Value of life = \$5m & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_vsl') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_vsl') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Value of life = \$10m & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_vsl') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_vsl') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Incarceration 50\% utility & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'incarceration') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'incarceration') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'Unemployment 50\% leisure & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'unemployment') & (df.year == 1984), 'lambda'].values[0])) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'unemployment') & (df.year == 2022), 'lambda'].values[0])) + r' \\',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: See the main text for a discussion of the various robustness cases.',
         r'\end{tablenotes}',
         r'\end{threeparttable}',
         r'\label{tab:robust}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

################################################################################
#                                                                              #
# This section of the script tabulates consumption-equivalent welfare growth   #
# for Black and to White Americans from 1984 to 2022.                          #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1), :]

# Calculate consumption-equivalent welfare growth
df = expand({'year': [2022], 'race': [1, 2]})
for column in ['log_lambda', 'LE', 'I', 'C', 'CI', 'L', 'LI']:
    df.loc[:, column] = np.nan
for race in [1, 2]:
    S_i = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'S'].values
    S_j = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'S'].values
    I_i = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'I'].values
    I_j = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'I'].values
    c_i_bar = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == 2022) & (dignity.race == race), 'Ev_of_ell'].values
    T = 2022 - 1984
    for i in ['log_lambda', 'LE', 'I', 'C', 'CI', 'L', 'LI']:
        df.loc[(df.year == 2022) & (df.race == race), i] = cew_growth(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)[i]

# Load the CPS data and compute average income by year and race
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin([1984, 2022]) & cps.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'earnings': np.average(x.earnings, weights=x.weight)}))

# Write a table with the consumption-equivalent welfare growth decomposition
table = open(os.path.join(tables, 'Welfare growth.tex'), 'w')
lines = [r'\begin{table}[ht]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{Welfare growth between 1984 and 2022 (\%)}',
         r'\begin{tabular}{lccccccccc}',
         r'& & & & \multicolumn{6}{c}{\textcolor{ChadBlue}{\it ------ Decomposition ------}} \\',
         r'& Welfare & Earnings & & $LE$ & $I$ & $c$ & $\sigma\left(c\right)$ & $\ell$ & $\sigma\left(\ell\right)$ \\',
         r'\hline',
         r'Black & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 * (np.exp(np.log(cps.loc[(cps.year == 2022) & (cps.race == 2), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 2), 'earnings'].values) / (2022 - 1984)).squeeze() - 1)) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'I'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'CI'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'L'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LI'])) + r' \\',
         r'White & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 *(np.exp(np.log(cps.loc[(cps.year == 2022) & (cps.race == 1), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 1), 'earnings'].values) / (2022 - 1984)).squeeze() - 1)) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'I'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'CI'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'L'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'LI'])) + r' \\',
         r'\hline',
         r'Gap & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'log_lambda']) - 100 * float(df.loc[df.race == 1, 'log_lambda'])) + ' & ' \
                   + '{:.2f}'.format(100 * (np.exp(np.log(cps.loc[(cps.year == 2022) & (cps.race == 2), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 2), 'earnings'].values) / (2022 - 1984)).squeeze() - np.exp(np.log(cps.loc[(cps.year == 2022) & (cps.race == 1), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 1), 'earnings'].values) / (2022 - 1984)).squeeze())) + ' & & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LE']) - 100 * float(df.loc[df.race == 1, 'LE'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'I']) - 100 * float(df.loc[df.race == 1, 'I'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'C']) - 100 * float(df.loc[df.race == 1, 'C'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'CI']) - 100 * float(df.loc[df.race == 1, 'CI'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'L']) - 100 * float(df.loc[df.race == 1, 'L'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LI']) - 100 * float(df.loc[df.race == 1, 'LI'])) + r' \\',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: The last six columns report the additive decomposition in equation~\eqref{eq:lambda}, where $\sigma$ denotes the inequality terms.',
         r'\end{tablenotes}',
         r'\label{tab:Welfare growth}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()