# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sodapy import Socrata
from dotenv import load_dotenv
load_dotenv()
import os

# Import functions and directories
from functions import *
from directories import *

# Start the BEA client
bea = beapy.BEA(key=os.getenv('bea_api_key'))

################################################################################
#                                                                              #
# This section of the script tabulates the consumption-equivalent welfare of   #
# Black relative to White Americans in 1984 and 2019 for different parameter   #
# values and assumptions.                                                      #
#                                                                              #
################################################################################

# Define a list of years
years = [1984, 2019]

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Instantiate an empty data frame
df = expand({'year': years, 'case': ['benchmark', 'beta_and_g', 'age_min_1', 'age_min_5', 'CV', 'EV', 'sqrt', 'nipa', 'high_vsl', 'low_vsl', 'gamma', 'high_frisch', 'low_frisch'], 'lambda': [np.nan]})

# Calculate the benchmark consumption-equivalent welfare of Black relative to White Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'benchmark'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans with discounting and growth
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'beta_and_g'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, beta=0.99, g=0.02,
                                                                                       S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                       inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans from age one
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'age_min_1'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, age_min=1,
                                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans from age five
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'age_min_5'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, age_min=5,
                                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the CV consumption-equivalent welfare of Black relative to White Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'CV'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                               S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda_CV'])

# Calculate the EV consumption-equivalent welfare of Black relative to White Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'EV'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                               S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda_EV'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans with household consumption divided by the square root of household size
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_sqrt'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_sqrt'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'sqrt'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                 S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                 inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans with consumption re-scaled by NIPA PCE categories
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nipa'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nipa'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar_nipa'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nipa_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nipa_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nipa'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nipa'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nipa_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nipa_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.case == 'nipa'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                 S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                 inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans for a VSL of 10M USD
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'high_vsl'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                     S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal, vsl=10e6,
                                                                                     inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate the consumption-equivalent welfare of Black relative to White Americans for a VSL of 5M USD
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'low_vsl'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                                    S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal, vsl=5e6,
                                                                                    inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
ell_bar = np.average(cex.loc[(cex.year == 2006) & cex.age.isin(range(25, 55)), 'leisure'], weights=cex.loc[(cex.year == 2006) & cex.age.isin(range(25, 55)), 'weight'])
cex = cex.loc[cex.year.isin(years), :]
gamma = 2.0
c_nominal_gamma = c_nominal * np.average(cex.loc[cex.year == 2019, 'consumption'], weights=cex.loc[cex.year == 2019, 'weight'])**((1 - gamma) / gamma)
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[cex.year == 2019, 'consumption'], weights=cex.loc[cex.year == 2019, 'weight'])

# Define the flow utility function from consumption and leisure
def u(c, ell, gamma=2, epsilon=1, theta=14.2):
    return c**(1 - gamma) * (1 + (gamma - 1) * theta * epsilon * (1 - ell)**((1 + epsilon) / epsilon) / (1 + epsilon))**gamma / (1 - gamma)

# Calculate CEX consumption statistics by year, race and age
df_gamma = cex.loc[cex.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'Eu_of_c_and_ell': np.average(u(x.consumption, x.leisure), weights=x.weight)}))
df_gamma = pd.merge(expand({'year': df_gamma.year.unique(), 'age': range(101), 'race': [1, 2]}), df_gamma, how='left')
df_gamma.loc[:, 'Eu_of_c_and_ell'] = df_gamma.groupby(['year', 'race'], as_index=False).Eu_of_c_and_ell.transform(lambda x: filter(x, 1600)).values

# Calculate the consumption-equivalent welfare of Black relative to White Americans for gamma equal to two
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    Eu_of_c_and_ell_i = df_gamma.loc[(df_gamma.year == year) & (df_gamma.race == 1), 'Eu_of_c_and_ell'].values
    Eu_of_c_and_ell_j = df_gamma.loc[(df_gamma.year == year) & (df_gamma.race == 2), 'Eu_of_c_and_ell'].values
    df.loc[(df.year == year) & (df.case == 'gamma'), 'lambda'] = cew_level_gamma(S_i=S_i, S_j=S_j, Eu_of_c_and_ell_i=Eu_of_c_and_ell_i, Eu_of_c_and_ell_j=Eu_of_c_and_ell_j,
                                                                                 S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal_gamma, ell_bar=ell_bar)['lambda_average']

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2021 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Define the leisure utility function
def v_of_ell(x, epsilon=1.0):
    theta = 2.597173415765069 * (1 - 0.353) / (1 - 0.656)**(1 / epsilon + 1)
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
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'high_frisch'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, frisch=2,
                                                                                        S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
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
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    df.loc[(df.year == year) & (df.case == 'low_frisch'), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, frisch=0.5,
                                                                                       S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                       inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Write a table with the robustness results
table = open(os.path.join(tables, 'Robustness.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\caption{Robustness results}',
         r'\begin{threeparttable}',
         r'\begin{tabular}{l C{3.25cm} C{3.25cm}}',
         r'\hline',
         r'\hline',
         r'& \multicolumn{2}{c}{Consumption-equivalent welfare (\%)} \\',
         r'\cmidrule(lr){2-3}',
         r'& 1984 & 2019 \\',
         r'\hline',
         r'Benchmark case & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'benchmark') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'benchmark') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Equivalent variation & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'EV') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'EV') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Compensating variation & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'CV') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'CV') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'$\beta = 0.99$ and $g = 0.02$ & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'beta_and_g') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'beta_and_g') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Household size (square root) & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'sqrt') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'sqrt') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'NIPA PCE categories & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'nipa') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'nipa') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Ages 1 and above & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_1') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_1') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Ages 5 and above & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_5') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'age_min_5') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'$\gamma = 2$ & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'gamma') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'gamma') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Frisch elasticity = 0.5 & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_frisch') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_frisch') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Frisch elasticity = 2 & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_frisch') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_frisch') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Value of life = \$5m & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_vsl') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'low_vsl') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'Value of life = \$10m & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_vsl') & (df.year == 1984), 'lambda'].values)) + ' & ' + '{:.1f}'.format(float(100 * df.loc[(df.case == 'high_vsl') & (df.year == 2019), 'lambda'].values)) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: See main text for discussion of the various robustness cases.',
         r'\end{tablenotes}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

################################################################################
#                                                                              #
# This section of the script tabulates COVID-19 statistics and its associated  #
# consumption-equivalent welfare loss by race.                                 #
#                                                                              #
################################################################################

# Define a list of years
years = [2019, 2020]

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[dignity.historical == False, :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Instantiate an empty data frame
df = expand({'year': years[1:], 'race': [1, 2, -1], 'latin': [0, 1, -1], 'log_lambda': [np.nan], 'LE': [np.nan], 'C': [np.nan], 'L': [np.nan], 'CI': [np.nan], 'LI': [np.nan]})

# Calculate consumption-equivalent welfare growth for all and non-Latinx Black and White Americans
for race in [1, 2]:
    for latin in [0, -1]:
        S_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race) & (dignity.latin == latin), 'S'].values
        S_j = dignity.loc[(dignity.year == year) & (dignity.race == race) & (dignity.latin == latin), 'S'].values
        c_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race) & (dignity.latin == latin), 'c_bar'].values
        c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race) & (dignity.latin == latin), 'c_bar'].values
        ell_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race) & (dignity.latin == latin), 'ell_bar'].values
        ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race) & (dignity.latin == latin), 'ell_bar'].values
        T = year - years[years.index(year) - 1]
        S_intercept = dignity_intercept.loc[:, 'S'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        c_i_bar_nd = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race) & (dignity.latin == latin), 'c_bar_nd'].values
        c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == race) & (dignity.latin == latin), 'c_bar_nd'].values
        Elog_of_c_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race) & (dignity.latin == latin), 'Elog_of_c'].values
        Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == race) & (dignity.latin == latin), 'Elog_of_c'].values
        Elog_of_c_i_nd = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race) & (dignity.latin == latin), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == race) & (dignity.latin == latin), 'Elog_of_c_nd'].values
        Ev_of_ell_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race) & (dignity.latin == latin), 'Ev_of_ell'].values
        Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == race) & (dignity.latin == latin), 'Ev_of_ell'].values
        for i in ['log_lambda', 'LE', 'C', 'L', 'CI', 'LI']:
            df.loc[(df.year == year) & (df.race == race) & (df.latin == latin), i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                                S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                                inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)[i]

# Calculate consumption-equivalent welfare growth for all and Latinx Americans
for latin in [1, -1]:
    S_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == -1) & (dignity.latin == latin), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == latin), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == -1) & (dignity.latin == latin), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == latin), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == -1) & (dignity.latin == latin), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == latin), 'ell_bar'].values
    T = year - years[years.index(year) - 1]
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == -1) & (dignity.latin == latin), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == latin), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == -1) & (dignity.latin == latin), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == latin), 'Ev_of_ell'].values
    for i in ['log_lambda', 'LE', 'C', 'L', 'CI', 'LI']:
        df.loc[(df.year == year) & (df.race == -1) & (df.latin == latin), i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)[i]

# Write a table with the consumption-equivalent welfare growth decomposition
table = open(os.path.join(tables, 'Welfare loss 2020 with caption.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{Welfare loss in 2020 relative to 2019 (\%)}',
         r'\begin{tabular}{lccccccc}',
         r'\hline',
         r'\hline',
         r'& & & \multicolumn{5}{c}{\textcolor{ChadBlue}{\it ------ Decomposition ------}} \\',
         r'& $\lambda$ & $\log\left(\lambda\right)$ & $LE$ & $c$ & $\sigma\left(c\right)$ & $\ell$ & $\sigma\left(\ell\right)$ \\',
         r'\hline',
         r'Black & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 2) & (df.latin == -1), 'log_lambda']))) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'CI'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'L'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'LI'])) + r' \\',
         r'White & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 1) & (df.latin == -1), 'log_lambda']))) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'CI'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'L'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'LI'])) + r' \\',
         r'\hline',
         r'Black non-Latinx & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 2) & (df.latin == 0), 'log_lambda']))) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'log_lambda'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'LE'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'C'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'CI'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'L'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'LI'])) + r' \\',
         r'White non-Latinx & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 1) & (df.latin == 0), 'log_lambda']))) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'log_lambda'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'LE'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'C'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'CI'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'L'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'LI'])) + r' \\',
         r'Latinx & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == -1) & (df.latin == 1), 'log_lambda']))) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'log_lambda'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'LE'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'C'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'CI'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'L'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'LI'])) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: The last five columns report the additive decomposition in equation~\eqref{eq:lambda}, where $\sigma$ denotes the inequality terms.',
         r'\end{tablenotes}',
         r'\label{tab:Welfare loss 2020}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

# Write a table with the consumption-equivalent welfare growth decomposition
table = open(os.path.join(tables, 'Welfare loss 2020.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\begin{tabular}{lccccccc}',
         r'\hline',
         r'\hline',
         r'& & & \multicolumn{5}{c}{\textcolor{ChadBlue}{\it ------ Decomposition ------}} \\',
         r'& $\lambda$ & $\log\left(\lambda\right)$ & $LE$ & $c$ & $\sigma\left(c\right)$ & $\ell$ & $\sigma\left(\ell\right)$ \\',
         r'\hline',
         r'Black & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 2) & (df.latin == -1), 'log_lambda']))) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'CI'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'L'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == -1), 'LI'])) + r' \\',
         r'White & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 1) & (df.latin == -1), 'log_lambda']))) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'CI'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'L'])) + ' & ' \
                     + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == -1), 'LI'])) + r' \\',
         r'\hline',
         r'Black non-Latinx & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 2) & (df.latin == 0), 'log_lambda']))) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'log_lambda'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'LE'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'C'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'CI'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'L'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 2) & (df.latin == 0), 'LI'])) + r' \\',
         r'White non-Latinx & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == 1) & (df.latin == 0), 'log_lambda']))) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'log_lambda'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'LE'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'C'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'CI'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'L'])) + ' & ' \
                                + '{:.2f}'.format(float(df.loc[(df.race == 1) & (df.latin == 0), 'LI'])) + r' \\',
         r'Latinx & ' + '{:.2f}'.format(float(np.exp(df.loc[(df.race == -1) & (df.latin == 1), 'log_lambda']))) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'log_lambda'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'LE'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'C'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'CI'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'L'])) + ' & ' \
                      + '{:.2f}'.format(float(df.loc[(df.race == -1) & (df.latin == 1), 'LI'])) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: The last five columns report the additive decomposition in equation~\eqref{eq:lambda}, where $\sigma$ denotes the inequality terms.',
         r'\end{tablenotes}',
         r'\label{tab:Welfare loss 2020}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

################################################################################
#                                                                              #
# This section of the script tabulates COVID-19 statistics and its associated  #
# consumption-equivalent welfare loss by race.                                 #
#                                                                              #
################################################################################

# Load the COVID-19 data
covid = pd.read_csv(os.path.join(cdc_f_data, 'covid-19.csv'))

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.year == 2019) & (dignity.historical == False) & ((dignity.race.isin([1, 2]) & (dignity.latin == 0)) | (dignity.latin.isin([-1, 1]) & (dignity.race == -1))), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate survival rates adjusted for COVID-19 mortality in 2019
df_covid = dignity.copy()
df_covid.loc[:, 'M'] = df_covid.groupby(['race', 'latin'], as_index=False).S.transform(lambda x: 1 - x.shift(-1) / x).values
df_covid = pd.merge(df_covid, covid, how='left')
df_covid.loc[:, 'M'] = df_covid.M + df_covid.deaths / df_covid.population
df_covid.loc[:, 'S_covid'] = 1 - df_covid.M
df_covid.loc[:, 'S_covid'] = df_covid.groupby(['race', 'latin'], as_index=False).S_covid.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).S_covid.values
df_covid = df_covid.drop(['S', 'M', 'deaths', 'population'], axis=1)
dignity = pd.merge(dignity, df_covid, how='left')

# Initialize a data frame
df = expand({'race': [1, 2], 'latin': [0]}).append(expand({'race': [-1], 'latin': [-1, 1]}), ignore_index=True)
for column in ['lambda', 'deaths', 'age', 'YLL', 'life_expectancy']:
    df.loc[:, column] = np.nan

# Calculate consumption-equivalent welfare by race for non-Latinos
for race in [1, 2]:
    S_i = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'S'].values
    S_j = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'S_covid'].values
    c_i_bar = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.race == race) & (dignity.latin == 0), 'Ev_of_ell'].values
    df.loc[(df.race == race) & (df.latin == 0), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                             S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                             inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Calculate consumption-equivalent welfare for Latinos and all groups
for latin in [-1, 1]:
    S_i = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'S'].values
    S_j = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'S_covid'].values
    c_i_bar = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.race == -1) & (dignity.latin == latin), 'Ev_of_ell'].values
    df.loc[(df.race == -1) & (df.latin == latin), 'lambda'] = np.exp(cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                     S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                     inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda'])

# Compute the number of COVID-19 deaths per capita by race
df_covid = covid.groupby(['race', 'latin'], as_index=False).agg({'deaths':'sum', 'population':'sum'})
df_covid.loc[:, 'deaths'] = 1e3 * df_covid.deaths / df_covid.population
df.loc[df.race == 1, 'deaths'] = df_covid.loc[df_covid.race == 1, 'deaths'].values
df.loc[df.race == 2, 'deaths'] = df_covid.loc[df_covid.race == 2, 'deaths'].values
df.loc[df.latin == -1, 'deaths'] = df_covid.loc[df_covid.latin == -1, 'deaths'].values
df.loc[df.latin == 1, 'deaths'] = df_covid.loc[df_covid.latin == 1, 'deaths'].values

# Compute the average age of COVID-19 deaths by race
df_covid = covid.groupby(['race', 'latin'], as_index=False).apply(lambda x: pd.Series({'age': np.average(x.age, weights=x.deaths)}))
df.loc[df.race == 1, 'age'] = df_covid.loc[df_covid.race == 1, 'age'].values
df.loc[df.race == 2, 'age'] = df_covid.loc[df_covid.race == 2, 'age'].values
df.loc[df.latin == -1, 'age'] = df_covid.loc[df_covid.latin == -1, 'age'].values
df.loc[df.latin == 1, 'age'] = df_covid.loc[df_covid.latin == 1, 'age'].values

# Compute average years of life lost by race for non-Latinos
df_covid = dignity.loc[dignity.latin == 0, :]
for race in [1, 2]:
    for age in range(100):
        df_covid.loc[(df_covid.race == race) & (df_covid.age == age), 'YLL'] = df_covid.loc[(df_covid.race == race) & (df_covid.age >= age), 'S'].sum() / float(df_covid.loc[(df_covid.race == race) & (df_covid.age == age), 'S'])
df_covid = df_covid.dropna(subset=['YLL'])
df_covid = pd.merge(df_covid, covid, how='left')
df_covid = df_covid.groupby('race', as_index=False).apply(lambda x: pd.Series({'YLL': np.average(x.YLL, weights=x.deaths)}))
df.loc[df.race == 1, 'YLL'] = df_covid.loc[df_covid.race == 1, 'YLL'].values
df.loc[df.race == 2, 'YLL'] = df_covid.loc[df_covid.race == 2, 'YLL'].values

# Compute average years of life lost for Latinos and all groups
df_covid = dignity.loc[dignity.race == -1, :]
for latin in [-1, 1]:
    for age in range(100):
        df_covid.loc[(df_covid.latin == latin) & (df_covid.age == age), 'YLL'] = df_covid.loc[(df_covid.latin == latin) & (df_covid.age >= age), 'S'].sum() / float(df_covid.loc[(df_covid.latin == latin) & (df_covid.age == age), 'S'])
df_covid = df_covid.dropna(subset=['YLL'])
df_covid = pd.merge(df_covid, covid, how='left')
df_covid = df_covid.groupby('latin', as_index=False).apply(lambda x: pd.Series({'YLL': np.average(x.YLL, weights=x.deaths)}))
df.loc[df.latin == -1, 'YLL'] = df_covid.loc[df_covid.latin == -1, 'YLL'].values
df.loc[df.latin == 1, 'YLL'] = df_covid.loc[df_covid.latin == 1, 'YLL'].values

# Compute the reduction in life expectancy for all groups
df.loc[df.race == 1, 'life_expectancy'] = dignity.loc[dignity.race == 1, 'S'].sum() - dignity.loc[dignity.race == 1, 'S_covid'].sum()
df.loc[df.race == 2, 'life_expectancy'] = dignity.loc[dignity.race == 2, 'S'].sum() - dignity.loc[dignity.race == 2, 'S_covid'].sum()
df.loc[df.latin == 1, 'life_expectancy'] = dignity.loc[dignity.latin == 1, 'S'].sum() - dignity.loc[dignity.latin == 1, 'S_covid'].sum()
df.loc[df.latin == -1, 'life_expectancy'] = dignity.loc[dignity.latin == -1, 'S'].sum() - dignity.loc[dignity.latin == -1, 'S_covid'].sum()

# Compute the total number of COVID-19 deaths
client = Socrata('data.cdc.gov', os.getenv('cdc_api_key'))
covid = pd.DataFrame.from_records(client.get('m74n-4hbs', limit=500000)).loc[:, ['mmwryear', 'mmwrweek', 'raceethnicity', 'sex', 'agegroup', 'covid19_weighted']]
covid.loc[:, 'day'] = 7 * covid.mmwrweek.astype('int') - 6
covid.loc[:, 'date'] = pd.to_datetime(covid.mmwryear.astype(str), format='%Y') + pd.to_timedelta(covid.day.astype(str) + 'days')
start_date = pd.to_datetime('2020-04-01')
covid = covid.loc[covid.date >= start_date, :].drop(['mmwryear', 'mmwrweek', 'day', 'date'], axis=1)
covid = covid.astype({'covid19_weighted': 'int'})
deaths = covid.loc[(covid.raceethnicity == 'All Race/Ethnicity Groups') & (covid.sex == 'All Sexes') & (covid.agegroup == 'All Ages'), 'covid19_weighted'].sum()

# Write a table with the COVID-19 welfare statistics
table = open(os.path.join(tables, 'Welfare and COVID-19.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\begin{tabular}{lccccc}',
         r'\hline',
         r'\hline',
         r'& \makecell{Deaths per \\ thousand} & \makecell{Age of \\ victims} & \makecell{Years lost \\ per victim} & \makecell{Lower \\ lifespan} & \makecell{Welfare \\ loss (\%)} \\',
         r'\hline',
         r'Black non-Latinx & ' + '{:.2f}'.format(float(df.loc[df.race == 2, 'deaths'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 2, 'age'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 2, 'YLL'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 2, 'life_expectancy'])) + ' & ' \
                                + '{:.1f}'.format(100 * (1 - float(df.loc[df.race == 2, 'lambda']))) + r' \\',
         r'White non-Latinx & ' + '{:.2f}'.format(float(df.loc[df.race == 1, 'deaths'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 1, 'age'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 1, 'YLL'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 1, 'life_expectancy'])) + ' & ' \
                                + '{:.1f}'.format(100 * (1 - float(df.loc[df.race == 1, 'lambda']))) + r' \\',
         r'Latinx & ' + '{:.2f}'.format(float(df.loc[df.latin == 1, 'deaths'])) + ' & ' \
                      + '{:.1f}'.format(float(df.loc[df.latin == 1, 'age'])) + ' & ' \
                      + '{:.1f}'.format(float(df.loc[df.latin == 1, 'YLL'])) + ' & ' \
                      + '{:.1f}'.format(float(df.loc[df.latin == 1, 'life_expectancy'])) + ' & ' \
                      + '{:.1f}'.format(100 * (1 - float(df.loc[df.latin == 1, 'lambda']))) + r' \\',
         r'\hline',
         r'All groups & ' + '{:.2f}'.format(float(df.loc[df.latin == -1, 'deaths'])) + ' & ' \
                          + '{:.1f}'.format(float(df.loc[df.latin == -1, 'age'])) + ' & ' \
                          + '{:.1f}'.format(float(df.loc[df.latin == -1, 'YLL'])) + ' & ' \
                          + '{:.1f}'.format(float(df.loc[df.latin == -1, 'life_expectancy'])) + ' & ' \
                          + '{:.1f}'.format(100 * (1 - float(df.loc[df.latin == -1, 'lambda']))) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: Since April 2020, the CDC reports a total of ' + '{:,.0f}'.format(float(deaths)) + ' COVID-19 deaths.',
         r'\end{tablenotes}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

# Write a table with the COVID-19 welfare statistics
table = open(os.path.join(tables, 'Welfare and COVID-19 with caption.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\caption{Welfare and COVID-19}',
         r'\begin{threeparttable}',
         r'\begin{tabular}{lccccc}',
         r'\hline',
         r'\hline',
         r'& \makecell{Deaths per \\ thousand} & \makecell{Age of \\ victims} & \makecell{Years lost \\ per victim} & \makecell{Lower \\ lifespan} & \makecell{Welfare \\ loss (\%)} \\',
         r'\hline',
         r'Black non-Latinx & ' + '{:.2f}'.format(float(df.loc[df.race == 2, 'deaths'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 2, 'age'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 2, 'YLL'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 2, 'life_expectancy'])) + ' & ' \
                                + '{:.1f}'.format(100 * (1 - float(df.loc[df.race == 2, 'lambda']))) + r' \\',
         r'White non-Latinx & ' + '{:.2f}'.format(float(df.loc[df.race == 1, 'deaths'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 1, 'age'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 1, 'YLL'])) + ' & ' \
                                + '{:.1f}'.format(float(df.loc[df.race == 1, 'life_expectancy'])) + ' & ' \
                                + '{:.1f}'.format(100 * (1 - float(df.loc[df.race == 1, 'lambda']))) + r' \\',
         r'Latinx & ' + '{:.2f}'.format(float(df.loc[df.latin == 1, 'deaths'])) + ' & ' \
                      + '{:.1f}'.format(float(df.loc[df.latin == 1, 'age'])) + ' & ' \
                      + '{:.1f}'.format(float(df.loc[df.latin == 1, 'YLL'])) + ' & ' \
                      + '{:.1f}'.format(float(df.loc[df.latin == 1, 'life_expectancy'])) + ' & ' \
                      + '{:.1f}'.format(100 * (1 - float(df.loc[df.latin == 1, 'lambda']))) + r' \\',
         r'\hline',
         r'All groups & ' + '{:.2f}'.format(float(df.loc[df.latin == -1, 'deaths'])) + ' & ' \
                          + '{:.1f}'.format(float(df.loc[df.latin == -1, 'age'])) + ' & ' \
                          + '{:.1f}'.format(float(df.loc[df.latin == -1, 'YLL'])) + ' & ' \
                          + '{:.1f}'.format(float(df.loc[df.latin == -1, 'life_expectancy'])) + ' & ' \
                          + '{:.1f}'.format(100 * (1 - float(df.loc[df.latin == -1, 'lambda']))) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: Since April 2020, the CDC reports a total of ' + '{:,.0f}'.format(float(deaths)) + ' COVID-19 deaths.',
         r'\end{tablenotes}',
         r'\end{threeparttable}',
         r'\label{tab:Welfare and COVID-19}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

################################################################################
#                                                                              #
# This section of the script tabulates the consumption-equivalent welfare      #
# decomposition of Black relative to White Americans from 1984 to 2019.        #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2019 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate consumption-equivalent welfare
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years)),
                                  'LE':   np.zeros(len(years)),
                                  'C':    np.zeros(len(years)),
                                  'CI':   np.zeros(len(years)),
                                  'L':    np.zeros(len(years)),
                                  'LI':   np.zeros(len(years))})
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
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
    for i in ['log_lambda', 'LE', 'C', 'CI', 'L', 'LI']:
        df.loc[df.year == year, i] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                               S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)[i]

# Write a table with the consumption-equivalent welfare level decomposition
table = open(os.path.join(tables, 'Welfare decompositon with caption.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{Welfare decomposition}',
         r'\begin{tabular}{lccccccc}',
         r'\hline',
         r'\hline',
         r'& & & \multicolumn{5}{c}{\textcolor{ChadBlue}{\it ------ Decomposition ------}} \\',
         r'& $\lambda$ & $\log\left(\lambda\right)$ & $LE$ & $c$ & $\sigma\left(c\right)$ & $\ell$ & $\sigma\left(\ell\right)$ \\',
         r'\hline',
         r'2019 & ' + '{:.2f}'.format(np.exp(df.loc[df.year == 2019, 'log_lambda']).values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'log_lambda'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'LE'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'C'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'CI'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'L'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'LI'].values.squeeze()) + r' \\',
         r'2000 & ' + '{:.2f}'.format(np.exp(df.loc[df.year == 2000, 'log_lambda']).values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'log_lambda'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'LE'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'C'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'CI'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'L'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'LI'].values.squeeze()) + r' \\',
         r'1984 & ' + '{:.2f}'.format(np.exp(df.loc[df.year == 1984, 'log_lambda']).values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'log_lambda'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'LE'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'C'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'CI'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'L'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'LI'].values.squeeze()) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: The last five columns report the additive decomposition in equation~\eqref{eq:lambda}, where $\sigma$ denotes the inequality terms.',
         r'\end{tablenotes}',
         r'\label{tab:Welfare decompositon}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

# Write a table with the consumption-equivalent welfare level decomposition
table = open(os.path.join(tables, 'Welfare decompositon.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\begin{tabular}{lccccccc}',
         r'\hline',
         r'\hline',
         r'& & & \multicolumn{5}{c}{\textcolor{ChadBlue}{\it ------ Decomposition ------}} \\',
         r'& $\lambda$ & $\log\left(\lambda\right)$ & $LE$ & $c$ & $\sigma\left(c\right)$ & $\ell$ & $\sigma\left(\ell\right)$ \\',
         r'\hline',
         r'2019 & ' + '{:.2f}'.format(np.exp(df.loc[df.year == 2019, 'log_lambda']).values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'log_lambda'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'LE'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'C'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'CI'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'L'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2019, 'LI'].values.squeeze()) + r' \\',
         r'2000 & ' + '{:.2f}'.format(np.exp(df.loc[df.year == 2000, 'log_lambda']).values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'log_lambda'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'LE'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'C'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'CI'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'L'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 2000, 'LI'].values.squeeze()) + r' \\',
         r'1984 & ' + '{:.2f}'.format(np.exp(df.loc[df.year == 1984, 'log_lambda']).values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'log_lambda'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'LE'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'C'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'CI'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'L'].values.squeeze()) + ' & ' \
                    + '{:.2f}'.format(df.loc[df.year == 1984, 'LI'].values.squeeze()) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: The last five columns report the additive decomposition in equation~\eqref{eq:lambda}, where $\sigma$ denotes the inequality terms.',
         r'\end{tablenotes}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

################################################################################
#                                                                              #
# This section of the script tabulates consumption-equivalent welfare growth   #
# for Black and to White Americans from 1984 to 2019.                          #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate consumption-equivalent welfare growth
df = expand({'year': [2019], 'race': [1, 2]})
for column in ['log_lambda', 'LE', 'C', 'CI', 'L', 'LI']:
    df.loc[:, column] = np.nan
for race in [1, 2]:
    S_i = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'S'].values
    S_j = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == 1984) & (dignity.race == race), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'Ev_of_ell'].values
    T = 2019 - 1984
    for i in ['log_lambda', 'LE', 'C', 'CI', 'L', 'LI']:
        df.loc[(df.year == 2019) & (df.race == race), i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)[i]

# Load the CPS data and compute average income by year and race
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2020 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin([1984, 2019]) & cps.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'earnings': np.average(x.earnings, weights=x.weight)}))

# Write a table with the consumption-equivalent welfare growth decomposition
table = open(os.path.join(tables, 'Welfare growth with caption.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{Welfare growth between 1984 and 2019 (\%)}',
         r'\begin{tabular}{lcccccccc}',
         r'\hline',
         r'\hline',
         r'& & & & \multicolumn{5}{c}{\textcolor{ChadBlue}{\it ------ Decomposition ------}} \\',
         r'& Welfare & Earnings & & $LE$ & $c$ & $\sigma\left(c\right)$ & $\ell$ & $\sigma\left(\ell\right)$ \\',
         r'\hline',
         r'Black & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 * (np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 2), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 2), 'earnings'].values) / (2019 - 1984)).squeeze() - 1)) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'CI'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'L'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LI'])) + r' \\',
         r'White & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 *(np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 1), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 1), 'earnings'].values) / (2019 - 1984)).squeeze() - 1)) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'CI'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'L'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'LI'])) + r' \\',
         r'\hline',
         r'Gap & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'log_lambda']) - 100 * float(df.loc[df.race == 1, 'log_lambda'])) + ' & ' \
                   + '{:.2f}'.format(100 * (np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 2), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 2), 'earnings'].values) / (2019 - 1984)).squeeze() - np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 1), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 1), 'earnings'].values) / (2019 - 1984)).squeeze())) + ' & & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LE']) - 100 * float(df.loc[df.race == 1, 'LE'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'C']) - 100 * float(df.loc[df.race == 1, 'C'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'CI']) - 100 * float(df.loc[df.race == 1, 'CI'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'L']) - 100 * float(df.loc[df.race == 1, 'L'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LI']) - 100 * float(df.loc[df.race == 1, 'LI'])) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: The last five columns report the additive decomposition in equation~\eqref{eq:lambda}, where $\sigma$ denotes the inequality terms.',
         r'\end{tablenotes}',
         r'\label{tab:Welfare growth}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

# Write a table with the consumption-equivalent welfare growth decomposition
table = open(os.path.join(tables, 'Welfare growth.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\begin{tabular}{lcccccccc}',
         r'\hline',
         r'\hline',
         r'& & & & \multicolumn{5}{c}{\textcolor{ChadBlue}{\it ------ Decomposition ------}} \\',
         r'& Welfare & Earnings & & $LE$ & $c$ & $\sigma\left(c\right)$ & $\ell$ & $\sigma\left(\ell\right)$ \\',
         r'\hline',
         r'Black & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 * (np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 2), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 2), 'earnings'].values) / (2019 - 1984)).squeeze() - 1)) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'CI'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'L'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LI'])) + r' \\',
         r'White & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 *(np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 1), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 1), 'earnings'].values) / (2019 - 1984)).squeeze() - 1)) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'CI'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'L'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[df.race == 1, 'LI'])) + r' \\',
         r'\hline',
         r'Gap & ' + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'log_lambda']) - 100 * float(df.loc[df.race == 1, 'log_lambda'])) + ' & ' \
                   + '{:.2f}'.format(100 * (np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 2), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 2), 'earnings'].values) / (2019 - 1984)).squeeze() - np.exp(np.log(cps.loc[(cps.year == 2019) & (cps.race == 1), 'earnings'].values / cps.loc[(cps.year == 1984) & (cps.race == 1), 'earnings'].values) / (2019 - 1984)).squeeze())) + ' & & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LE']) - 100 * float(df.loc[df.race == 1, 'LE'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'C']) - 100 * float(df.loc[df.race == 1, 'C'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'CI']) - 100 * float(df.loc[df.race == 1, 'CI'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'L']) - 100 * float(df.loc[df.race == 1, 'L'])) + ' & ' \
                   + '{:.2f}'.format(100 * float(df.loc[df.race == 2, 'LI']) - 100 * float(df.loc[df.race == 1, 'LI'])) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: The last five columns report the additive decomposition in equation~\eqref{eq:lambda}, where $\sigma$ denotes the inequality terms.',
         r'\end{tablenotes}',
         r'\label{tab:Welfare growth}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

################################################################################
#                                                                              #
# This section of the script tabulates consumption-equivalent welfare growth   #
# for Black and to White Americans from 1940 to 2019.                          #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == True) & (dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Create a data frame
df = expand({'year': [1980, 2019], 'race': [1, 2]})
for column in ['log_lambda', 'LE', 'C', 'L']:
    df.loc[:, column] = np.nan

# Calculate consumption-equivalent welfare growth
for race in [1, 2]:
    S_i = dignity.loc[(dignity.year == 1940) & (dignity.race == race), 'S'].values
    S_j = dignity.loc[(dignity.year == 1980) & (dignity.race == race), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == 1940) & (dignity.race == race), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == 1980) & (dignity.race == race), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == 1940) & (dignity.race == race), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == 1980) & (dignity.race == race), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    T = 1980 - 1940
    for i in ['log_lambda', 'LE', 'C', 'L']:
        df.loc[(df.year == 1980) & (df.race == race), i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)[i]

# Calculate consumption-equivalent welfare growth
for race in [1, 2]:
    S_i = dignity.loc[(dignity.year == 1940) & (dignity.race == race), 'S'].values
    S_j = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == 1940) & (dignity.race == race), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == 1940) & (dignity.race == race), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == 2019) & (dignity.race == race), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    T = 2019 - 1940
    for i in ['log_lambda', 'LE', 'C', 'L']:
        df.loc[(df.year == 2019) & (df.race == race), i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)[i]

# Write a table with the consumption-equivalent welfare growth decomposition
table = open(os.path.join(tables, 'Welfare growth historical with caption.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{Welfare growth between 1940 and 2019 (\%)}',
         r'\begin{tabular}{lcccccccccc}',
         r'\hline',
         r'\hline',
         r'& & \multicolumn{4}{c}{1940--1980} & & \multicolumn{4}{c}{1940--2019} \\',
         r'\cmidrule(lr){3-6} \cmidrule(lr){8-11}',
         r'& & $\lambda$ & $LE$ & $c$ & $\ell$ & & $\lambda$ & $LE$ & $c$ & $\ell$ \\',
         r'\hline',
         r'Black & & ' + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'L'])) + ' & & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'L'])) + r' \\',
         r'White & & ' + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'L'])) + ' & & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'L'])) + r' \\',
         r'\hline',
         r'Gap & & ' + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'log_lambda']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'LE']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'C']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'L']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'L'])) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'log_lambda']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'LE']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'C']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'L']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'L'])) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: Column $\lambda$ is decomposed in columns $LE$, $c$ and $\ell$.',
         r'\end{tablenotes}',
         r'\label{tab:Welfare growth historical}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

# Write a table with the consumption-equivalent welfare growth decomposition
table = open(os.path.join(tables, 'Welfare growth historical.tex'), 'w')
lines = [r'\begin{table}[H]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\begin{tabular}{lcccccccccc}',
         r'\hline',
         r'\hline',
         r'& & \multicolumn{4}{c}{1940--1980} & & \multicolumn{4}{c}{1940--2019} \\',
         r'\cmidrule(lr){3-6} \cmidrule(lr){8-11}',
         r'& & $\lambda$ & $LE$ & $c$ & $\ell$ & & $\lambda$ & $LE$ & $c$ & $\ell$ \\',
         r'\hline',
         r'Black & & ' + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'L'])) + ' & & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'L'])) + r' \\',
         r'White & & ' + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'L'])) + ' & & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'log_lambda'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'LE'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'C'])) + ' & ' \
                       + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'L'])) + r' \\',
         r'\hline',
         r'Gap & & ' + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'log_lambda']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'LE']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'C']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 1980) & (df.race == 2), 'L']) - 100 * float(df.loc[(df.year == 1980) & (df.race == 1), 'L'])) + ' & & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'log_lambda']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'log_lambda'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'LE']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'LE'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'C']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'C'])) + ' & ' \
                     + '{:.2f}'.format(100 * float(df.loc[(df.year == 2019) & (df.race == 2), 'L']) - 100 * float(df.loc[(df.year == 2019) & (df.race == 1), 'L'])) + r' \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note: Column $\lambda$ is decomposed in columns $LE$, $c$ and $\ell$.',
         r'\end{tablenotes}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# food stamps, medicaid and medicare adjustment of Black relative to White     #
# Americans in 2019.                                                           #
#                                                                              #
################################################################################

# Calculate per capita food stamps, medicaid and medicare expenditures by race in 2019
welfare = pd.read_csv(os.path.join(acs_f_data, 'welfare.csv'))
welfare = welfare.loc[welfare.race.isin([1, 2]), :]
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
population = population.loc[(population.year == 2019) & population.race.isin([1, 2]), :].groupby('race', as_index=False).agg({'population': 'sum'})
welfare = pd.merge(welfare, population)
for i in ['foodstamps', 'medicaid', 'medicare']:
    welfare.loc[:, i] = welfare.loc[:, i] / welfare.population
welfare = welfare.drop('population', axis=1)

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = pd.merge(cex.loc[cex.year == 2019, :], welfare, how='left').dropna(subset=['foodstamps', 'medicaid', 'medicare'])
for column in ['consumption', 'consumption_nd']:
    cex.loc[:, column] = cex.loc[:, column] + cex.foodstamps + cex.medicaid + cex.medicare
    cex.loc[:, column] = cex.loc[:, column] / np.average(cex.loc[:, column], weights=cex.weight)

# Define a function to perform the CEX aggregation
def f_cex(x):
    d = {}
    columns = ['consumption', 'consumption_nd']
    for column in columns:
        d[column.replace('consumption', 'Elog_of_c')] = np.average(np.log(x.loc[:, column]), weights=x.weight)
        d[column.replace('consumption', 'c_bar')] = np.average(x.loc[:, column], weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Define a list of CEX column names
columns = ['Elog_of_c', 'Elog_of_c_nd', 'c_bar', 'c_bar_nd']

# Calculate CEX consumption statistics by race and age
df = cex.loc[cex.race.isin([1, 2]), :].groupby(['race', 'age'], as_index=False).apply(f_cex)
df = pd.merge(expand({'age': range(101), 'race': [1, 2]}), df, how='left')
df.loc[:, columns] = df.groupby('race', as_index=False)[columns].transform(lambda x: filter(x, 1600)).values

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1) & (dignity.year == 2019), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the consumption-equivalent welfare of Black relative to White Americans without the welfare programs adjustment
S_i = dignity.loc[(dignity.race == 1), 'S'].values
S_j = dignity.loc[(dignity.race == 2), 'S'].values
c_i_bar = dignity.loc[(dignity.race == 1), 'c_bar'].values
c_j_bar = dignity.loc[(dignity.race == 2), 'c_bar'].values
ell_i_bar = dignity.loc[(dignity.race == 1), 'ell_bar'].values
ell_j_bar = dignity.loc[(dignity.race == 2), 'ell_bar'].values
S_intercept = dignity_intercept.loc[:, 'S'].values
c_intercept = dignity_intercept.loc[:, 'c_bar'].values
ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
c_i_bar_nd = dignity.loc[(dignity.race == 1), 'c_bar_nd'].values
c_j_bar_nd = dignity.loc[(dignity.race == 2), 'c_bar_nd'].values
Elog_of_c_i = dignity.loc[(dignity.race == 1), 'Elog_of_c'].values
Elog_of_c_j = dignity.loc[(dignity.race == 2), 'Elog_of_c'].values
Elog_of_c_i_nd = dignity.loc[(dignity.race == 1), 'Elog_of_c_nd'].values
Elog_of_c_j_nd = dignity.loc[(dignity.race == 2), 'Elog_of_c_nd'].values
Ev_of_ell_i = dignity.loc[(dignity.race == 1), 'Ev_of_ell'].values
Ev_of_ell_j = dignity.loc[(dignity.race == 2), 'Ev_of_ell'].values
log_lambda = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                       S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                       inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the welfare programs adjustment
S_i = dignity.loc[(dignity.race == 1), 'S'].values
S_j = dignity.loc[(dignity.race == 2), 'S'].values
c_i_bar = df.loc[(df.race == 1), 'c_bar'].values
c_j_bar = df.loc[(df.race == 2), 'c_bar'].values
ell_i_bar = dignity.loc[(dignity.race == 1), 'ell_bar'].values
ell_j_bar = dignity.loc[(dignity.race == 2), 'ell_bar'].values
S_intercept = dignity_intercept.loc[:, 'S'].values
c_intercept = dignity_intercept.loc[:, 'c_bar'].values
ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
c_i_bar_nd = df.loc[(df.race == 1), 'c_bar_nd'].values
c_j_bar_nd = df.loc[(df.race == 2), 'c_bar_nd'].values
Elog_of_c_i = df.loc[(df.race == 1), 'Elog_of_c'].values
Elog_of_c_j = df.loc[(df.race == 2), 'Elog_of_c'].values
Elog_of_c_i_nd = df.loc[(df.race == 1), 'Elog_of_c_nd'].values
Elog_of_c_j_nd = df.loc[(df.race == 2), 'Elog_of_c_nd'].values
Ev_of_ell_i = dignity.loc[(dignity.race == 1), 'Ev_of_ell'].values
Ev_of_ell_j = dignity.loc[(dignity.race == 2), 'Ev_of_ell'].values
log_lambda_welfare = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                               S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']
