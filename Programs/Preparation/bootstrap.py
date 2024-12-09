# Import libraries
import os
import sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Import functions and directories
sys.path.append(os.path.join(os.getcwd(), 'Preparation'))
from functions import *
from directories import *

# Perform the consumption-equivalent welfare calculations on each bootstrap sample
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1), :]
dignity_cex_bootstrap = pd.read_csv(os.path.join(cex_f_data, 'dignity_cex_bootstrap.csv'))
dignity_cps_bootstrap = pd.read_csv(os.path.join(cps_f_data, 'dignity_cps_bootstrap.csv'))
dignity_bootstrap = pd.merge(dignity_cex_bootstrap, dignity_cps_bootstrap)
c_nominal = 31046.442985362326
def cew(b, dignity, dignity_bootstrap):
    # Use the data for the consumption-equivalent welfare of Black relative to White Americans calculation
    years = range(1984, 2022 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1), :]

    # Calculate the consumption-equivalent welfare of Black relative to White Americans
    df = expand({'year': years, 'log_lambda': [np.nan], 'bootstrap': [b]})
    for year in years:
        S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
        S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
        I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
        I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_intercept.loc[:, 'S'].values
        I_intercept = dignity_intercept.loc[:, 'I'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Ev_of_ell'].values
        df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                          S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Save the data
    df.to_csv(os.path.join(f_data, 'cew_bootstrap_' + str(b) + '.csv'), index=False)

# Calculate the consumption-equivalent welfare statistics across 1000 bootstrap samples
for b in range(1, 1000 + 1):
    cew(b, dignity, dignity_bootstrap)

# Append all bootstrap samples in a single data frame
cew_bootstrap = pd.DataFrame()
for b in range(1, 1000 + 1, 1):
    df = pd.read_csv(os.path.join(f_data, 'cew_bootstrap_' + str(b) + '.csv'))
    cew_bootstrap = pd.concat([cew_bootstrap, df], ignore_index=True)
    os.remove(os.path.join(f_data, 'cew_bootstrap_' + str(b) + '.csv'))
    del df
cew_bootstrap.to_csv(os.path.join(f_data, 'cew_bootstrap.csv'), index=False)