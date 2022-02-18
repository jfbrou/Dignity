# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from joblib import Parallel, delayed
import os

# Find the number of available CPUs
n_cpu = os.cpu_count()

# Import functions
from functions import *

# Set the Sherlock data directory
data = '/scratch/users/jfbrou/Dignity'

# Append all bootstrap samples in a single data frame
dignity_cex_bootstrap = pd.DataFrame()
dignity_cps_bootstrap = pd.DataFrame()
for n in range(2000):
    b = np.mod(n, 1000)
    m = int(np.floor(n / 1000) + 1)
    df_cex = pd.read_csv(os.path.join(data, 'dignity_cex_bootstrap_' + str(b) + '_method_' + str(m) + '.csv'))
    df_cps = pd.read_csv(os.path.join(data, 'dignity_cps_bootstrap_' + str(b) + '_method_' + str(m) + '.csv'))
    dignity_cex_bootstrap = dignity_cex_bootstrap.append(df_cex, ignore_index=True)
    dignity_cps_bootstrap = dignity_cps_bootstrap.append(df_cps, ignore_index=True)
    del df_cex, df_cps
dignity_bootstrap = pd.merge(dignity_cex_bootstrap, dignity_cps_bootstrap, how='left')
dignity_bootstrap.to_csv(os.path.join(data, 'dignity_bootstrap.csv'), index=False)

# Perform the consumption-equivalent welfare calculations on each bootstrap sample
dignity = pd.read_csv(os.path.join(data, 'dignity.csv'))
dignity_u_bar = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity_bootstrap = pd.read_csv(os.path.join(data, 'dignity_bootstrap.csv'))
c_nominal = 31046.442985362326
def cew(n):
    # Define the bootstrap sample and the sampling method
    b = np.mod(n, 1000)
    m = int(np.floor(n / 1000) + 1)

    # Use the data for the consumption-equivalent welfare of Black relative to White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == m) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the consumption-equivalent welfare of Black relative to White Americans
    df = expand({'year': years, 'race': [2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [m], 'description': ['black']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Ev_of_ell'].values
        df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == m) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == 0), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == 0), :]

    # Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans
    df_black = expand({'year': years, 'race': [2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [m], 'description': ['black non-latino']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'Ev_of_ell'].values
        df_black.loc[df_black.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == m) & (dignity_bootstrap.simple == False), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False), :]
    
    # Calculate the consumption-equivalent welfare of Latino relative to White non-Latino Americans
    df_latin = expand({'year': years, 'race': [-1], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [m], 'description': ['latino']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1) & (df_survival.latin == 0), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == -1) & (df_survival.latin == 1), 'S'].values
        c_bar_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar'].values
        c_bar_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar'].values
        ell_bar_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'ell_bar'].values
        ell_bar_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'ell_bar'].values
        S_intercept = dignity_u_bar.loc[:, 'S'].values
        c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
        ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Ev_of_ell'].values
        df_latin.loc[df_latin.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the cumulative welfare growth of Black and White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.method == m) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the cumulative welfare growth of Black and White Americans
    df_growth = expand({'year': years[1:], 'race': [1, 2], 'log_lambda': [np.nan], 'bootstrap': [b], 'method': [m], 'description': ['growth']})
    for race in [1, 2]:
        for year in years[1:]:
            S_i = df_survival.loc[(df_survival.year == years[years.index(year) - 1]) & (df_survival.race == race), 'S'].values
            S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == race), 'S'].values
            c_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar'].values
            c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar'].values
            ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'ell_bar'].values
            ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'ell_bar'].values
            T = year - years[years.index(year) - 1]
            S_intercept = dignity_u_bar.loc[:, 'S'].values
            c_intercept = dignity_u_bar.loc[:, 'c_bar'].values
            ell_intercept = dignity_u_bar.loc[:, 'ell_bar'].values
            c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar_nd'].values
            c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar_nd'].values
            Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            df_growth.loc[(df_growth.year == year) & (df_growth.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']
    
    # Cumulate the growth rates
    df_growth.loc[:, 'log_lambda'] = df_growth.groupby('race', as_index=False).log_lambda.transform(lambda x: np.exp(np.cumsum(x))).log_lambda.values

    # Append and save the data frames
    df = df.append(df_black, ignore_index=True)
    df = df.append(df_latin, ignore_index=True)
    df = df.append(df_growth, ignore_index=True)
    df.to_csv(os.path.join(data, 'cew_bootstrap_ ' + str(b) + '_method_' + str(m) + '.csv'), index=False)

# Calculate the consumption-equivalent welfare statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(cew)(n) for n in range(2000))

# Append all bootstrap samples in a single data frame
cew_bootstrap = pd.DataFrame()
for n in range(2000):
    b = np.mod(n, 1000)
    m = int(np.floor(n / 1000) + 1)
    df = pd.read_csv(os.path.join(data, 'cew_bootstrap_ ' + str(b) + '_method_' + str(m) + '.csv'))
    cew_bootstrap = cew_bootstrap.append(df, ignore_index=True)
    del df
cew_bootstrap.to_csv(os.path.join(data, 'cew_bootstrap.csv'), index=False)