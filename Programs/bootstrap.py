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
dignity_bootstrap_historical = pd.DataFrame()
for b in range(1000):
    df_cex = pd.read_csv(os.path.join(data, 'dignity_cex_bootstrap_' + str(b) + '.csv'))
    df_cps = pd.read_csv(os.path.join(data, 'dignity_cps_bootstrap_' + str(b) + '.csv'))
    df_acs = pd.read_csv(os.path.join(data, 'dignity_acs_bootstrap_' + str(b) + '.csv'))
    dignity_cex_bootstrap = dignity_cex_bootstrap.append(df_cex, ignore_index=True)
    dignity_cps_bootstrap = dignity_cps_bootstrap.append(df_cps, ignore_index=True)
    dignity_bootstrap_historical = dignity_bootstrap_historical.append(df_acs, ignore_index=True)
    del df_cex, df_cps
dignity_bootstrap = pd.merge(dignity_cex_bootstrap, dignity_cps_bootstrap, how='left')
dignity_bootstrap.to_csv(os.path.join(data, 'dignity_bootstrap.csv'), index=False)
dignity_bootstrap_historical.to_csv(os.path.join(data, 'dignity_bootstrap_historical.csv'), index=False)

# Perform the consumption-equivalent welfare calculations on each bootstrap sample
dignity = pd.read_csv(os.path.join(data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity_bootstrap = pd.read_csv(os.path.join(data, 'dignity_bootstrap.csv'))
dignity_bootstrap_historical = pd.read_csv(os.path.join(data, 'dignity_bootstrap_historical.csv'))
c_nominal = 31046.442985362326
def cew(b):
    # Use the data for the consumption-equivalent welfare of Black relative to White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the consumption-equivalent welfare of Black relative to White Americans
    df_1 = expand({'year': years, 'log_lambda': [np.nan], 'bootstrap': [b], 'description': ['Welfare']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_intercept.loc[:, 'S'].values
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
        df_1.loc[df_1.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                              S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                              inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == 0), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == 0), :]

    # Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans
    df_2 = expand({'year': years, 'latin': [0], 'log_lambda': [np.nan], 'bootstrap': [b], 'description': ['Welfare ethnicity']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_intercept.loc[:, 'S'].values
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
        df_2.loc[df_2.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                              S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                              inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the consumption-equivalent welfare of Latino relative to White non-Latino Americans calculation
    years = range(2006, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False), :]
    
    # Calculate the consumption-equivalent welfare of Latino relative to White non-Latino Americans
    df_3 = expand({'year': years, 'latin': [1], 'log_lambda': [np.nan], 'bootstrap': [b], 'description': ['Welfare ethnicity']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1) & (df_survival.latin == 0), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == -1) & (df_survival.latin == 1), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'ell_bar'].values
        S_intercept = dignity_intercept.loc[:, 'S'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'c_bar_nd'].values
        c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'c_bar_nd'].values
        Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c'].values
        Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c'].values
        Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Elog_of_c_nd'].values
        Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1) & (df_bootstrap.latin == 0), 'Ev_of_ell'].values
        Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == -1) & (df_bootstrap.latin == 1), 'Ev_of_ell'].values
        df_3.loc[df_3.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                              S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                              inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

    # Use the data for the cumulative welfare growth of Black and White Americans calculation
    years = range(1984, 2019 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the cumulative welfare growth of Black and White Americans
    df_4 = expand({'year': years[1:], 'race': [1, 2], 'log_lambda': [np.nan], 'bootstrap': [b], 'description': ['Cumulative welfare growth']})
    for race in [1, 2]:
        for year in years[1:]:
            S_i = df_survival.loc[(df_survival.year == years[years.index(year) - 1]) & (df_survival.race == race), 'S'].values
            S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == race), 'S'].values
            c_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar'].values
            c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar'].values
            ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'ell_bar'].values
            ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'ell_bar'].values
            T = year - years[years.index(year) - 1]
            S_intercept = dignity_intercept.loc[:, 'S'].values
            c_intercept = dignity_intercept.loc[:, 'c_bar'].values
            ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
            c_i_bar_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar_nd'].values
            c_j_bar_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar_nd'].values
            Elog_of_c_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c'].values
            Elog_of_c_i_nd = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Elog_of_c_j_nd = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Elog_of_c_nd'].values
            Ev_of_ell_i = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            Ev_of_ell_j = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'Ev_of_ell'].values
            df_4.loc[(df_4.year == year) & (df_4.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                           S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                           inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']
    
    # Cumulate the growth rates
    df_4.loc[:, 'log_lambda'] = df_4.groupby('race', as_index=False).log_lambda.transform(lambda x: np.exp(np.cumsum(x))).log_lambda.values

    # Use the data for the historical consumption-equivalent welfare of Black relative to White Americans calculation
    years = list(range(1940, 1990 + 1, 10)) + list(range(2000, 2019 + 1))
    df_bootstrap = dignity_bootstrap_historical.loc[dignity_bootstrap_historical.year.isin(years) & (dignity_bootstrap_historical.bootstrap == b) & (dignity_bootstrap_historical.simple == False) & (dignity_bootstrap_historical.race != -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == True) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the historical consumption-equivalent welfare of Black relative to White Americans
    df_5 = expand({'year': years, 'log_lambda': [np.nan], 'bootstrap': [b], 'description': ['Welfare historical']})
    for year in years:
        S_i = df_survival.loc[(df_survival.year == year) & (df_survival.race == 1), 'S'].values
        S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == 2), 'S'].values
        c_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'c_bar'].values
        c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 1), 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == 2), 'ell_bar'].values
        S_intercept = dignity_intercept.loc[:, 'S'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        df_5.loc[df_5.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                              S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)['log_lambda']

    # Use the data for the historical consumption-equivalent welfare growth by decade for Black and White Americans calculation
    years = list(range(1940, 2010 + 1, 10)) + [2019]
    df_bootstrap = dignity_bootstrap_historical.loc[dignity_bootstrap_historical.year.isin(years) & (dignity_bootstrap_historical.bootstrap == b) & (dignity_bootstrap_historical.simple == False) & (dignity_bootstrap_historical.race != -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == True) & (dignity.race != -1) & (dignity.latin == -1), :]

    # Calculate the historical consumption-equivalent welfare of Black relative to White Americans
    df_6 = expand({'year': years[1:], 'race': [1, 2], 'log_lambda': [np.nan], 'bootstrap': [b], 'description': ['Welfare growth historical']})
    for race in [1, 2]:
        for year in years[1:]:
            S_i = df_survival.loc[(df_survival.year == years[years.index(year) - 1]) & (df_survival.race == race), 'S'].values
            S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == race), 'S'].values
            c_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar'].values
            c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar'].values
            ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'ell_bar'].values
            ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'ell_bar'].values
            T = year - years[years.index(year) - 1]
            S_intercept = dignity_intercept.loc[:, 'S'].values
            c_intercept = dignity_intercept.loc[:, 'c_bar'].values
            ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
            df_6.loc[(df_6.year == year) & (df_6.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                           S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)['log_lambda']

    # Use the data for the historical consumption-equivalent welfare growth and consumption by decade calculation
    years = list(range(1940, 2010 + 1, 10)) + [2019]
    df_bootstrap = dignity_bootstrap_historical.loc[dignity_bootstrap_historical.year.isin(years) & (dignity_bootstrap_historical.bootstrap == b) & (dignity_bootstrap_historical.simple == False) & (dignity_bootstrap_historical.race == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == True) & (dignity.race == -1) & (dignity.latin == -1), :]

    # Calculate the historical consumption-equivalent welfare growth and consumption by decade calculation
    df_7 = expand({'year': years[1:], 'log_lambda': [np.nan], 'C': [np.nan], 'bootstrap': [b], 'description': ['Welfare and consumption growth historical']})
    for year in years[1:]:
        S_i = df_survival.loc[df_survival.year == years[years.index(year) - 1], 'S'].values
        S_j = df_survival.loc[df_survival.year == year, 'S'].values
        c_i_bar = df_bootstrap.loc[df_bootstrap.year == years[years.index(year) - 1], 'c_bar'].values
        c_j_bar = df_bootstrap.loc[df_bootstrap.year == year, 'c_bar'].values
        ell_i_bar = df_bootstrap.loc[df_bootstrap.year == years[years.index(year) - 1], 'ell_bar'].values
        ell_j_bar = df_bootstrap.loc[df_bootstrap.year == year, 'ell_bar'].values
        T = year - years[years.index(year) - 1]
        S_intercept = dignity_intercept.loc[:, 'S'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        for i in ['log_lambda', 'C']:
            df_7.loc[df_7.year == year, i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                        S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)[i]

    # Use the data for the historical consumption-equivalent welfare growth by decade for Black and White Americans calculation
    years = list(range(1940, 2010 + 1, 10)) + [2019]
    df_bootstrap = dignity_bootstrap_historical.loc[dignity_bootstrap_historical.year.isin(years) & (dignity_bootstrap_historical.bootstrap == b) & (dignity_bootstrap_historical.simple == False) & dignity_bootstrap_historical.race.isin([-1, 1, 2]), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.historical == True) & dignity.race.isin([-1, 1, 2]) & (dignity.latin == -1), :]

    # Calculate the historical consumption-equivalent welfare of Black relative to White Americans
    df_8 = expand({'year': years[1:], 'race': [-1, 1, 2], 'log_lambda': [np.nan], 'C': [np.nan], 'bootstrap': [b], 'description': ['Cumulative welfare growth historical']})
    for race in [-1, 1, 2]:
        for year in years[1:]:
            S_i = df_survival.loc[(df_survival.year == years[years.index(year) - 1]) & (df_survival.race == race), 'S'].values
            S_j = df_survival.loc[(df_survival.year == year) & (df_survival.race == race), 'S'].values
            c_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'c_bar'].values
            c_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'c_bar'].values
            ell_i_bar = df_bootstrap.loc[(df_bootstrap.year == years[years.index(year) - 1]) & (df_bootstrap.race == race), 'ell_bar'].values
            ell_j_bar = df_bootstrap.loc[(df_bootstrap.year == year) & (df_bootstrap.race == race), 'ell_bar'].values
            T = year - years[years.index(year) - 1]
            S_intercept = dignity_intercept.loc[:, 'S'].values
            c_intercept = dignity_intercept.loc[:, 'c_bar'].values
            ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
            for i in ['log_lambda', 'C']:
                df_8.loc[(df_8.year == year) & (df_8.race == race), i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                    S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)[i]
    # Cumulate the growth rates
    df_8.loc[:, ['log_lambda', 'C']] = df_8.groupby('race', as_index=False)[['log_lambda', 'C']].transform(lambda x: np.exp(np.cumsum(x * np.diff(years)))).values

    # Append and save the data frames
    df = df_1.append(df_2, ignore_index=True)
    df = df.append(df_3, ignore_index=True)
    df = df.append(df_4, ignore_index=True)
    df = df.append(df_5, ignore_index=True)
    df = df.append(df_6, ignore_index=True)
    df = df.append(df_7, ignore_index=True)
    df = df.append(df_8, ignore_index=True)
    df.to_csv(os.path.join(data, 'cew_bootstrap_ ' + str(b) + '.csv'), index=False)

# Calculate the consumption-equivalent welfare statistics across 1000 bootstrap samples
Parallel(n_jobs=n_cpu)(delayed(cew)(b) for b in range(1000))

# Append all bootstrap samples in a single data frame
cew_bootstrap = pd.DataFrame()
for b in range(1000):
    df = pd.read_csv(os.path.join(data, 'cew_bootstrap_ ' + str(b) + '.csv'))
    cew_bootstrap = cew_bootstrap.append(df, ignore_index=True)
    del df
cew_bootstrap.to_csv(os.path.join(data, 'cew_bootstrap.csv'), index=False)