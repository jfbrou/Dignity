# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as smf
import os

# Set the job index
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

# Import functions
from functions import *

# Set the Sherlock data directory
data = '/scratch/users/jfbrou/Dignity'

# Perform the consumption-equivalent welfare calculations on each bootstrap sample
dignity = pd.read_csv(os.path.join(data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity_bootstrap = pd.read_csv(os.path.join(data, 'dignity_bootstrap.csv'))
c_nominal = 31046.442985362326
def cew(b):
    # Use the data for the consumption-equivalent welfare of Black relative to White Americans calculation
    years = range(1984, 2022 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.race != -1) & (dignity.latin == -1), :]

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
    years = range(2006, 2022 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == 0), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.race != -1) & (dignity.latin == 0), :]

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
    years = range(2006, 2022 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False), :]
    df_survival = dignity.loc[dignity.year.isin(years), :]
    
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
    years = range(1984, 2022 + 1)
    df_bootstrap = dignity_bootstrap.loc[dignity_bootstrap.year.isin(years) & (dignity_bootstrap.bootstrap == b) & (dignity_bootstrap.simple == False) & (dignity_bootstrap.race != -1) & (dignity_bootstrap.latin == -1), :]
    df_survival = dignity.loc[dignity.year.isin(years) & (dignity.race != -1) & (dignity.latin == -1), :]

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

    # Append and save the data frames
    df = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True)
    df.to_csv(os.path.join(data, 'cew_bootstrap_ ' + str(b) + '.csv'), index=False)

# Calculate the consumption-equivalent welfare statistics across 1000 bootstrap samples
samples = range((idx - 1) * 5 + 1, np.minimum(idx * 5, 1000) + 1, 1)
for sample in samples:
    cew(sample)