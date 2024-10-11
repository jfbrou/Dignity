# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import os

# Import functions and directories
from functions import *
from directories import *

# Set the font for the plots
rc('font', **{'family':'serif', 'serif':['Palatino']})
rc('text', usetex=True)

# Set the first color palette
colors = [(119 / 256, 163 / 256, 48 / 256), # Green
          (0 / 256, 114 / 256, 189 / 256),  # Blue
          (205 / 256, 26 / 256, 51 / 256),  # Red
          (26 / 256, 0 / 256, 51 / 256)]    # Dark blue

# Set the second color palette
newcolors = sns.color_palette('viridis', 5)

# Set the third color palette
newnewcolors = sns.color_palette('viridis', 5)

# Start the BEA client
bea = beapy.BEA(key=bea_api_key)

# Load the bootstrap data from the CEX and CPS
dignity_bs = pd.read_csv(os.path.join(f_data, 'dignity_bootstrap.csv'))

################################################################################
#                                                                              #
# This section of the script plots log average consumption by year for Black   #
# and White Americans from 1984 to 2022.                                       #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2022 + 1)), :]

# Normalize consumption by that of the reference group
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[(cex.year == 2022) & (cex.race == 1), 'consumption'], weights=cex.loc[(cex.year == 2022) & (cex.race == 1), 'weight'])

# Calculate the logarithm of average consumption by year and race
df = cex.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.log(np.average(x.consumption, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2022 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.025)}).rename(columns={'consumption_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.975)}).rename(columns={'consumption_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2022.25, df.loc[df.race == 1, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2022.25, df.loc[df.race == 2, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(np.log(0.29), np.log(1))
ax.set_yticks(np.log(np.linspace(0.3, 1, 8)))
ax.set_yticklabels(np.linspace(30, 100, 8).astype('int'))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average consumption.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the standard deviation of log nondurable    #
# consumption by year for Black and White Americans from 1984 to 2022.         #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2022 + 1)), :]

# Normalize consumption by that of the reference group
cex.loc[:, 'consumption_nd'] = cex.consumption_nd / np.average(cex.loc[(cex.year == 2022) & (cex.race == 1), 'consumption_nd'], weights=cex.loc[(cex.year == 2022) & (cex.race == 1), 'weight'])

# Calculate the standard deviation of log consumption by year and race
df = cex.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption_nd': np.sqrt(np.average((np.log(x.consumption_nd) - np.average(np.log(x.consumption_nd), weights=x.weight))**2, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2022 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_sd': lambda x: x.quantile(0.025)}).rename(columns={'consumption_sd': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_sd': lambda x: x.quantile(0.975)}).rename(columns={'consumption_sd': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots()

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption_nd'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2010, 0.585), color='k', fontsize=12, ha='center', va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption_nd'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2012, 0.50), color='k', fontsize=12, ha='center', va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(0.47, 0.65)
ax.set_yticks(np.linspace(0.5, 0.65, 4))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Standard deviation of consumption.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots average leisure by year for Black and White #
# Americans from 1984 to 2022.                                                 #
#                                                                              #
################################################################################

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate average leisure by year and race
df = cps.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'leisure': np.average(x.leisure, weights=x.weight)}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2022 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.025)}).rename(columns={'leisure_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.975)}).rename(columns={'leisure_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * df.loc[df.race == 1, 'leisure'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 1, 'lb'], y2=100 * df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2022.25, 100 * df.loc[df.race == 1, 'leisure'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'leisure'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 2, 'lb'], y2=100 * df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2022.25, 100 * df.loc[df.race == 2, 'leisure'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(82, 90)
ax.set_yticks(np.linspace(82, 90, 5))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average leisure.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the standard deviation of leisure by year   #
# for Black and White Americans from 1984 to 2022.                             #
#                                                                              #
################################################################################

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate the standard deviation of leisure by year and race
df = cps.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'leisure': np.sqrt(np.average((x.leisure - np.average(x.leisure, weights=x.weight))**2, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2022 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_sd': lambda x: x.quantile(0.025)}).rename(columns={'leisure_sd': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_sd': lambda x: x.quantile(0.975)}).rename(columns={'leisure_sd': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots()

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'leisure'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2004.25, 0.171), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'leisure'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2008.5, 0.154), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(0.14, 0.18)
ax.set_yticks(np.linspace(0.14, 0.18, 5))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Standard deviation of leisure.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the unemployment+ rate of Black and White   #
# Americans from 1984 to 2022.                                                 #
#                                                                              #
################################################################################

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate the unemployment+ rate
df = pd.merge(cps.loc[cps.status == 'unemployed', :].groupby(['year', 'race'], as_index=False).agg({'weight': 'sum'}).rename(columns={'weight': 'unemployed'}),
              cps.loc[cps.laborforce == 1, :].groupby(['year', 'race'], as_index=False).agg({'weight': 'sum'}).rename(columns={'weight': 'laborforce'}), how='left')
df.loc[:, 'unemployed'] = df.unemployed / df.laborforce

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * df.loc[df.race == 1, 'unemployed'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2022.25, 100 * df.loc[df.race == 1, 'unemployed'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'unemployed'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2022.25, 100 * df.loc[df.race == 2, 'unemployed'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(5, 30)
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Unemployment.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots life expectancy by year for Black and White #
# Americans from 1984 to 2022.                                                 #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity = dignity.loc[dignity.year <= 2022, :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.latin == -1) & (dignity.region == -1), :]

# Compute life expectancy by year and race
df = dignity.groupby(['year', 'race'], as_index=False).agg({'S': 'sum'})

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'S'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2022.25, df.loc[df.race == 1, 'S'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'S'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2022.25, df.loc[df.race == 2, 'S'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(69, 80)
ax.set_ylabel('Years', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Life expectancy.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the incarceration rate for Black and White  #
# Americans from 1984 to 2022.                                                 #
#                                                                              #
################################################################################

# Load the incarceration data
df = pd.read_csv(os.path.join(incarceration_f_data, 'incarceration.csv'))
df = df.loc[df.age == 18, :].drop('age', axis=1)

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * df.loc[df.race == 1, 'incarceration_rate'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2022.25, 100 * df.loc[df.race == 1, 'incarceration_rate'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'incarceration_rate'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2022.25, 100 * df.loc[df.race == 2, 'incarceration_rate'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(0, 4)
ax.set_ylabel(r'\%', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Incarceration.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare of Black #
# relative to White Americans from 1984 to 2022.                               #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2022 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.latin == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.latin == -1) & (dignity.region == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the 95% confidence interval
df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
df_bs = df_bs.loc[(df_bs.description == 'Welfare'), :]
df_bs = pd.merge(df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'lb'}),
                 df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'ub'}), how='left')

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years))})
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
    df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, df.log_lambda, color=colors[1], linewidth=2.5)
ax.fill_between(years, df_bs.lb, y2=df_bs.ub, color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[-1])), xy=(2022.25, df.log_lambda.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(np.log(0.35), np.log(0.7))
ax.set_yticks(np.log(np.linspace(0.4, 0.7, 4)))
ax.set_yticklabels(np.round_(np.linspace(0.4, 0.7, 4), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare of Black #
# relative to White Americans by region from 1999 to 2022.                     #
#                                                                              #
################################################################################

# Define a list of years
years = range(1999, 2022 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.latin == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = expand({'year': years, 'region': [-1, 1, 2]})
df.loc[:, 'log_lambda'] = 0
for year in years:
    for region in [-1, 1, 2]:
        S_i = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'S'].values
        S_j = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'S'].values
        I_i = np.zeros(101)
        I_j = np.zeros(101)
        c_i_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'c_bar'].values
        c_j_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'c_bar'].values
        ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'ell_bar'].values
        ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'ell_bar'].values
        S_intercept = dignity_intercept.loc[:, 'S'].values
        I_intercept = np.zeros(101)
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'c_bar_nd'].values
        c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'c_bar_nd'].values
        Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'Elog_of_c'].values
        Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'Elog_of_c'].values
        Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'Elog_of_c_nd'].values
        Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'Ev_of_ell'].values
        Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'Ev_of_ell'].values
        df.loc[(df.year == year) & (df.region == region), 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                        S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                        inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, df.loc[df.region == 1, 'log_lambda'], color=colors[0], linewidth=2.5)
ax.annotate('Non-South', xy=(2021, np.log(0.55)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.loc[df.region == 1, 'log_lambda'].iloc[-1])), xy=(2022.25, df.loc[df.region == 1, 'log_lambda'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(years, df.loc[df.region == 2, 'log_lambda'], color=colors[1], linewidth=2.5)
ax.annotate('South', xy=(2012, np.log(0.705)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.loc[df.region == 2, 'log_lambda'].iloc[-1])), xy=(2022.25, df.loc[df.region == 2, 'log_lambda'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1999, 2022)
ax.set_xticks(np.append(np.linspace(2000, 2020, 5), 2022))
ax.set_xticklabels(np.append(range(2000, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(np.log(0.45), np.log(0.8))
ax.set_yticks(np.log(np.linspace(0.5, 0.8, 4)))
ax.set_yticklabels(np.round_(np.linspace(0.5, 0.8, 4), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare by region.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare,         #
# consumption, earnings and wealth of Black relative to White Americans from   #
# 1984 to 2022.                                                                #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2022 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.latin == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.latin == -1) & (dignity.region == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Compute average consumption by year and race
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(years) & cex.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.average(x.consumption, weights=x.weight)}))
cex = cex.groupby('year', as_index=False).agg({'consumption': lambda x: x.iloc[1] / x.iloc[0]})

# Compute average earnings by year and race
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin(years) & cps.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'earnings': np.average(x.earnings, weights=x.weight)}))
cps = cps.groupby('year', as_index=False).agg({'earnings': lambda x: x.iloc[1] / x.iloc[0]})

# Compute average wealth by year and race from Aladangady and Forde (2021)
scf_data = np.array([[462.59, 81.76],
                     [403.50, 88.88],
                     [423.74, 73.18],
                     [532.16, 100.42],
                     [705.43, 109.96],
                     [756.25, 149.29],
                     [834.94, 165.83],
                     [743.28, 114.38],
                     [747.23, 104.26],
                     [961.44, 149.05],
                     [952.91, 140.52]])

# Calculate the 95% confidence interval
df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
df_bs = df_bs.loc[(df_bs.description == 'Welfare'), :]
df_bs = pd.merge(df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'lb'}),
                 df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'ub'}), how='left')

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years))})
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
    df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

# Plot the lines
ax1.plot(years, df.log_lambda, color=colors[1], linewidth=2.5)
ax1.fill_between(years, df_bs.lb, y2=df_bs.ub, color=colors[1], alpha=0.2, linewidth=0)
ax1.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[-1])), xy=(2022.25, df.log_lambda.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.annotate('Welfare', xy=(2013, np.log(0.615)), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.plot(years, np.log(cex.consumption), color=colors[1], linewidth=2, linestyle='dashed')
ax1.annotate('{0:.2f}'.format(cex.consumption.iloc[-1]), xy=(2022.25, np.log(cex.consumption.iloc[-1])), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.annotate('Consumption', xy=(1995, np.log(0.6)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax1.plot(years, np.log(cps.earnings), color=colors[1], linewidth=2, linestyle='dotted')
ax1.annotate('{0:.2f}'.format(cps.earnings.iloc[-1]), xy=(2022.25, np.log(cps.earnings.iloc[-1])), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.annotate('Earnings', xy=(2011, np.log(0.81)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax2.plot(np.linspace(1989, 2019, 11), np.log(scf_data[:, 1] / scf_data[:, 0]), color=colors[0], linewidth=2, markersize=4, marker='o', clip_on=False)
ax2.annotate('Wealth (right scale)', xy=(2008, np.log(0.125)), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax1.set_xlim(1984, 2022)
ax1.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax1.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axes
ax1.set_ylim(np.log(0.35), np.log(0.9))
ax1.set_yticks(np.log(np.linspace(0.4, 0.9, 6)))
ax1.set_yticklabels(np.round_(np.linspace(0.4, 0.9, 6), 1))
ax1.spines['left'].set_color(colors[1])
ax1.tick_params(axis='y', colors=colors[1])
ax2.set_ylim(np.log(0.115), np.log(0.8))
ax2.set_yticks(np.log(np.linspace(0.15, 0.25, 3)))
ax2.set_yticklabels(list(map('{0:.2f}'.format, np.linspace(0.15, 0.25, 3))))
ax2.spines['right'].set_color(colors[0])
ax2.tick_params(axis='y', colors=colors[0])

# Remove the top and right axes
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare, consumption, earnings, and wealth.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# decomposition of Black relative to White Americans from 1984 to 2022.        #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2022 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.latin == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.latin == -1) & (dignity.region == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = pd.DataFrame({
    'year': years, 
    'LE': np.zeros(len(years)),
    'I':  np.zeros(len(years)),
    'C':  np.zeros(len(years)),
    'CI': np.zeros(len(years)),
    'L':  np.zeros(len(years)),
    'LI': np.zeros(len(years))
})

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
    for i in ['LE', 'I', 'C', 'CI', 'L', 'LI']:
        df.loc[df.year == year, i] = cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                               S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)[i]

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.stackplot(years, [df.LE, df.C, df.I], colors=newcolors[2:], edgecolor='Black', linewidth=0.75)
ax.stackplot(years, [df.L, df.CI + df.LI], colors=[newcolors[1], newcolors[0]], edgecolor='Black', linewidth=0.75)
ax.arrow(1989, np.log(1.01), 0, 0.07, linewidth=1, color='Black')

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(np.log(0.35), np.log(1.1))
ax.set_yticks(np.log(np.linspace(0.4, 1, 4)))
ax.set_yticklabels(np.linspace(0.4, 1, 4))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(1989, np.log(1.11), 'Leisure', fontsize=12, ha='center')
ax.text(2011, np.log(1.1), 'Inequality', fontsize=12, ha='center')
ax.text(1990, np.log(0.78), 'Life expectancy', fontsize=12, ha='center')
ax.text(1990, np.log(0.49), 'Consumption', fontsize=12, ha='center')
ax.text(2004, np.log(0.39), 'Incarceration', fontsize=12, ha='center')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare decomposition.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots cumulative consumption-equivalent welfare   #
# growth for Black and White Americans from 1984 to 2022.                      #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2022 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.latin == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.latin == -1) & (dignity.region == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the 95% confidence interval
df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
df_bs = df_bs.loc[(df_bs.description == 'Cumulative welfare growth'), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'ub'}), how='left')

# Calculate consumption-equivalent welfare growth
df = expand({'year': years[1:], 'race': [1, 2]})
df.loc[:, 'log_lambda'] = np.nan
for race in [1, 2]:
    for year in years[1:]:
        S_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'S'].values
        S_j = dignity.loc[(dignity.year == year) & (dignity.race == race), 'S'].values
        I_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'I'].values
        I_j = dignity.loc[(dignity.year == year) & (dignity.race == race), 'I'].values
        c_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'c_bar'].values
        c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'c_bar'].values
        ell_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'ell_bar'].values
        ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'ell_bar'].values
        T = year - years[years.index(year) - 1]
        S_intercept = dignity_intercept.loc[:, 'S'].values
        I_intercept = dignity_intercept.loc[:, 'I'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        c_i_bar_nd = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'c_bar_nd'].values
        c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == race), 'c_bar_nd'].values
        Elog_of_c_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'Elog_of_c'].values
        Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == race), 'Elog_of_c'].values
        Elog_of_c_i_nd = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'Elog_of_c_nd'].values
        Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == race), 'Elog_of_c_nd'].values
        Ev_of_ell_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'Ev_of_ell'].values
        Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == race), 'Ev_of_ell'].values
        df.loc[(df.year == year) & (df.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                 S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                 inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Cumulate the growth rates
df.loc[:, 'log_lambda'] = df.groupby('race', as_index=False).log_lambda.transform(lambda x: np.exp(np.cumsum(x))).values

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, np.log(np.append(1, df.loc[df.race == 1, 'log_lambda'])), color=colors[0], linewidth=2.5)
ax.annotate(str(np.round_(df.loc[df.race == 1, 'log_lambda'].iloc[-1], decimals=1)) + 'x', xy=(2022.25, np.log(df.loc[df.race == 1, 'log_lambda'].iloc[-1])), color='k', fontsize=10, va='center', annotation_clip=False)
ax.plot(years, np.log(np.append(1, df.loc[df.race == 2, 'log_lambda'])), color=colors[1], linewidth=2.5)
ax.annotate(str(np.round_(df.loc[df.race == 2, 'log_lambda'].iloc[-1], decimals=1)) + 'x', xy=(2022.25, np.log(df.loc[df.race == 2, 'log_lambda'].iloc[-1])), color='k', fontsize=10, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""))

# Set the vertical axis
ax.set_ylim(np.log(1), np.log(4))
ax.set_yticks(np.log(np.linspace(1, 4, 7)))
ax.set_yticklabels(np.linspace(1, 4, 7))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(2012, np.log(2.2), 'White', fontsize=12, ha='center', color='k')
ax.text(2012, np.log(3.3), 'Black', fontsize=12, ha='center', color='k')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Cumulative welfare growth.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption to disposable income ratio  #
# by year for Black and White Americans from 1991 to 2022.                     #
#                                                                              #
################################################################################

# Define a list of years
years = range(1991, 2022 + 1)

# Compute average consumption by year and race
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(years) & cex.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.average(x.consumption, weights=x.weight)}))

# Compute average earnings by year and race
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin(years) & cps.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'income': np.average(x.income, weights=x.weight)}))

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, 100 * cex.loc[cex.race == 1, 'consumption'].values / cps.loc[cps.race == 1, 'income'].values, color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2009.75, 82), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(years, 100 * cex.loc[cex.race == 2, 'consumption'].values / cps.loc[cps.race == 2, 'income'].values, color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2010.75, 98.75), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1991, 2022)
ax.set_xticks(np.append(1991, np.append(np.linspace(1995, 2020, 6), 2022)))
ax.set_xticklabels(np.append("", np.append(range(1995, 2020 + 1, 5), "")))

# Set the vertical axis
ax.set_ylim(75, 100)
ax.set_yticks(np.linspace(75, 100, 6))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Consumption to disposable income ratio.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# incarceration adjustment of Black relative to White Americans in 2019.       #
#                                                                              #
################################################################################

# Load the incarceration data
incarceration = pd.read_csv(os.path.join(incarceration_f_data, 'incarceration.csv'))
incarceration = incarceration.loc[incarceration.year == 2020, :]

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2022 + 1)), :]
for column in [column for column in cex.columns if column.startswith('consumption')]:
    cex.loc[:, column] = cex.loc[:, column] / np.average(cex.loc[cex.year == 2022, column], weights=cex.loc[cex.year == 2022, 'weight'])
cex = cex.loc[cex.year.isin([2006, 2019]) & (cex.education == 1), :]

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
df_cex = cex.loc[cex.year == 2019, :].groupby('age', as_index=False).apply(f_cex)
df_cex = pd.merge(expand({'age': range(101)}), df_cex, how='left')
df_cex.loc[:, columns] = df_cex[columns].transform(lambda x: filter(x, 1600)).values

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2023 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin([2006, 2019]) & (cps.education == 1), :]

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
df_cps = cps.loc[cps.year == 2019, :].groupby('age', as_index=False).apply(f_cps)
df_cps = pd.merge(expand({'age': range(101)}), df_cps, how='left')
df_cps.loc[:, ['Ev_of_ell_I', 'ell_bar_I']] = df_cps[['Ev_of_ell_I', 'ell_bar_I']].transform(lambda x: filter(x, 100)).values
df_cps.loc[df_cps.loc[:, 'Ev_of_ell_I'] > 0, 'Ev_of_ell_I'] = 0
df_cps.loc[df_cps.loc[:, 'ell_bar_I'] > 1, 'ell_bar_I'] = 1