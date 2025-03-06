# Import libraries
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

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

# Load the bootstrap data from the CEX and CPS
dignity_bs = pd.read_csv(os.path.join(f_data, 'dignity_bootstrap.csv'))

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
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2022 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.025)}).rename(columns={'consumption_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.975)}).rename(columns={'consumption_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2022.25, df.loc[df.race == 1, 'consumption'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2022.25, df.loc[df.race == 2, 'consumption'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""), fontsize=16)

# Set the vertical axis
ax.set_ylim(np.log(0.29), np.log(1))
ax.set_yticks(np.log(np.linspace(0.3, 1, 8)))
ax.set_yticklabels(np.linspace(30, 100, 8).astype('int'), fontsize=16)
ax.set_ylabel('$\%$', fontsize=16, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average consumption.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average consumption.eps'), format='eps')
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
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2022 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_sd': lambda x: x.quantile(0.025)}).rename(columns={'consumption_sd': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_sd': lambda x: x.quantile(0.975)}).rename(columns={'consumption_sd': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots()

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption_nd'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2010, 0.585), color='k', fontsize=16, ha='center', va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption_nd'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2012, 0.50), color='k', fontsize=16, ha='center', va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""), fontsize=16)

# Set the vertical axis
ax.set_ylim(0.47, 0.65)
ax.set_yticks(np.linspace(0.5, 0.65, 4))
ax.set_yticklabels(['{:.2f}'.format(x) for x in np.linspace(0.5, 0.65, 4)], fontsize=16)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Standard deviation of consumption.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Standard deviation of consumption.eps'), format='eps')
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
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2022 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.025)}).rename(columns={'leisure_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.975)}).rename(columns={'leisure_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * df.loc[df.race == 1, 'leisure'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 1, 'lb'], y2=100 * df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2022.25, 100 * df.loc[df.race == 1, 'leisure'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'leisure'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 2, 'lb'], y2=100 * df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2022.25, 100 * df.loc[df.race == 2, 'leisure'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""), fontsize=16)

# Set the vertical axis
ax.set_ylim(81, 86)
ax.set_yticks(np.linspace(81, 86, 6))
ax.set_yticklabels(np.linspace(81, 86, 6).astype('int'), fontsize=16)
ax.set_ylabel('$\%$', fontsize=16, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average leisure.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average leisure.eps'), format='eps')
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
ax.annotate('White', xy=(2022.25, 100 * df.loc[df.race == 1, 'unemployed'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'unemployed'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2022.25, 100 * df.loc[df.race == 2, 'unemployed'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""), fontsize=16)

# Set the vertical axis
ax.set_ylim(5, 30)
ax.set_yticks(np.linspace(5, 30, 6))
ax.set_yticklabels(np.linspace(5, 30, 6).astype('int'), fontsize=16)
ax.set_ylabel('$\%$', fontsize=16, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Unemployment.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Unemployment.eps'), format='eps')
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
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1), :]

# Compute life expectancy by year and race
df = dignity.groupby(['year', 'race'], as_index=False).agg({'S': 'sum'})

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'S'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2022.25, df.loc[df.race == 1, 'S'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'S'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2022.25, df.loc[df.race == 2, 'S'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""), fontsize=16)

# Set the vertical axis
ax.set_ylim(69, 80)
ax.set_yticks(np.linspace(70, 80, 6))
ax.set_yticklabels(np.linspace(70, 80, 6).astype('int'), fontsize=16)
ax.set_ylabel('Years', fontsize=16, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Life expectancy.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Life expectancy.eps'), format='eps')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the incarceration rate for Black and White  #
# Americans from 1984 to 2022.                                                 #
#                                                                              #
################################################################################

# Load the incarceration data
df = pd.read_csv(os.path.join(incarceration_f_data, 'incarceration.csv'))
df = df.loc[(df.age == 18) & (df.region == -1), :].drop('age', axis=1)

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * df.loc[df.race == 1, 'incarceration_rate'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2022.25, 100 * df.loc[df.race == 1, 'incarceration_rate'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'incarceration_rate'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2022.25, 100 * df.loc[df.race == 2, 'incarceration_rate'].iloc[-1]), color='k', fontsize=16, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2022)
ax.set_xticks(np.append(np.linspace(1985, 2020, 8), 2022))
ax.set_xticklabels(np.append(range(1985, 2020 + 1, 5), ""), fontsize=16)

# Set the vertical axis
ax.set_ylim(0, 4)
ax.set_yticks(np.linspace(0, 4, 5))
ax.set_yticklabels(np.linspace(0, 4, 5).astype('int'), fontsize=16)
ax.set_ylabel(r'\%', fontsize=16, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Incarceration.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Incarceration.eps'), format='eps')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare of Black #
# relative to White Americans by region from 1999 to 2019.                     #
#                                                                              #
################################################################################

# Define a list of years
years = range(1999, 2019 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[dignity.race != -1, :]

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = expand({'year': years, 'region': [-1, 1, 2]})
df.loc[:, 'log_lambda'] = 0
for year in years:
    for region in [-1, 1, 2]:
        S_i = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'S'].values
        S_j = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'S'].values
        I_i = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'I'].values
        I_j = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'I'].values
        c_i_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'c_bar'].values
        c_j_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'c_bar'].values
        ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 1), 'ell_bar'].values
        ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.region == region) & (dignity.race == 2), 'ell_bar'].values
        S_intercept = dignity_intercept.loc[:, 'S'].values
        I_intercept = dignity_intercept.loc[:, 'I'].values
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
ax.annotate('Non-South', xy=(2016, np.log(0.545)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.loc[df.region == 1, 'log_lambda'].iloc[-1])), xy=(2019.25, df.loc[df.region == 1, 'log_lambda'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(years, df.loc[df.region == 2, 'log_lambda'], color=colors[1], linewidth=2.5)
ax.annotate('South', xy=(2012, np.log(0.645)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.loc[df.region == 2, 'log_lambda'].iloc[-1])), xy=(2019.25, df.loc[df.region == 2, 'log_lambda'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1999, 2019)
ax.set_xticks(np.append(np.linspace(2000, 2015, 4), 2019))
ax.set_xticklabels(np.append(range(2000, 2015 + 1, 5), "2019"))

# Set the vertical axis
ax.set_ylim(np.log(0.4), np.log(0.7))
ax.set_yticks(np.log(np.linspace(0.4, 0.7, 4)))
ax.set_yticklabels(np.round_(np.linspace(0.4, 0.7, 4), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare by region.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare by region.eps'), format='eps')
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
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1), :]

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
ax1.annotate('Welfare', xy=(2013, np.log(0.63)), color='k', fontsize=12, va='center', annotation_clip=False)
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

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare, consumption, earnings, and wealth.eps'), format='eps')
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
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1), :]

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
ax.text(2004, np.log(0.4), 'Incarceration', fontsize=12, ha='center')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare decomposition.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare decomposition.eps'), format='eps')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption and earnings of Black and   #
# White Americans from 1991 to 2022.                                           #
#                                                                              #
################################################################################

# Define a list of years
years = range(1991, 2022 + 1)

# Compute average consumption by year and race
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(years) & cex.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.average(x.consumption, weights=x.weight)}))
cex = cex.groupby('year', as_index=False).agg({'consumption': lambda x: x.iloc[1] / x.iloc[0]})

# Compute average earnings by year and race
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps.loc[:, 'year'] = cps.year - 1
cps = cps.loc[cps.year.isin(years) & cps.race.isin([1, 2]), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({
    'earnings':         np.average(x.earnings, weights=x.weight),
    'earnings_posttax': np.average(x.earnings_posttax, weights=x.weight)
}))
cps = cps.groupby('year', as_index=False).agg({
    'earnings': lambda x: x.iloc[1] / x.iloc[0],
    'earnings_posttax': lambda x: x.iloc[1] / x.iloc[0]
})

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, np.log(cex.consumption), color=colors[0], linewidth=2)
ax.annotate('Consumption', xy=(2011, np.log(0.8)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.plot(years, np.log(cps.earnings), color=colors[1], linewidth=2)
ax.annotate('Earnings', xy=(2010, np.log(0.66)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.plot(years, np.log(cps.earnings_posttax), color=colors[2], linewidth=2)
ax.annotate('Post-tax-and-transfer earnings', xy=(1999, np.log(0.78)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1991, 2022)
ax.set_xticks(np.append(np.linspace(1995, 2020, 6), 2022))
ax.set_xticklabels(np.append(range(1995, 2020 + 1, 5), ""))

# Set the vertical axes
ax.set_ylim(np.log(0.6), np.log(0.9))
ax.set_yticks(np.log(np.linspace(0.6, 0.9, 4)))
ax.set_yticklabels(np.round_(np.linspace(0.6, 0.9, 4), 1))

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Consumption and earnings.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Consumption and earnings.eps'), format='eps')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# health adjustment of Black relative to White Americans in 2018.              #
#                                                                              #
################################################################################

# Load the NHIS data and calculate the average HALex by year, race, and age
nhis = pd.read_csv(os.path.join(nhis_f_data, 'nhis.csv')).dropna(subset=['halex'])
nhis_intercept = nhis.loc[nhis.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = nhis.loc[(nhis.year == 2018) & nhis.race.isin([1, 2]), :].groupby(['race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis_intercept = pd.merge(expand({'age': range(101)}), nhis_intercept, how='left')
nhis = pd.merge(expand({'race': [1, 2], 'age': range(101)}), nhis, how='left')
nhis_intercept.loc[:, 'halex'] = nhis_intercept.halex.transform(lambda x: filter(x, 100)).values
nhis.loc[:, 'halex'] = nhis.groupby('race', as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhis_intercept.loc[nhis_intercept.halex < 0, 'halex'] = 0
nhis_intercept.loc[nhis_intercept.halex > 1, 'halex'] = 1
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1) & (dignity.year == 2018), :]
dignity_intercept = pd.merge(dignity_intercept, nhis_intercept, how='left')
dignity = pd.merge(dignity, nhis, how='left')

# Calculate the consumption-equivalent welfare of Black relative to White Americans without the morbidity adjustment
S_i = dignity.loc[dignity.race == 1, 'S'].values
S_j = dignity.loc[dignity.race == 2, 'S'].values
I_i = dignity.loc[dignity.race == 1, 'I'].values
I_j = dignity.loc[dignity.race == 2, 'I'].values
c_i_bar = dignity.loc[dignity.race == 1, 'c_bar'].values
c_j_bar = dignity.loc[dignity.race == 2, 'c_bar'].values
ell_i_bar = dignity.loc[dignity.race == 1, 'ell_bar'].values
ell_j_bar = dignity.loc[dignity.race == 2, 'ell_bar'].values
S_intercept = dignity_intercept.loc[:, 'S'].values
I_intercept = dignity_intercept.loc[:, 'I'].values
c_intercept = dignity_intercept.loc[:, 'c_bar'].values
ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
c_i_bar_nd = dignity.loc[dignity.race == 1, 'c_bar_nd'].values
c_j_bar_nd = dignity.loc[dignity.race == 2, 'c_bar_nd'].values
Elog_of_c_i = dignity.loc[dignity.race == 1, 'Elog_of_c'].values
Elog_of_c_j = dignity.loc[dignity.race == 2, 'Elog_of_c'].values
Elog_of_c_i_nd = dignity.loc[dignity.race == 1, 'Elog_of_c_nd'].values
Elog_of_c_j_nd = dignity.loc[dignity.race == 2, 'Elog_of_c_nd'].values
Ev_of_ell_i = dignity.loc[dignity.race == 1, 'Ev_of_ell'].values
Ev_of_ell_j = dignity.loc[dignity.race == 2, 'Ev_of_ell'].values
log_lambda = cew_level(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                       S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                       inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the morbidity adjustment
df = pd.DataFrame({'parameter': np.linspace(0, 1, 101)})
df.loc[:, 'log_lambda'] = np.nan
for i in np.linspace(0, 1, 101):
    S_i = dignity.loc[dignity.race == 1, 'S'].values
    S_j = dignity.loc[dignity.race == 2, 'S'].values
    I_i = dignity.loc[dignity.race == 1, 'I'].values
    I_j = dignity.loc[dignity.race == 2, 'I'].values
    H_i = dignity.loc[dignity.race == 1, 'halex'].values
    H_j = dignity.loc[dignity.race == 2, 'halex'].values
    c_i_bar = dignity.loc[dignity.race == 1, 'c_bar'].values
    c_j_bar = dignity.loc[dignity.race == 2, 'c_bar'].values
    ell_i_bar = dignity.loc[dignity.race == 1, 'ell_bar'].values
    ell_j_bar = dignity.loc[dignity.race == 2, 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    H_intercept = dignity_intercept.loc[:, 'halex'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[dignity.race == 1, 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[dignity.race == 2, 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[dignity.race == 1, 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[dignity.race == 2, 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[dignity.race == 1, 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[dignity.race == 2, 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[dignity.race == 1, 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[dignity.race == 2, 'Ev_of_ell'].values
    df.loc[df.parameter == i, 'log_lambda'] = cew_level_morbidity(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, H_i=H_i, H_j=H_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                  S_intercept=S_intercept, I_intercept=I_intercept, H_intercept=H_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                  c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j, morbidity_parameter=i)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(np.linspace(0, 1, 101), df.log_lambda, color=colors[1], linewidth=2.5)
ax.plot(0.1, float(df.loc[df.parameter == 0.1, 'log_lambda']), color=colors[1], marker='o', markersize=8)

# Set the horizontal axis
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xticklabels(np.linspace(0, 100, 11).astype('int'))
ax.set_xlabel(r'Worst morbidity (\%)', fontsize=12, rotation=0, ha='center')

# Set the vertical axis
ax.set_ylim(np.log(0.35), np.log(0.6))
ax.set_yticks(np.log(np.linspace(0.35, 0.6, 6)))
ax.set_yticklabels(list(map('{0:.2f}'.format, np.linspace(0.35, 0.6, 6))))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity sensitivity.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity sensitivity.eps'), format='eps')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# decomposition of Black relative to White Americans from 1997 to 2018 with    #
# the health adjustment.                                                       #
#                                                                              #
################################################################################

# Define a list of years
years = range(1997, 2018 + 1)

# Load the NHIS data and calculate the average HALex by year, race and age
nhis = pd.read_csv(os.path.join(nhis_f_data, 'nhis.csv')).dropna(subset=['halex'])
nhis_intercept = nhis.loc[nhis.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = nhis.loc[nhis.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis_intercept = pd.merge(expand({'age': range(101)}), nhis_intercept, how='left')
nhis = pd.merge(expand({'year': years, 'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhis_intercept.loc[:, 'halex'] = nhis_intercept.halex.transform(lambda x: filter(x, 100)).values
nhis.loc[:, 'halex'] = nhis.groupby(['year', 'race'], as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhis_intercept.loc[nhis_intercept.halex < 0, 'halex'] = 0
nhis_intercept.loc[nhis_intercept.halex > 1, 'halex'] = 1
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1) & dignity.year.isin(years), :]
dignity_intercept = pd.merge(dignity_intercept, nhis_intercept, how='left')
dignity = pd.merge(dignity, nhis, how='left')

# Create a data frame
df = pd.DataFrame({
    'year': years, 
    'LE': np.zeros(len(years)),
    'I': np.zeros(len(years)),
    'M':  np.zeros(len(years)),
    'C':  np.zeros(len(years)),
    'CI': np.zeros(len(years)),
    'L':  np.zeros(len(years)),
    'LI': np.zeros(len(years))
})

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the health adjustment
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    H_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'halex'].values
    H_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'halex'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    H_intercept = dignity_intercept.loc[:, 'halex'].values
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
    for i in ['LE', 'I', 'M', 'C', 'CI', 'L', 'LI']:
        df.loc[df.year == year, i] = cew_level_morbidity(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, H_i=H_i, H_j=H_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                         S_intercept=S_intercept, I_intercept=I_intercept, H_intercept=H_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                         c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j, morbidity_parameter=0.1)[i]


# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))
newnewnewcolors = sns.color_palette('viridis', 6)

# Plot the lines
ax.stackplot(years, [df.LE, df.C, df.I, df.M], colors=newnewnewcolors[2:], edgecolor='Black', linewidth=0.75)
ax.stackplot(years, [df.L, df.CI + df.LI], colors=[newnewnewcolors[1], newnewnewcolors[0]], edgecolor='Black', linewidth=0.75)
ax.arrow(1999, np.log(1.01), 0, 0.09, linewidth=1, color='Black')
ax.arrow(2013.5, np.log(0.49), 0, 0.09, linewidth=1, color='Black')

# Set the horizontal axis
ax.set_xlim(1997, 2018)
ax.set_xticks(np.linspace(1998, 2018, 6))

# Set the vertical axis
ax.set_ylim(np.log(0.2), np.log(1.12))
ax.set_yticks(np.log(np.linspace(0.2, 1, 9)))
ax.set_yticklabels(list(map('{0:.1f}'.format, np.linspace(0.2, 1, 9))))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(1999, np.log(1.14), 'Leisure', fontsize=12, ha='center')
ax.text(2011, np.log(1.11), 'Inequality', fontsize=12, ha='center')
ax.text(2001, np.log(0.76), 'Life expectancy', fontsize=12, ha='center')
ax.text(2001, np.log(0.53), 'Consumption', fontsize=12, ha='center')
ax.text(2013.5, np.log(0.46), 'Incarceration', fontsize=12, ha='center')
ax.text(2001, np.log(0.32), 'Morbidity', fontsize=12, ha='center')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity decomposition.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity decomposition.eps'), format='eps')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# decomposition of Black relative to White Americans from 1997 to 2018 with    #
# the health adjustment.                                                       #
#                                                                              #
################################################################################

# Define a list of years
years = range(1997, 2018 + 1)

# Load the NHIS data and calculate the average HALex by year, race and age
nhis = pd.read_csv(os.path.join(nhis_f_data, 'nhis.csv')).dropna(subset=['halex'])
nhis_intercept = nhis.loc[nhis.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = nhis.loc[nhis.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis_intercept = pd.merge(expand({'age': range(101)}), nhis_intercept, how='left')
nhis = pd.merge(expand({'year': years, 'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhis_intercept.loc[:, 'halex'] = nhis_intercept.halex.transform(lambda x: filter(x, 100)).values
nhis.loc[:, 'halex'] = nhis.groupby(['year', 'race'], as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhis_intercept.loc[nhis_intercept.halex < 0, 'halex'] = 0
nhis_intercept.loc[nhis_intercept.halex > 1, 'halex'] = 1
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.race == -1) & (dignity.region == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.region == -1) & dignity.year.isin(years), :]
dignity_intercept = pd.merge(dignity_intercept, nhis_intercept, how='left')
dignity = pd.merge(dignity, nhis, how='left')

# Create a data frame
df = pd.DataFrame({
    'year':                 years, 
    'log_lambda':           np.zeros(len(years)), 
    'log_lambda_morbidity': np.zeros(len(years))
})

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the health adjustment
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    I_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'I'].values
    I_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'I'].values
    H_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'halex'].values
    H_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'halex'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = dignity_intercept.loc[:, 'I'].values
    H_intercept = dignity_intercept.loc[:, 'halex'].values
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
    df.loc[df.year == year, 'log_lambda'] = cew_level_morbidity(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, H_i=H_i, H_j=H_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                S_intercept=S_intercept, I_intercept=I_intercept, H_intercept=H_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j, morbidity_parameter=1)['log_lambda']
    df.loc[df.year == year, 'log_lambda_morbidity'] = cew_level_morbidity(S_i=S_i, S_j=S_j, I_i=I_i, I_j=I_j, H_i=H_i, H_j=H_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, I_intercept=I_intercept, H_intercept=H_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j, morbidity_parameter=0.1)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, df.log_lambda, color=colors[1], linewidth=2.5)
ax.plot(years, df.log_lambda_morbidity, color=colors[1], linewidth=2.5, linestyle='--')
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[-1])), xy=(2018.25, df.log_lambda.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[0])), xy=(1995.5, df.log_lambda.iloc[0]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda_morbidity.iloc[-1])), xy=(2018.25, df.log_lambda_morbidity.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda_morbidity.iloc[0])), xy=(1995.5, df.log_lambda_morbidity.iloc[0]), color='k', fontsize=12, va='center', annotation_clip=False)

# Add annotations for the "baseline" and "morbidity" labels
ax.text(2004, np.log(0.41), 'Baseline', fontsize=12, color='k', va='center', ha='center')
ax.text(2005, np.log(0.26), 'Morbidity-adjusted', fontsize=12, color='k', va='center', ha='center')

# Set the horizontal axis
ax.set_xlim(1997, 2018)
ax.set_xticks(np.linspace(1998, 2018, 6))

# Set the vertical axis
ax.set_ylim(np.log(0.2), np.log(0.7))
ax.set_yticks(np.log(np.linspace(0.2, 0.7, 6)))
ax.set_yticklabels(np.round_(np.linspace(0.2, 0.7, 6), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity.pdf'), format='pdf')
plt.close()

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity.eps'), format='eps')
plt.close()