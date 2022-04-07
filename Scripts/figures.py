# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()
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
newcolors = sns.color_palette('viridis', 4)

# Set the third color palette
newnewcolors = sns.color_palette('viridis', 5)

# Start the BEA client
bea = beapy.BEA(key=os.getenv('bea_api_key'))

# Load the bootstrap data from the CEX, CPS and ACS
dignity_bs = pd.read_csv(os.path.join(f_data, 'dignity_bootstrap.csv'))
dignity_bs_historical = pd.read_csv(os.path.join(f_data, 'dignity_bootstrap_historical.csv'))

################################################################################
#                                                                              #
# This section of the script plots log average consumption by year for Black   #
# and White Americans from 1984 to 2019.                                       #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2019 + 1)), :]

# Normalize consumption by that of the reference group
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[(cex.year == 2019) & (cex.race == 1), 'consumption'], weights=cex.loc[(cex.year == 2019) & (cex.race == 1), 'weight'])

# Calculate the logarithm of average consumption by year and race
df = cex.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.log(np.average(x.consumption, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.025)}).rename(columns={'consumption_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.975)}).rename(columns={'consumption_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2019.25, df.loc[df.race == 1, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2019.25, df.loc[df.race == 2, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.3), np.log(1))
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
# This section of the script plots log average consumption by year for Black   #
# and White Americans from 1984 to 2019.                                       #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2019 + 1)), :]

# Normalize consumption by that of the reference group
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[(cex.year == 2019) & (cex.race == 1), 'consumption'], weights=cex.loc[(cex.year == 2019) & (cex.race == 1), 'weight'])
cex.loc[:, 'consumption_nipa'] = cex.consumption_nipa / np.average(cex.loc[(cex.year == 2019) & (cex.race == 1), 'consumption_nipa'], weights=cex.loc[(cex.year == 2019) & (cex.race == 1), 'weight'])

# Calculate the logarithm of average consumption by year and race
df = cex.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption':      np.log(np.average(x.consumption, weights=x.weight)),
                                                                              'consumption_nipa': np.log(np.average(x.consumption_nipa, weights=x.weight))}))

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption'], color=colors[0], linewidth=2.5, label='_nolegend_')
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption_nipa'], color=colors[0], linewidth=2.5, alpha=0.2)
ax.annotate('White', xy=(2019.25, df.loc[df.race == 1, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption'], color=colors[1], linewidth=2.5, label='_nolegend_')
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption_nipa'], color=colors[1], linewidth=2.5, alpha=0.2)
ax.annotate('Black', xy=(2019.25, df.loc[df.race == 2, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.28), np.log(1))
ax.set_yticks(np.log(np.linspace(0.3, 1, 8)))
ax.set_yticklabels(np.linspace(30, 100, 8).astype('int'))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Set the legend
ax.legend(['White (NIPA adjustment)', 'Black (NIPA adjustment)'], frameon=False, fontsize=12)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average consumption nipa.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the standard deviation of log nondurable    #
# consumption by year for Black and White Americans from 1984 to 2019.         #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2019 + 1)), :]

# Normalize consumption by that of the reference group
cex.loc[:, 'consumption_nd'] = cex.consumption_nd / np.average(cex.loc[(cex.year == 2019) & (cex.race == 1), 'consumption_nd'], weights=cex.loc[(cex.year == 2019) & (cex.race == 1), 'weight'])

# Calculate the standard deviation of log consumption by year and race
df = cex.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption_nd': np.sqrt(np.average((np.log(x.consumption_nd) - np.average(np.log(x.consumption_nd), weights=x.weight))**2, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_sd': lambda x: x.quantile(0.025)}).rename(columns={'consumption_sd': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_sd': lambda x: x.quantile(0.975)}).rename(columns={'consumption_sd': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots()

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption_nd'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2000, 0.6), color='k', fontsize=12, ha='center', va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption_nd'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2004, 0.5), color='k', fontsize=12, ha='center', va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

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
# This section of the script plots log average consumption by year for Black   #
# and White Americans from 1940 to 2019.                                       #
#                                                                              #
################################################################################

# Load the ACS data
acs = pd.read_csv(os.path.join(acs_f_data, 'acs.csv'))
acs = acs.loc[acs.year <= 2019, :]

# Normalize consumption by that of the reference group
acs.loc[:, 'consumption'] = acs.consumption / np.average(acs.loc[(acs.year == 2019) & (acs.race == 1), 'consumption'], weights=acs.loc[(acs.year == 2019) & (acs.race == 1), 'weight'])

# Calculate the logarithm of average consumption by year and race
df = acs.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.log(np.average(x.consumption, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs_historical.loc[dignity_bs_historical.year.isin(df.year.unique()) & dignity_bs_historical.race.isin([1, 2]) & (dignity_bs_historical.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.025)}).rename(columns={'consumption_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.975)}).rename(columns={'consumption_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'consumption'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2019.5, df.loc[df.race == 1, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'consumption'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2019.5, df.loc[df.race == 2, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1940, 2019)
ax.set_xticks(np.append(np.linspace(1940, 2010, 8), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.1), np.log(1))
ax.set_yticks(np.log(np.linspace(0.1, 1, 10)))
ax.set_yticklabels(np.linspace(10, 100, 10).astype('int'))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average consumption historical.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots log average consumption by year for Black   #
# non-Latino, White non-Latino and Latino Americans from 2006 to 2019.         #
#                                                                              #
################################################################################

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(2006, 2019 + 1)), :]

# Normalize consumption by that of the reference group
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[(cex.year == 2019) & (cex.race == 1) & (cex.latin == 0), 'consumption'], weights=cex.loc[(cex.year == 2019) & (cex.race == 1) & (cex.latin == 0), 'weight'])

# Calculate the logarithm of average consumption by year and race
df = cex.groupby(['year', 'race', 'latin'], as_index=False).apply(lambda x: pd.Series({'consumption': np.log(np.average(x.consumption, weights=x.weight))}))
df_latin = cex.groupby(['year', 'latin'], as_index=False).apply(lambda x: pd.Series({'consumption': np.log(np.average(x.consumption, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == 0) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.025)}).rename(columns={'consumption_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'consumption_average': lambda x: x.quantile(0.975)}).rename(columns={'consumption_average': 'ub'}), how='left')
df_bs_latin = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & (dignity_bs.race == -1) & (dignity_bs.latin == 1) & (dignity_bs.simple == True), :]
df_bs_latin = pd.merge(df_bs_latin.groupby('year', as_index=False).agg({'consumption_average': lambda x: x.quantile(0.025)}).rename(columns={'consumption_average': 'lb'}),
                       df_bs_latin.groupby('year', as_index=False).agg({'consumption_average': lambda x: x.quantile(0.975)}).rename(columns={'consumption_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(range(2006, 2019 + 1), df.loc[(df.race == 1) & (df.latin == 0), 'consumption'], color=colors[0], linewidth=2.5)
ax.fill_between(range(2006, 2019 + 1), df_bs.loc[df_bs.race == 1, 'lb'], y2=df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White non-Latinx', xy=(2007, np.log(0.9)), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(range(2006, 2019 + 1), df.loc[(df.race == 2) & (df.latin == 0), 'consumption'], color=colors[1], linewidth=2.5)
ax.fill_between(range(2006, 2019 + 1), df_bs.loc[df_bs.race == 2, 'lb'], y2=df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black non-Latinx', xy=(2015.5, np.log(0.7)), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(range(2006, 2019 + 1), df_latin.loc[df_latin.latin == 1, 'consumption'], color=colors[2], linewidth=2.5)
ax.fill_between(range(2006, 2019 + 1), df_bs_latin.lb, y2=df_bs_latin.ub, color=colors[2], alpha=0.2, linewidth=0)
ax.annotate('Latinx', xy=(2017.75, np.log(0.56)), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(2006, 2019)
ax.set_xticks(np.linspace(2007, 2019, 7))

# Set the vertical axis
ax.set_ylim(np.log(0.48), np.log(1))
ax.set_yticks(np.log(np.linspace(0.5, 1, 6)))
ax.set_yticklabels(np.linspace(50, 100, 6).astype('int'))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average consumption ethnicity.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption imputation.                 #
#                                                                              #
################################################################################

# Load the ACS data
acs = pd.read_csv(os.path.join(acs_f_data, 'acs.csv'))
acs = acs.loc[(acs.year >= 1990) & (acs.year <= 2019), :]

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex = cex.loc[cex.year.isin(range(1984, 2019 + 1)), :]
cex = cex.loc[cex.year >= 1990, :]

# Normalize consumption in the ACS and CEX by that of the reference group
acs.loc[:, 'consumption'] = acs.consumption / np.average(acs.loc[(acs.year == 2019) & (acs.race == 1), 'consumption'], weights=acs.loc[(acs.year == 2019) & (acs.race == 1), 'weight'])
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[(cex.year == 2019) & (cex.race == 1), 'consumption'], weights=cex.loc[(cex.year == 2019) & (cex.race == 1), 'weight'])

# Calculate the logarithm of average consumption by year and race in the ACS and CEX
acs = acs.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.log(np.average(x.consumption, weights=x.weight))}))
cex = cex.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'consumption': np.log(np.average(x.consumption, weights=x.weight))}))

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.scatter(acs.year.unique(), acs.loc[acs.race == 1, 'consumption'], color=colors[0], s=15, clip_on=False)
ax.plot(cex.year.unique(), cex.loc[cex.race == 1, 'consumption'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2019.25, cex.loc[cex.race == 1, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.scatter(acs.year.unique(), acs.loc[acs.race == 2, 'consumption'], color=colors[1], s=15, clip_on=False)
ax.plot(cex.year.unique(), cex.loc[cex.race == 2, 'consumption'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2019.25, cex.loc[cex.race == 2, 'consumption'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1990, 2019)
ax.set_xticks(np.append(np.linspace(1990, 2015, 6), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.35), np.log(1))
ax.set_yticks(np.log(np.linspace(0.4, 1, 7)))
ax.set_yticklabels(np.linspace(40, 100, 7).astype('int'))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Consumption imputation.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots housing consumption against the number of   #
# rooms in a CU's living quarters for Black and White Americans in 2019.       #
#                                                                              #
################################################################################

# Load the CEX data
#cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))

# Calculate monthly housing expenditures by number of rooms and race
#df = cex.loc[(cex.year == 2019) & (cex.rooms >= 1) & (cex.rooms <= cex.rooms.quantile(0.95)), :]
#df.loc[:, 'housing'] = df.housing * df.consumption
#df = df.groupby(['rooms', 'race'], as_index=False).agg({'housing': lambda x: weighted_average(x, data=df, weights='weight')})

# Initialize the figure
#fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
#ax.scatter(df.loc[df.race == 1, 'rooms'], df.loc[df.race == 1, 'housing'], color=colors[0], s=45, clip_on=False)
#ax.annotate('White', xy=(df.loc[df.race == 1, 'rooms'].iloc[-1] + 0.25, df.loc[df.race == 1, 'housing'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
#ax.scatter(df.loc[df.race == 2, 'rooms'], df.loc[df.race == 2, 'housing'], color=colors[1], s=45, clip_on=False)
#ax.annotate('Black', xy=(df.loc[df.race == 2, 'rooms'].iloc[-1] + 0.25, df.loc[df.race == 2, 'housing'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
#ax.set_xlim(0.75, 10)
#ax.set_xticks(np.linspace(1, 10, 10))
#ax.set_xlabel('Rooms', fontsize=12, ha='center', va='center')
#ax.xaxis.set_label_coords(0.5, -0.135)

# Set the vertical axis
#ax.set_ylim(0, 1800)
#ax.set_ylabel(r'\$', fontsize=12, rotation=0, ha='center', va='center')
#ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

# Save and close the figure
#fig.tight_layout()
#fig.savefig(os.path.join(figures, 'Housing.pdf'), format='pdf')
#plt.close()

################################################################################
#                                                                              #
# This section of the script plots average leisure by year for Black and White #
# Americans from 1984 to 2019.                                                 #
#                                                                              #
################################################################################

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2020 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate average leisure by year and race
df = cps.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'leisure': np.average(x.leisure, weights=x.weight)}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.025)}).rename(columns={'leisure_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.975)}).rename(columns={'leisure_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * df.loc[df.race == 1, 'leisure'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 1, 'lb'], y2=100 * df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White', xy=(2019.25, 100 * df.loc[df.race == 1, 'leisure'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'leisure'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 2, 'lb'], y2=100 * df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black', xy=(2019.25, 100 * df.loc[df.race == 2, 'leisure'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

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
# This section of the script plots average leisure by year for Black           #
# non-Latino, White non-Latino and Latino Americans from 2006 to 2019.         #
#                                                                              #
################################################################################

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(2007, 2020 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate average leisure by year and race
df = cps.groupby(['year', 'race', 'latin'], as_index=False).apply(lambda x: pd.Series({'leisure': np.average(x.leisure, weights=x.weight)}))
df_latin = cps.groupby(['year', 'latin'], as_index=False).apply(lambda x: pd.Series({'leisure': np.average(x.leisure, weights=x.weight)}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == 0) & (dignity_bs.simple == True), :]
df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.025)}).rename(columns={'leisure_average': 'lb'}),
                 df_bs.groupby(['year', 'race'], as_index=False).agg({'leisure_average': lambda x: x.quantile(0.975)}).rename(columns={'leisure_average': 'ub'}), how='left')
df_bs_latin = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & (dignity_bs.race == -1) & (dignity_bs.latin == 1) & (dignity_bs.simple == True), :]
df_bs_latin = pd.merge(df_bs_latin.groupby('year', as_index=False).agg({'leisure_average': lambda x: x.quantile(0.025)}).rename(columns={'leisure_average': 'lb'}),
                       df_bs_latin.groupby('year', as_index=False).agg({'leisure_average': lambda x: x.quantile(0.975)}).rename(columns={'leisure_average': 'ub'}), how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(range(2006, 2019 + 1), 100 * df.loc[(df.race == 1) & (df.latin == 0), 'leisure'], color=colors[0], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 1, 'lb'], y2=100 * df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.annotate('White non-Latinx', xy=(2008.5, 83.8), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(range(2006, 2019 + 1), 100 * df.loc[(df.race == 2) & (df.latin == 0), 'leisure'], color=colors[1], linewidth=2.5)
ax.fill_between(df_bs.year.unique(), 100 * df_bs.loc[df_bs.race == 2, 'lb'], y2=100 * df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black non-Latinx', xy=(2008.5, 87.4), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(range(2006, 2019 + 1), 100 * df_latin.loc[df_latin.latin == 1, 'leisure'], color=colors[2], linewidth=2.5)
ax.fill_between(df_bs_latin.year.unique(), 100 * df_bs_latin.lb, y2=100 * df_bs_latin.ub, color=colors[2], alpha=0.2, linewidth=0)
ax.annotate('Latinx', xy=(2006.35, 84.3), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(2006, 2019)
ax.set_xticks(np.linspace(2007, 2019, 7))

# Set the vertical axis
ax.set_ylim(81.5, 88)
ax.set_yticks(np.linspace(82, 88, 7))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Average leisure ethnicity.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the standard deviation of leisure by year   #
# for Black and White Americans from 1984 to 2019.                             #
#                                                                              #
################################################################################

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(1985, 2020 + 1)), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate the standard deviation of leisure by year and race
df = cps.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'leisure': np.sqrt(np.average((x.leisure - np.average(x.leisure, weights=x.weight))**2, weights=x.weight))}))

# Calculate the 95% confidence interval
df_bs = dignity_bs.loc[dignity_bs.year.isin(range(1984, 2019 + 1)) & dignity_bs.race.isin([1, 2]) & (dignity_bs.latin == -1) & (dignity_bs.simple == True), :]
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
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

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
# This section of the script plots life expectancy by year for Black and White #
# Americans from 1984 to 2019.                                                 #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity = dignity.loc[dignity.year <= 2019, :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.race != -1) & (dignity.latin == -1), :]

# Compute life expectancy by year and race
df = dignity.groupby(['year', 'race'], as_index=False).agg({'S': 'sum'})

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'S'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2019.25, df.loc[df.race == 1, 'S'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'S'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2019.25, df.loc[df.race == 2, 'S'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

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
# This section of the script plots life expectancy by year for Black and White #
# Americans from 1940 to 2019.                                                 #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity = dignity.loc[dignity.year <= 2019, :]
dignity = dignity.loc[(dignity.historical == True) & (dignity.race != -1) & (dignity.latin == -1), :]

# Compute life expectancy by year and race
df = dignity.groupby(['year', 'race'], as_index=False).agg({'S': 'sum'})

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'S'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2019.25, df.loc[df.race == 1, 'S'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'S'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2019.25, df.loc[df.race == 2, 'S'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1940, 2019)
ax.set_xticks(np.append(np.linspace(1940, 2010, 8), 2019))

# Set the vertical axis
ax.set_ylim(54, 80)
ax.set_ylabel('Years', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Life expectancy historical.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots life expectancy by year for Black           #
# non-Latino, White non-Latino and Latino Americans from 2006 to 2019.         #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity = dignity.loc[dignity.year <= 2019, :]
dignity = dignity.loc[(dignity.historical == False) & (((dignity.race != -1) & (dignity.latin == 0)) | (dignity.latin == 1)), :]

# Compute life expectancy by year and race
df = dignity.groupby(['year', 'race'], as_index=False).agg({'S': 'sum'})

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'S'], color=colors[0], linewidth=2.5)
ax.annotate('White non-Latinx', xy=(2015, 79.7), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'S'], color=colors[1], linewidth=2.5)
ax.annotate('Black non-Latinx', xy=(2015.3, 76), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == -1, 'S'], color=colors[2], linewidth=2.5)
ax.annotate('Latinx', xy=(2017.3, 82.9), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(2006, 2019)
ax.set_xticks(np.linspace(2007, 2019, 7))

# Set the vertical axis
ax.set_ylim(72, 84)
ax.set_ylabel('Years', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Life expectancy ethnicity.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots incarceration rates by year for Black and   #
# White Americans from 1999 to 2019.                                           #
#                                                                              #
################################################################################

# Load the NCR data
ncr = pd.read_csv(os.path.join(ncr_f_data, 'ncr.csv'))
df = ncr.groupby(['year', 'race'], as_index=False).agg({'incarcerated': 'sum', 'population': 'sum'})

# Calculate incarceration rates
df.loc[:, 'incarceration'] = df.incarcerated / df.population

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * df.loc[df.race == 1, 'incarceration'], color=colors[0], linewidth=2.5)
ax.annotate('White non-Latinx', xy=(2008, 0.4), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * df.loc[df.race == 2, 'incarceration'], color=colors[1], linewidth=2.5)
ax.annotate('Black non-Latinx', xy=(2007, 1.85), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1999, 2019)
ax.set_xticks(np.linspace(1999, 2019, 11))

# Set the vertical axis
ax.set_ylim(0, 2)
ax.set_yticks(np.linspace(0, 2, 5))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
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
# This section of the script plots life expectancy and quality-adjusted life   #
# expectancy by year for Black and White Americans from 1997 to 2018.          #
#                                                                              #
################################################################################

# Load the NHIS data and calculate the average HALex by year, race and age
nhis = pd.read_csv(os.path.join(nhis_f_data, 'nhis.csv')).dropna(subset=['halex'])
nhis = nhis.loc[nhis.year.isin(range(1997, 2018 + 1)) & nhis.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = pd.merge(expand({'year': nhis.year.unique(), 'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhis.loc[:, 'halex'] = nhis.groupby(['year', 'race'], as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data and merge it with the NHIS data
df = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
df = df.loc[df.year.isin(range(1997, 2018 + 1)) & (df.historical == False) & df.race.isin([1, 2]) & (df.latin == -1), ['year', 'race', 'age', 'S']]
df = pd.merge(df, nhis, how='left')

# Calculate quality-adjusted survival rates
df.loc[:, 'Q'] = df.S * (0.1 + 0.9 * df.halex)

# Calculate life expectancy and quality-adjusted life expectancy by year and race
df = df.groupby(['year', 'race'], as_index=False).agg({'S': 'sum', 'Q': 'sum'})

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'Q'], color=colors[0], linewidth=2.5)
ax.annotate('White adjusted', xy=(2013, df.loc[df.race == 1, 'Q'].iloc[-1] + 1), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'Q'], color=colors[1], linewidth=2.5)
ax.annotate('Black adjusted', xy=(2013, df.loc[df.race == 2, 'Q'].iloc[-1] - 1), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 1, 'S'], color=colors[0], alpha=0.2, linewidth=2.5)
ax.annotate('White unadjusted', xy=(2012, df.loc[df.race == 1, 'S'].iloc[-1] + 1), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'S'], color=colors[1], alpha=0.2, linewidth=2.5)
ax.annotate('Black unadjusted', xy=(2012, df.loc[df.race == 2, 'S'].iloc[-1] - 1), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1997, 2018)
ax.set_xticks(np.linspace(1998, 2018, 6))

# Set the vertical axis
ax.set_ylim(55, 80)
ax.set_ylabel('Years', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Quality-adjusted life expectancy.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the HALex by year for Black and White       #
# Americans from 1997 to 2018.                                                 #
#                                                                              #
################################################################################

# Load the NHIS data and calculate the average HALex by year, race and age
nhis = pd.read_csv(os.path.join(nhis_f_data, 'nhis.csv')).dropna(subset=['halex'])
nhis = nhis.loc[nhis.year.isin(range(1997, 2018 + 1)) & nhis.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = pd.merge(expand({'year': nhis.year.unique(), 'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhis.loc[:, 'halex'] = nhis.groupby(['year', 'race'], as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1
nhis.loc[:, 'halex'] = 0.1 + 0.9 * nhis.halex

# Average the HALex across ages with the 2018 age distributions by race
population = pd.read_csv(os.path.join(population_f_data, 'population.csv'))
df = pd.merge(nhis, population.loc[(population.year == 2018) & population.race.isin([1, 2]), :].groupby(['race', 'age'], as_index=False).agg({'population': 'sum'}), how='left')
df = df.groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.population)}))

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'halex'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2018.25, df.loc[df.race == 1, 'halex'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'halex'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2018.25, df.loc[df.race == 2, 'halex'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1997, 2018)
ax.set_xticks(np.linspace(1998, 2018, 6))

# Set the vertical axis
ax.set_ylim(0.82, 0.88)
ax.set_ylabel('HALex', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'HALex.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the difference between life expectancy and  #
# quality-adjusted life expectancy by year for Black and White Americans from  #
# 1997 to 2018.                                                                #
#                                                                              #
################################################################################

# Load the NHIS data and calculate the average HALex by year, race and age
nhis = pd.read_csv(os.path.join(nhis_f_data, 'nhis.csv')).dropna(subset=['halex'])
nhis = nhis.loc[nhis.year.isin(range(1997, 2018 + 1)) & nhis.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = pd.merge(expand({'year': nhis.year.unique(), 'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhis.loc[:, 'halex'] = nhis.groupby(['year', 'race'], as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data and merge it with the NHIS data
df = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
df = df.loc[df.year.isin(range(1997, 2018 + 1)) & (df.historical == False) & df.race.isin([1, 2]) & (df.latin == -1), ['year', 'race', 'age', 'S']]
df = pd.merge(df, nhis, how='left')

# Calculate quality-adjusted survival rates
df.loc[:, 'Q'] = df.S * (0.1 + 0.9 * df.halex)

# Calculate life expectancy and quality-adjusted life expectancy by year and race
df = df.groupby(['year', 'race'], as_index=False).agg({'S': 'sum', 'Q': 'sum'})
df.loc[:, 'gap'] = df.S - df.Q

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), df.loc[df.race == 1, 'gap'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2018.25, df.loc[df.race == 1, 'gap'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(df.year.unique(), df.loc[df.race == 2, 'gap'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2018.25, df.loc[df.race == 2, 'gap'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1997, 2018)
ax.set_xticks(np.linspace(1998, 2018, 6))

# Set the vertical axis
ax.set_ylim(10, 15)
ax.set_ylabel('Years', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Life expectancy quality adjustment.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare of Black #
# relative to White Americans from 1984 to 2019.                               #
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
    df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, df.log_lambda, color=colors[1], linewidth=2.5)
ax.fill_between(years, df_bs.lb, y2=df_bs.ub, color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[-1])), xy=(2019.25, df.log_lambda.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[0])), xy=(1981.5, df.log_lambda.iloc[0]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.4), np.log(0.7))
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
# relative to White Americans from 1984 to 2019.                               #
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

# Create a data frame
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years)), 'log_lambda_nipa': np.zeros(len(years))})

# Calculate the consumption-equivalent welfare of Black relative to White Americans without the NIPA PCE adjustment
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
    df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the NIPA PCE adjustment
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
    df.loc[df.year == year, 'log_lambda_nipa'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                           S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                           inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, df.log_lambda, color=colors[1], linewidth=2.5, label='_nolegend_')
ax.plot(years, df.log_lambda_nipa, color=colors[1], linewidth=2.5, alpha=0.2)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[-1])), xy=(2019.25, df.log_lambda.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[0])), xy=(1981.5, df.log_lambda.iloc[0]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.37), np.log(0.7))
ax.set_yticks(np.log(np.linspace(0.4, 0.7, 4)))
ax.set_yticklabels(np.round_(np.linspace(0.4, 0.7, 4), 1))

# Set the legend
ax.legend(['NIPA adjustment'], frameon=False, fontsize=12)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare nipa.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare,         #
# consumption, earnings and wealth of Black relative to White Americans from   #
# 1984 to 2019.                                                                #
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

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years))})
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
    df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

# Plot the lines
ax1.plot(years, df.log_lambda, color=colors[1], linewidth=2.5)
ax1.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[-1])), xy=(2019.25, df.log_lambda.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[0])), xy=(1981.5, df.log_lambda.iloc[0]), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.annotate('Welfare', xy=(2012, np.log(0.575)), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.plot(years, np.log(cex.consumption), color=colors[1], linewidth=2, linestyle='dashed')
ax1.annotate('{0:.2f}'.format(cex.consumption.iloc[-1]), xy=(2019.25, np.log(cex.consumption.iloc[-1])), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.annotate('Consumption', xy=(1999, np.log(0.625)), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.plot(years, np.log(cps.earnings), color=colors[1], linewidth=2, linestyle='dotted')
ax1.annotate('{0:.2f}'.format(cps.earnings.iloc[-1]), xy=(2019.25, np.log(cps.earnings.iloc[-1])), color='k', fontsize=12, va='center', annotation_clip=False)
ax1.annotate('Earnings', xy=(2001, np.log(0.72)), color='k', fontsize=12, va='center', annotation_clip=False)
ax2.plot(np.linspace(1989, 2019, 11), np.log(scf_data[:, 1] / scf_data[:, 0]), color=colors[0], linewidth=2, markersize=4, marker='o', clip_on=False)
ax2.annotate('Wealth (right scale)', xy=(2008, np.log(0.1225)), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax1.set_xlim(1984, 2019)
ax1.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axes
ax1.set_ylim(np.log(0.35), np.log(0.8))
ax1.set_yticks(np.log(np.linspace(0.4, 0.8, 5)))
ax1.set_yticklabels(np.round_(np.linspace(0.4, 0.8, 5), 1))
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
fig.savefig(os.path.join(figures, 'Welfare, consumption, earnings and wealth.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare of Black #
# relative to White Americans from 1940 to 2019.                               #
#                                                                              #
################################################################################

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the 95% confidence interval
df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
df_bs = df_bs.loc[(df_bs.description == 'Welfare historical'), :]
df_bs = pd.merge(df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'lb'}),
                 df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'ub'}), how='left')

# Calculate the consumption-equivalent welfare of Black relative to White Americans from the U.S. censuses and ACS
historical_years = list(range(1940, 1990 + 1, 10)) + list(range(2000, 2019 + 1))
historical_df = pd.DataFrame({'year': historical_years, 'log_lambda': np.zeros(len(historical_years))})
for year in historical_years:
    S_i = dignity.loc[(dignity.historical == True) & (dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.historical == True) & (dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.historical == True) & (dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.historical == True) & (dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.historical == True) & (dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.historical == True) & (dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    historical_df.loc[historical_df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                            S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)['log_lambda']

# Calculate the consumption-equivalent welfare of Black relative to White Americans from the CEX and CPS
years = range(1984, 2019 + 1)
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years))})
for year in years:
    S_i = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.historical == False) & (dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.scatter(years, df.log_lambda, color=colors[1], s=15, clip_on=False)
ax.annotate('CEX/CPS', xy=(2014, np.log(0.65)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)
ax.plot(historical_years, historical_df.log_lambda, color=colors[1], linewidth=2.5)
ax.fill_between(historical_years, df_bs.lb, y2=df_bs.ub, color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Census/ACS', xy=(1980, np.log(0.49)), color='k', fontsize=12, va='center', ha='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1940, 2019)
ax.set_xticks(np.append(np.linspace(1940, 2010, 8), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.28), np.log(0.7))
ax.set_yticks(np.log(np.linspace(0.3, 0.7, 5)))
ax.set_yticklabels(np.round_(np.linspace(0.3, 0.7, 5), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare historical.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare and      #
# income of Black non-Latino and Latino Americans relative to White non-Latino #
# Americans from 2006 to 2019.                                                 #
#                                                                              #
################################################################################

# Define a list of years
years = range(2006, 2019 + 1)

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[dignity.year.isin(years) & (dignity.historical == False), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the 95% confidence interval
df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
df_bs = df_bs.loc[(df_bs.description == 'Welfare ethnicity'), :]
df_bs = pd.merge(df_bs.groupby(['year', 'latin'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'lb'}),
                 df_bs.groupby(['year', 'latin'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'ub'}), how='left')
                 
# Create a data frame and a list of years
df = expand({'year': years, 'latin': [0, 1]})
df.loc[:, 'log_lambda'] = np.nan

# Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2) & (dignity.latin == 0), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2) & (dignity.latin == 0), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2) & (dignity.latin == 0), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2) & (dignity.latin == 0), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2) & (dignity.latin == 0), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2) & (dignity.latin == 0), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2) & (dignity.latin == 0), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.latin == 0), 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Calculate the consumption-equivalent welfare of Latino relative to White non-Latino Americans
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == 1), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == 1), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == 1), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == 1), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == 1), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == 1), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1) & (dignity.latin == 0), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == -1) & (dignity.latin == 1), 'Ev_of_ell'].values
    df.loc[(df.year == year) & (df.latin == 1), 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, df.loc[df.latin == 0, 'log_lambda'], color=colors[1], linewidth=2.5)
ax.fill_between(years, df_bs.loc[df_bs.latin == 0, 'lb'], y2=df_bs.loc[df_bs.latin == 0, 'ub'], color=colors[1], alpha=0.2, linewidth=0)
ax.annotate('Black non-Latinx', xy=(2010, np.log(0.62)), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(years, df.loc[df.latin == 1, 'log_lambda'], color=colors[2], linewidth=2.5)
ax.fill_between(years, df_bs.loc[df_bs.latin == 1, 'lb'], y2=df_bs.loc[df_bs.latin == 1, 'ub'], color=colors[2], alpha=0.2, linewidth=0)
ax.annotate('Latinx', xy=(2016, np.log(0.83)), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(2006, 2019)
ax.set_xticks(np.linspace(2007, 2019, 7))

# Set the vertical axis
ax.set_ylim(np.log(0.4), np.log(1))
ax.set_yticks(np.log(np.linspace(0.4, 1, 7)))
ax.set_yticklabels(np.round_(np.linspace(0.4, 1, 7), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare ethnicity.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
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

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = pd.DataFrame({'year': years, 'LE': np.zeros(len(years)),
                                  'C':  np.zeros(len(years)),
                                  'CI': np.zeros(len(years)),
                                  'L':  np.zeros(len(years)),
                                  'LI': np.zeros(len(years))})
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
    for i in ['LE', 'C', 'CI', 'L', 'LI']:
        df.loc[df.year == year, i] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                               S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                               inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)[i]

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.stackplot(years, [df.LE, df.C], colors=newcolors[2:], edgecolor='Black', linewidth=0.75)
ax.stackplot(years, [df.L, df.CI + df.LI], colors=[newcolors[1], newcolors[0]], edgecolor='Black', linewidth=0.75)
ax.arrow(1990, np.log(1.02), 0, 0.09, linewidth=1, color='Black')

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.37), np.log(1.12))
ax.set_yticks(np.log(np.linspace(0.4, 1, 4)))
ax.set_yticklabels(np.linspace(0.4, 1, 4))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(1990, np.log(1.14), 'Leisure', fontsize=12, ha='center')
ax.text(2002, np.log(1.11), 'Inequality', fontsize=12, ha='center')
ax.text(1990, np.log(0.78), 'Life expectancy', fontsize=12, ha='center')
ax.text(1990, np.log(0.49), 'Consumption', fontsize=12, ha='center')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare decomposition.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# decomposition of Black relative to White Americans from 1940 to 2019.        #
#                                                                              #
################################################################################

# Define a list of years
years = list(range(1940, 1990 + 1, 10)) + list(range(2000, 2019 + 1))

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == True) & (dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = pd.DataFrame({'year': years, 'LE': np.zeros(len(years)),
                                  'C':  np.zeros(len(years)),
                                  'L':  np.zeros(len(years))})
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
    for i in ['LE', 'C', 'L']:
        df.loc[df.year == year, i] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                               S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)[i]
# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.stackplot(years, [df.LE, df.C], colors=newcolors[2:], edgecolor='Black', linewidth=0.75)
ax.stackplot(years, [df.L], colors=[newcolors[1], newcolors[0]], edgecolor='Black', linewidth=0.75)

# Set the horizontal axis
ax.set_xlim(1940, 2019)
ax.set_xticks(np.append(np.linspace(1940, 2010, 8), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.28), np.log(1.07))
ax.set_yticks(np.log(np.linspace(0.3, 1, 8)))
ax.set_yticklabels(np.round_(np.linspace(0.3, 1, 8), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(1990, np.log(1.1), 'Leisure', fontsize=12, ha='center')
ax.text(1961, np.log(0.72), 'Life expectancy', fontsize=12, ha='center')
ax.text(1961, np.log(0.44), 'Consumption', fontsize=12, ha='center')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare decomposition historical.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# health adjustment of Black relative to White Americans in 2018.              #
#                                                                              #
################################################################################

# Load the NHIS data and calculate the average HALex by year, race and age
nhis = pd.read_csv(os.path.join(nhis_f_data, 'nhis.csv')).dropna(subset=['halex'])
nhiS_intercept = nhis.loc[nhis.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = nhis.loc[(nhis.year == 2018) & nhis.race.isin([1, 2]), :].groupby(['race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhiS_intercept = pd.merge(expand({'age': range(101)}), nhiS_intercept, how='left')
nhis = pd.merge(expand({'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhiS_intercept.loc[:, 'halex'] = nhiS_intercept.halex.transform(lambda x: filter(x, 100)).values
nhis.loc[:, 'halex'] = nhis.groupby('race', as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhiS_intercept.loc[nhiS_intercept.halex < 0, 'halex'] = 0
nhiS_intercept.loc[nhiS_intercept.halex > 1, 'halex'] = 1
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data and merge it with the NHIS and CEX data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & dignity.race.isin([1, 2]) & (dignity.latin == -1) & (dignity.year == 2018), :]
dignity_intercept = pd.merge(dignity_intercept, nhiS_intercept, how='left')
dignity = pd.merge(dignity, nhis, how='left')

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the consumption-equivalent welfare of Black relative to White Americans without the morbidity adjustment
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

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the morbidity adjustment
df = pd.DataFrame({'parameter': np.linspace(0, 1, 101)})
df.loc[:, 'log_lambda'] = np.nan
for i in np.linspace(0, 1, 101):
    S_i = dignity.loc[(dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.race == 1), 'c_bar_nh'].values
    c_j_bar = dignity.loc[(dignity.race == 2), 'c_bar_nh'].values
    ell_i_bar = dignity.loc[(dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    halex_intercept = dignity_intercept.loc[:, 'halex'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar_nh'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.race == 1), 'c_bar_nh_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.race == 2), 'c_bar_nh_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.race == 1), 'Elog_of_c_nh'].values
    Elog_of_c_j = dignity.loc[(dignity.race == 2), 'Elog_of_c_nh'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.race == 1), 'Elog_of_c_nh_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.race == 2), 'Elog_of_c_nh_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.race == 2), 'Ev_of_ell'].values
    halex_i = dignity.loc[(dignity.race == 1), 'halex'].values
    halex_j = dignity.loc[(dignity.race == 2), 'halex'].values
    df.loc[df.parameter == i, 'log_lambda'] = cew_level_morbidity(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                  S_intercept=S_intercept, halex_intercept=halex_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                  c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j,
                                                                  halex_i=halex_i, halex_j=halex_j, morbidity_parameter=i)['log_lambda']

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
ax.set_ylim(np.log(0.35), np.log(0.65))
ax.set_yticks(np.log(np.linspace(0.35, 0.65, 7)))
ax.set_yticklabels(list(map('{0:.2f}'.format, np.linspace(0.35, 0.65, 7))))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity in levels.pdf'), format='pdf')
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(np.linspace(0, 1, 101), 100 * (df.log_lambda - log_lambda), color=colors[1], linewidth=2.5)
ax.plot(0.1, 100 * (float(df.loc[df.parameter == 0.1, 'log_lambda']) - log_lambda), color=colors[1], marker='o', markersize=8)

# Set the horizontal axis
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xticklabels(np.linspace(0, 100, 11).astype('int'))
ax.set_xlabel(r'Worst morbidity (\%)', fontsize=12, rotation=0, ha='center')

# Set the vertical axis
ax.set_ylim(-50, 0)
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity in log points.pdf'), format='pdf')
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
nhiS_intercept = nhis.loc[nhis.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = nhis.loc[nhis.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhiS_intercept = pd.merge(expand({'age': range(101)}), nhiS_intercept, how='left')
nhis = pd.merge(expand({'year': years, 'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhiS_intercept.loc[:, 'halex'] = nhiS_intercept.halex.transform(lambda x: filter(x, 100)).values
nhis.loc[:, 'halex'] = nhis.groupby(['year', 'race'], as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhiS_intercept.loc[nhiS_intercept.halex < 0, 'halex'] = 0
nhiS_intercept.loc[nhiS_intercept.halex > 1, 'halex'] = 1
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data and merge it with the NHIS and CEX data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & dignity.race.isin([1, 2]) & (dignity.latin == -1) & dignity.year.isin(years), :]
dignity_intercept = pd.merge(dignity_intercept, nhiS_intercept, how='left')
dignity = pd.merge(dignity, nhis, how='left')

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Create a data frame
df = pd.DataFrame({'year': years, 'LE': np.zeros(len(years)),
                                  'M':  np.zeros(len(years)),
                                  'C':  np.zeros(len(years)),
                                  'CI': np.zeros(len(years)),
                                  'L':  np.zeros(len(years)),
                                  'LI': np.zeros(len(years))})

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the health adjustment
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nh'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nh'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    halex_intercept = dignity_intercept.loc[:, 'halex'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar_nh'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nh_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nh_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nh'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nh'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nh_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nh_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    halex_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'halex'].values
    halex_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'halex'].values
    for i in ['LE', 'M', 'C', 'CI', 'L', 'LI']:
        df.loc[df.year == year, i] = cew_level_morbidity(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                         S_intercept=S_intercept, halex_intercept=halex_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                         c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j,
                                                         halex_i=halex_i, halex_j=halex_j, morbidity_parameter=0.1)[i]

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.stackplot(years, [df.LE, df.C, df.M], colors=newnewcolors[2:], edgecolor='Black', linewidth=0.75)
ax.stackplot(years, [df.L, df.CI + df.LI], colors=[newnewcolors[1], newnewcolors[0]], edgecolor='Black', linewidth=0.75)
ax.arrow(1999, np.log(1.02), 0, 0.09, linewidth=1, color='Black')

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
ax.text(2007, np.log(1.11), 'Inequality', fontsize=12, ha='center')
ax.text(2001, np.log(0.76), 'Life expectancy', fontsize=12, ha='center')
ax.text(2001, np.log(0.5), 'Consumption', fontsize=12, ha='center')
ax.text(2001, np.log(0.34), 'Morbidity', fontsize=12, ha='center')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity decomposition.pdf'), format='pdf')
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
nhiS_intercept = nhis.loc[nhis.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhis = nhis.loc[nhis.race.isin([1, 2]), :].groupby(['year', 'race', 'age'], as_index=False).apply(lambda x: pd.Series({'halex': np.average(x.halex, weights=x.weight)}))
nhiS_intercept = pd.merge(expand({'age': range(101)}), nhiS_intercept, how='left')
nhis = pd.merge(expand({'year': years, 'race': nhis.race.unique(), 'age': range(101)}), nhis, how='left')
nhiS_intercept.loc[:, 'halex'] = nhiS_intercept.halex.transform(lambda x: filter(x, 100)).values
nhis.loc[:, 'halex'] = nhis.groupby(['year', 'race'], as_index=False).halex.transform(lambda x: filter(x, 100)).values
nhiS_intercept.loc[nhiS_intercept.halex < 0, 'halex'] = 0
nhiS_intercept.loc[nhiS_intercept.halex > 1, 'halex'] = 1
nhis.loc[nhis.halex < 0, 'halex'] = 0
nhis.loc[nhis.halex > 1, 'halex'] = 1

# Load the dignity data and merge it with the NHIS and CEX data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & dignity.race.isin([1, 2]) & (dignity.latin == -1) & dignity.year.isin(years), :]
dignity_intercept = pd.merge(dignity_intercept, nhiS_intercept, how='left')
dignity = pd.merge(dignity, nhis, how='left')

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Create a data frame
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years)), 'log_lambda_morbidity':  np.zeros(len(years))})

# Calculate the consumption-equivalent welfare of Black relative to White Americans
df = pd.DataFrame({'year': years, 'log_lambda': np.zeros(len(years))})
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
    df.loc[df.year == year, 'log_lambda'] = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                      S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                      inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the health adjustment
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nh'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nh'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    halex_intercept = dignity_intercept.loc[:, 'halex'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar_nh'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nh_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nh_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nh'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nh'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nh_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nh_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    halex_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'halex'].values
    halex_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'halex'].values
    df.loc[df.year == year, 'log_lambda_morbidity'] = cew_level_morbidity(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                          S_intercept=S_intercept, halex_intercept=halex_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                          c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j,
                                                                          halex_i=halex_i, halex_j=halex_j, morbidity_parameter=0.1)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, df.log_lambda, color=colors[1], linewidth=2.5)
ax.plot(years, df.log_lambda_morbidity, color=colors[1], linewidth=2.5, alpha=0.2)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[-1])), xy=(2018.25, df.log_lambda.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda.iloc[0])), xy=(1995.5, df.log_lambda.iloc[0]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda_morbidity.iloc[-1])), xy=(2018.25, df.log_lambda_morbidity.iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.annotate('{0:.2f}'.format(np.exp(df.log_lambda_morbidity.iloc[0])), xy=(1995.5, df.log_lambda_morbidity.iloc[0]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1997, 2018)
ax.set_xticks(np.linspace(1998, 2018, 6))

# Set the vertical axis
ax.set_ylim(np.log(0.25), np.log(0.7))
ax.set_yticks(np.log(np.linspace(0.3, 0.7, 5)))
ax.set_yticklabels(np.round_(np.linspace(0.3, 0.7, 5), 1))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and morbidity.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# incarceration adjustment of Black non-Latino relative to White non-Latino    #
# Americans in 2019.                                                           #
#                                                                              #
################################################################################

# Load the NCR data
ncr = pd.read_csv(os.path.join(ncr_f_data, 'ncr.csv'))
ncr = ncr.loc[ncr.year == 2019, :]
ncr_u_bar = ncr.groupby('age', as_index=False).agg({'incarcerated': 'sum', 'population': 'sum'})
ncr_u_bar.loc[:, 'incarceration'] = ncr_u_bar.incarcerated / ncr_u_bar.population
ncr.loc[:, 'incarceration'] = ncr.incarcerated / ncr.population

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[cex.year == 2019, 'consumption'], weights=cex.loc[cex.year == 2019, 'weight'])
cex = cex.loc[cex.year.isin([2006, 2019]) & (cex.education == 1), :]

# Calculate CEX consumption statistics by age for individuals with a high school education or less
cex_intercept = cex.loc[cex.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'c_intercept_I': np.average(x.consumption, weights=x.weight)}))
cex = cex.loc[cex.year == 2019, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'Elog_of_c_I': np.average(np.log(x.consumption), weights=x.weight)}))
cex_intercept = pd.merge(pd.DataFrame({'age': range(101)}), cex_intercept, how='left')
cex = pd.merge(pd.DataFrame({'age': range(101)}), cex, how='left')
cex_intercept.loc[:, 'c_intercept_I'] = filter(cex_intercept.loc[:, 'c_intercept_I'], 1600)
cex.loc[:, 'Elog_of_c_I'] = filter(cex.loc[:, 'Elog_of_c_I'], 1600)

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin([2007, 2020]) & (cps.education == 1), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate cps leisure statistics by age for individuals with a high school education or less
cps_intercept = cps.loc[cps.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'ell_intercept_I': np.average(x.leisure, weights=x.weight)}))
cps = cps.loc[cps.year == 2019, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'Ev_of_ell_I': np.average(v_of_ell(x.leisure), weights=x.weight)}))
cps_intercept = pd.merge(pd.DataFrame({'age': range(101)}), cps_intercept, how='left')
cps = pd.merge(pd.DataFrame({'age': range(101)}), cps, how='left')
cps_intercept.loc[:, 'ell_intercept_I'] = filter(cps_intercept.loc[:, 'ell_intercept_I'], 100)
cps.loc[:, 'Ev_of_ell_I'] = filter(cps.loc[:, 'Ev_of_ell_I'], 100)
cps_intercept.loc[cps_intercept.loc[:, 'ell_intercept_I'] > 1, 'ell_intercept_I'] = 1
cps.loc[cps.loc[:, 'Ev_of_ell_I'] > 0, 'Ev_of_ell_I'] = 0

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.latin == 0) & (dignity.year == 2019), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans without the incarceration adjustment
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

# Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans with the incarceration adjustment
df = pd.DataFrame({'parameter': np.linspace(0, 1, 101)})
df.loc[:, 'log_lambda'] = np.nan
for i in np.linspace(0, 1, 101):
    S_i = dignity.loc[(dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = ncr_u_bar.loc[:, 'incarceration'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    c_intercept_I = cex_intercept.loc[:, 'c_intercept_I'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    ell_intercept_I = cps_intercept.loc[:, 'ell_intercept_I'].values
    c_i_bar_nd = dignity.loc[(dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.race == 2), 'Ev_of_ell'].values
    Elog_of_c_I = cex.loc[:, 'Elog_of_c_I'].values
    Ev_of_ell_I = cps.loc[:, 'Ev_of_ell_I'].values
    I_i = ncr.loc[(ncr.race == 1), 'incarceration'].values
    I_j = ncr.loc[(ncr.race == 2), 'incarceration'].values
    df.loc[df.parameter == i, 'log_lambda'] = cew_level_incarceration(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                                      S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, c_intercept_I=c_intercept_I, ell_intercept=ell_intercept, ell_intercept_I=ell_intercept_I, c_nominal=c_nominal,
                                                                      c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j,
                                                                      Elog_of_c_I=Elog_of_c_I, Ev_of_ell_I=Ev_of_ell_I, I_i=I_i, I_j=I_j, incarceration_parameter=i)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(np.linspace(0, 1, 101), df.log_lambda, color=colors[1], linewidth=2.5)

# Set the horizontal axis
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xticklabels(np.linspace(0, 100, 11).astype('int'))
ax.set_xlabel(r'Flow utility in prison relative to not in prison (\%)', fontsize=12, rotation=0, ha='center')

# Set the vertical axis
ax.set_ylim(np.log(0.515), np.log(0.55))
ax.set_yticks(np.log(np.linspace(0.52, 0.55, 4)))
ax.set_yticklabels(list(map('{0:.2f}'.format, np.linspace(0.52, 0.55, 4))))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and incarceration in levels.pdf'), format='pdf')
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(np.linspace(0, 1, 101), 100 * (df.log_lambda - log_lambda), color=colors[1], linewidth=2.5)

# Set the horizontal axis
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xticklabels(np.linspace(0, 100, 11).astype('int'))
ax.set_xlabel(r'Flow utility in prison relative to not in prison (\%)', fontsize=12, rotation=0, ha='center')

# Set the vertical axis
ax.set_ylim(-7, 0)
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and incarceration in log points.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# decomposition of Black non-Latino relative to White non-Latino Americans     #
# from 2006 to 2019 with the incarceration adjustment.                         #
#                                                                              #
################################################################################

# Define a list of years
years = range(2006, 2019 + 1)

# Load the NCR data
ncr = pd.read_csv(os.path.join(ncr_f_data, 'ncr.csv'))
ncr = ncr.loc[ncr.year.isin(years), :]
ncr_u_bar = ncr.groupby(['year', 'age'], as_index=False).agg({'incarcerated': 'sum', 'population': 'sum'})
ncr_u_bar.loc[:, 'incarceration'] = ncr_u_bar.incarcerated / ncr_u_bar.population
ncr.loc[:, 'incarceration'] = ncr.incarcerated / ncr.population

# Load the CEX data
cex = pd.read_csv(os.path.join(cex_f_data, 'cex.csv'))
cex.loc[:, 'consumption'] = cex.consumption / np.average(cex.loc[cex.year == 2019, 'consumption'], weights=cex.loc[cex.year == 2019, 'weight'])
cex = cex.loc[cex.year.isin(years) & (cex.education == 1), :]

# Calculate CEX consumption statistics by age for individuals with a high school education or less
cex_intercept = cex.loc[cex.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'c_intercept_I': np.average(x.consumption, weights=x.weight)}))
cex = cex.groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'Elog_of_c_I': np.average(np.log(x.consumption), weights=x.weight)}))
cex_intercept = pd.merge(pd.DataFrame({'age': range(101)}), cex_intercept, how='left')
cex = pd.merge(expand({'year': years, 'age': range(101)}), cex, how='left')
cex_intercept.loc[:, 'c_intercept_I'] = filter(cex_intercept.loc[:, 'c_intercept_I'], 1600)
cex.loc[:, 'Elog_of_c_I'] = cex.groupby('year', as_index=False)['Elog_of_c_I'].transform(lambda x: filter(x, 1600))['Elog_of_c_I'].values

# Load the CPS data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(range(2007, 2020 + 1)) & (cps.education == 1), :]
cps.loc[:, 'year'] = cps.year - 1

# Calculate CPS leisure statistics by age for individuals with a high school education or less
cps_intercept = cps.loc[cps.year == 2006, :].groupby('age', as_index=False).apply(lambda x: pd.Series({'ell_intercept_I': np.average(x.leisure, weights=x.weight)}))
cps = cps.groupby(['year', 'age'], as_index=False).apply(lambda x: pd.Series({'Ev_of_ell_I': np.average(v_of_ell(x.leisure), weights=x.weight)}))
cps_intercept = pd.merge(pd.DataFrame({'age': range(101)}), cps_intercept, how='left')
cps = pd.merge(expand({'year': years, 'age': range(101)}), cps, how='left')
cps_intercept.loc[:, 'ell_intercept_I'] = filter(cps_intercept.loc[:, 'ell_intercept_I'], 100)
cps.loc[:, 'Ev_of_ell_I'] = cps.groupby('year', as_index=False)['Ev_of_ell_I'].transform(lambda x: filter(x, 100))['Ev_of_ell_I'].values
cps_intercept.loc[cps_intercept.loc[:, 'ell_intercept_I'] > 1, 'ell_intercept_I'] = 1
cps.loc[cps.loc[:, 'Ev_of_ell_I'] > 0, 'Ev_of_ell_I'] = 0

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.latin == 0) & dignity.year.isin(years), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Create a data frame
df = pd.DataFrame({'year': years, 'LE': np.zeros(len(years)),
                                  'C':  np.zeros(len(years)),
                                  'CI': np.zeros(len(years)),
                                  'L':  np.zeros(len(years)),
                                  'LI': np.zeros(len(years)),
                                  'I':  np.zeros(len(years))})

# Calculate the consumption-equivalent welfare of Black non-Latino relative to White non-Latino Americans with the incarceration adjustment
for year in years:
    S_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar'].values
    ell_i_bar = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'ell_bar'].values
    ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    I_intercept = ncr_u_bar.loc[:, 'incarceration'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    c_intercept_I = cex_intercept.loc[:, 'c_intercept_I'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    ell_intercept_I = cps_intercept.loc[:, 'ell_intercept_I'].values
    c_i_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = dignity.loc[(dignity.year == year) & (dignity.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = dignity.loc[(dignity.year == year) & (dignity.race == 2), 'Ev_of_ell'].values
    Elog_of_c_I = cex.loc[(cex.year == year), 'Elog_of_c_I'].values
    Ev_of_ell_I = cps.loc[(cps.year == year), 'Ev_of_ell_I'].values
    I_i = ncr.loc[(ncr.year == year) & (ncr.race == 1), 'incarceration'].values
    I_j = ncr.loc[(ncr.year == year) & (ncr.race == 2), 'incarceration'].values
    for i in ['LE', 'C', 'CI', 'L', 'LI', 'I']:
        df.loc[df.year == year, i] = cew_level_incarceration(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                                                             S_intercept=S_intercept, I_intercept=I_intercept, c_intercept=c_intercept, c_intercept_I=c_intercept_I, ell_intercept=ell_intercept, ell_intercept_I=ell_intercept_I, c_nominal=c_nominal,
                                                             c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j,
                                                             Elog_of_c_I=Elog_of_c_I, Ev_of_ell_I=Ev_of_ell_I, I_i=I_i, I_j=I_j, incarceration_parameter=0.1)[i]

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.stackplot(years, [df.LE, df.C, df.I], colors=newnewcolors[2:], edgecolor='Black', linewidth=0.75)
ax.stackplot(years, [df.L, df.CI + df.LI], colors=[newnewcolors[1], newnewcolors[0]], edgecolor='Black', linewidth=0.75)
ax.arrow(2007, np.log(1.02), 0, 0.06, linewidth=1, color='Black')
ax.arrow(2011, np.log(0.485), 0, -0.07, linewidth=1, color='Black')

# Set the horizontal axis
ax.set_xlim(2006, 2019)

# Set the vertical axis
ax.set_ylim(np.log(0.4), np.log(1.12))
ax.set_yticks(np.log(np.linspace(0.4, 1, 7)))
ax.set_yticklabels(list(map('{0:.1f}'.format, np.linspace(0.4, 1, 7))))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(2007, np.log(1.1), 'Leisure', fontsize=12, ha='center')
ax.text(2011.5, np.log(1.13), 'Inequality', fontsize=12, ha='center')
ax.text(2008, np.log(0.80), 'Life expectancy', fontsize=12, ha='center')
ax.text(2008, np.log(0.53), 'Consumption', fontsize=12, ha='center')
ax.text(2011, np.log(0.43), 'Incarceration', fontsize=12, ha='center')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and incarceration decomposition.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the unemployment+ rate of Black and White   #
# Americans from 1984 to 2019.                                                 #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2019 + 1)

# Load the cps data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[cps.year.isin(years) & cps.race.isin([1, 2]), :]

# Calculate the unemployment+ rate
df = pd.merge(cps.loc[cps.status == 'unemployed', :].groupby(['year', 'race'], as_index=False).agg({'weight': 'sum'}).rename(columns={'weight': 'unemployed'}),
              cps.loc[cps.laborforce == 1, :].groupby(['year', 'race'], as_index=False).agg({'weight': 'sum'}).rename(columns={'weight': 'laborforce'}), how='left')
df.loc[:, 'unemployed'] = df.unemployed / df.laborforce

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, 100 * df.loc[df.race == 1, 'unemployed'], color=colors[0], linewidth=2.5)
ax.annotate('White', xy=(2019.25, 100 * df.loc[df.race == 1, 'unemployed'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(years, 100 * df.loc[df.race == 2, 'unemployed'], color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2019.25, 100 * df.loc[df.race == 2, 'unemployed'].iloc[-1]), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

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
# This section of the script plots the unemployment adjustment to the leisure  #
# gap between Black and White Americans from 1984 to 2019.                     #
#                                                                              #
################################################################################

# Load the cps data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))

# Calculate average leisure by year and race
cps.loc[:, 'adjusted_weekly_leisure'] = cps.weekly_leisure - cps.Δ_leisure
df = cps.loc[cps.year.isin(range(1984, 2019 + 1)), :].groupby(['year', 'race'], as_index=False).apply(lambda x: pd.Series({'leisure':                 np.average(x.leisure, weights=x.weight),
                                                                                                                           'adjusted_weekly_leisure': np.average(x.adjusted_weekly_leisure, weights=x.weight)}))

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(df.year.unique(), 100 * (df.loc[df.race == 2, 'leisure'].values - df.loc[df.race == 1, 'leisure'].values), color=colors[1], linewidth=2.5)
ax.annotate('Unadjusted gap', xy=(2012, 3.2), color='k', fontsize=12, ha='center', va='center', annotation_clip=False)
ax.plot(df.year.unique(), 100 * (df.loc[df.race == 2, 'adjusted_weekly_leisure'].values - df.loc[df.race == 1, 'adjusted_weekly_leisure'].values), color=colors[1], linewidth=2.5, linestyle='dashed')
ax.annotate('Adjusted gap', xy=(2000, 1.45), color='k', fontsize=12, ha='center', va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axis
ax.set_ylim(0, 5)
ax.set_ylabel('$\%$ (p.p.)', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Leisure and unemployment.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption-equivalent welfare          #
# unemployment adjustment of Black relative to White Americans in 2019.        #
#                                                                              #
################################################################################

# Load the cps data
cps = pd.read_csv(os.path.join(cps_f_data, 'cps.csv'))
cps = cps.loc[(cps.year == 2019) & cps.race.isin([1, 2]), :]

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == False) & (dignity.latin == -1) & (dignity.year == 2019), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Create a data frame
df = pd.DataFrame({'parameter': np.linspace(0, 1, 101)})
df.loc[:, 'log_lambda'] = np.nan

# Calculate the consumption-equivalent welfare of Black relative to White Americans without the unemployment adjustment
df_cps = cps.groupby(['race', 'age'], as_index=False).apply(lambda x: pd.Series({'Ev_of_ell': np.average(v_of_ell(x.weekly_leisure), weights=x.weight),
                                                                                 'ell_bar':   np.average(x.weekly_leisure, weights=x.weight)}))
df_cps_intercept = cps.groupby('age', as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.weekly_leisure, weights=x.weight)}))
df_cps = pd.merge(expand({'race': [1, 2], 'age': range(101)}), df_cps, how='left')
df_cps_intercept = pd.merge(pd.DataFrame({'age': range(101)}), df_cps_intercept, how='left')
df_cps.loc[:, ['Ev_of_ell', 'ell_bar']] = df_cps.groupby('race', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
df_cps_intercept.loc[:, 'ell_bar'] = df_cps_intercept.loc[:, 'ell_bar'].transform(lambda x: filter(x, 100)).values
df_cps.loc[df_cps.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
df_cps.loc[df_cps.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
df_cps_intercept.loc[df_cps_intercept.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
S_i = dignity.loc[(dignity.race == 1), 'S'].values
S_j = dignity.loc[(dignity.race == 2), 'S'].values
c_i_bar = dignity.loc[(dignity.race == 1), 'c_bar'].values
c_j_bar = dignity.loc[(dignity.race == 2), 'c_bar'].values
ell_i_bar = df_cps.loc[(df_cps.race == 1), 'ell_bar'].values
ell_j_bar = df_cps.loc[(df_cps.race == 2), 'ell_bar'].values
S_intercept = dignity_intercept.loc[:, 'S'].values
c_intercept = dignity_intercept.loc[:, 'c_bar'].values
ell_intercept = df_cps_intercept.loc[:, 'ell_bar'].values
c_i_bar_nd = dignity.loc[(dignity.race == 1), 'c_bar_nd'].values
c_j_bar_nd = dignity.loc[(dignity.race == 2), 'c_bar_nd'].values
Elog_of_c_i = dignity.loc[(dignity.race == 1), 'Elog_of_c'].values
Elog_of_c_j = dignity.loc[(dignity.race == 2), 'Elog_of_c'].values
Elog_of_c_i_nd = dignity.loc[(dignity.race == 1), 'Elog_of_c_nd'].values
Elog_of_c_j_nd = dignity.loc[(dignity.race == 2), 'Elog_of_c_nd'].values
Ev_of_ell_i = df_cps.loc[(df_cps.race == 1), 'Ev_of_ell'].values
Ev_of_ell_j = df_cps.loc[(df_cps.race == 2), 'Ev_of_ell'].values
d = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
              S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
              inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)
log_lambda = np.sum([d.get(x) for x in ['LE', 'C', 'CI', 'L']])

# Calculate the consumption-equivalent welfare of Black relative to White Americans with the unemployment adjustment
for i in np.linspace(0, 1, 101):
    cps.loc[:, 'adjusted_leisure'] = cps.weekly_leisure - (1 - i) * cps.Δ_leisure
    df_cps = cps.groupby(['race', 'age'], as_index=False).apply(lambda x: pd.Series({'Ev_of_ell': np.average(v_of_ell(x.adjusted_leisure), weights=x.weight),
                                                                                     'ell_bar':   np.average(x.adjusted_leisure, weights=x.weight)}))
    df_cps_intercept = cps.groupby('age', as_index=False).apply(lambda x: pd.Series({'ell_bar': np.average(x.weekly_leisure, weights=x.weight)}))
    df_cps = pd.merge(expand({'race': [1, 2], 'age': range(101)}), df_cps, how='left')
    df_cps_intercept = pd.merge(pd.DataFrame({'age': range(101)}), df_cps_intercept, how='left')
    df_cps.loc[:, ['Ev_of_ell', 'ell_bar']] = df_cps.groupby('race', as_index=False)[['Ev_of_ell', 'ell_bar']].transform(lambda x: filter(x, 100)).values
    df_cps_intercept.loc[:, 'ell_bar'] = df_cps_intercept.loc[:, 'ell_bar'].transform(lambda x: filter(x, 100)).values
    df_cps.loc[df_cps.loc[:, 'Ev_of_ell'] > 0, 'Ev_of_ell'] = 0
    df_cps.loc[df_cps.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    df_cps_intercept.loc[df_cps_intercept.loc[:, 'ell_bar'] > 1, 'ell_bar'] = 1
    S_i = dignity.loc[(dignity.race == 1), 'S'].values
    S_j = dignity.loc[(dignity.race == 2), 'S'].values
    c_i_bar = dignity.loc[(dignity.race == 1), 'c_bar'].values
    c_j_bar = dignity.loc[(dignity.race == 2), 'c_bar'].values
    ell_i_bar = df_cps.loc[(df_cps.race == 1), 'ell_bar'].values
    ell_j_bar = df_cps.loc[(df_cps.race == 2), 'ell_bar'].values
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = df_cps_intercept.loc[:, 'ell_bar'].values
    c_i_bar_nd = dignity.loc[(dignity.race == 1), 'c_bar_nd'].values
    c_j_bar_nd = dignity.loc[(dignity.race == 2), 'c_bar_nd'].values
    Elog_of_c_i = dignity.loc[(dignity.race == 1), 'Elog_of_c'].values
    Elog_of_c_j = dignity.loc[(dignity.race == 2), 'Elog_of_c'].values
    Elog_of_c_i_nd = dignity.loc[(dignity.race == 1), 'Elog_of_c_nd'].values
    Elog_of_c_j_nd = dignity.loc[(dignity.race == 2), 'Elog_of_c_nd'].values
    Ev_of_ell_i = df_cps.loc[(df_cps.race == 1), 'Ev_of_ell'].values
    Ev_of_ell_j = df_cps.loc[(df_cps.race == 2), 'Ev_of_ell'].values
    d = cew_level(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar,
                  S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                  inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)
    df.loc[df.parameter == i, 'log_lambda'] = np.sum([d.get(x) for x in ['LE', 'C', 'CI', 'L']])

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(np.linspace(0, 1, 101), 100 * (df.log_lambda - log_lambda), color=colors[1], linewidth=2.5)

# Set the horizontal axis
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xticklabels(np.linspace(0, 100, 11).astype('int'))
ax.set_xlabel(r'Fraction of extra time treated as leisure (\%)', fontsize=12, rotation=0, ha='center')

# Set the vertical axis
ax.set_ylim(-1.2, 0)
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and unemployment.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots consumption-equivalent welfare growth by    #
# decade for Black and White Americans from 1940 to 2019.                      #
#                                                                              #
################################################################################

# Define a list of years
years = list(range(1940, 2010 + 1, 10)) + [2019]

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == True) & (dignity.race != -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the 95% confidence interval
#df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
#df_bs = df_bs.loc[(df_bs.description == 'Welfare growth historical'), :]
#df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'lb'}),
#                 df_bs.groupby(['year', 'race'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'ub'}), how='left')

# Calculate consumption-equivalent welfare growth
df = expand({'year': years[1:], 'race': [1, 2]})
df.loc[:, 'log_lambda'] = np.nan
for race in [1, 2]:
    for year in years[1:]:
        S_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'S'].values
        S_j = dignity.loc[(dignity.year == year) & (dignity.race == race), 'S'].values
        c_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'c_bar'].values
        c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'c_bar'].values
        ell_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'ell_bar'].values
        ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'ell_bar'].values
        T = year - years[years.index(year) - 1]
        S_intercept = dignity_intercept.loc[:, 'S'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        df.loc[(df.year == year) & (df.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                 S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)['log_lambda']

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(np.linspace(1950, 2020, 8), 100 * df.loc[df.race == 1, 'log_lambda'], color=colors[0], linewidth=2.5, marker='o', clip_on=False)
#ax.fill_between(np.linspace(1950, 2020, 8), 100 * df_bs.loc[df_bs.race == 1, 'lb'], y2=100 * df_bs.loc[df_bs.race == 1, 'ub'], color=colors[0], alpha=0.2, linewidth=0)
ax.plot(np.linspace(1950, 2020, 8), 100 * df.loc[df.race == 2, 'log_lambda'], color=colors[1], linewidth=2.5, marker='o', clip_on=False)
#ax.fill_between(np.linspace(1950, 2020, 8), 100 * df_bs.loc[df_bs.race == 2, 'lb'], y2=100 * df_bs.loc[df_bs.race == 2, 'ub'], color=colors[1], alpha=0.2, linewidth=0)

# Set the horizontal axis
ax.set_xlim(1950, 2020)
ax.set_xticks(np.linspace(1950, 2020, 8))
ax.set_xticklabels([str(year) + 's' for year in range(1940, 2010 + 1, 10)])

# Set the vertical axis
ax.set_ylim(0.5, 9)
ax.set_yticks(np.linspace(1, 9, 9))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(1960, 1.3, 'White', fontsize=12, ha='center', color='k')
ax.text(2000, 4.9, 'Black', fontsize=12, ha='center', color='k')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare growth historical.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots consumption-equivalent welfare growth and   #
# consumption growth by decade from 1940 to 2019.                              #
#                                                                              #
################################################################################

# Define a list of years
years = list(range(1940, 2010 + 1, 10)) + [2019]

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == True) & (dignity.race == -1) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the 95% confidence interval
#df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
#df_bs = df_bs.loc[(df_bs.description == 'Welfare and consumption growth historical'), :]
#df_bs = pd.merge(df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025), 'C': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'log_lambda_lb', 'C': 'C_lb'}),
#                 df_bs.groupby('year', as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975), 'C': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'log_lambda_ub', 'C': 'C_ub'}), how='left')

# Calculate consumption-equivalent welfare growth
df = pd.DataFrame({'year': years[1:], 'log_lambda': np.zeros(len(years[1:])), 'C': np.zeros(len(years[1:]))})
for year in years[1:]:
    S_i = dignity.loc[dignity.year == years[years.index(year) - 1], 'S'].values
    S_j = dignity.loc[dignity.year == year, 'S'].values
    c_i_bar = dignity.loc[dignity.year == years[years.index(year) - 1], 'c_bar'].values
    c_j_bar = dignity.loc[dignity.year == year, 'c_bar'].values
    ell_i_bar = dignity.loc[dignity.year == years[years.index(year) - 1], 'ell_bar'].values
    ell_j_bar = dignity.loc[dignity.year == year, 'ell_bar'].values
    T = year - years[years.index(year) - 1]
    S_intercept = dignity_intercept.loc[:, 'S'].values
    c_intercept = dignity_intercept.loc[:, 'c_bar'].values
    ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
    for i in ['log_lambda', 'C']:
        df.loc[df.year == year, i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)[i]

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(np.linspace(1950, 2020, 8), 100 * df.log_lambda, color=colors[1], linewidth=2.5, marker='o', clip_on=False)
#ax.fill_between(np.linspace(1950, 2020, 8), 100 * df_bs.log_lambda_lb, y2=100 * df_bs.log_lambda_ub, color=colors[1], alpha=0.2, linewidth=0)
ax.plot(np.linspace(1950, 2020, 8), 100 * df.C, color=colors[0], linewidth=2.5, marker='o', clip_on=False)
#ax.fill_between(np.linspace(1950, 2020, 8), 100 * df_bs.C_lb, y2=100 * df_bs.C_ub, color=colors[0], alpha=0.2, linewidth=0)

# Set the horizontal axis
ax.set_xlim(1950, 2020)
ax.set_xticks(np.linspace(1950, 2020, 8))
ax.set_xticklabels([str(year) + 's' for year in range(1940, 2010 + 1, 10)])

# Set the vertical axis
ax.set_ylim(0.5, 7)
ax.set_yticks(np.linspace(1, 7, 7))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(1980, 4.2, 'Welfare', fontsize=12, ha='center', color='k')
ax.text(1980, 1.4, 'Consumption', fontsize=12, ha='center', color='k')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Welfare and consumption growth historical.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots cumulative consumption-equivalent welfare   #
# growth for Black and White Americans from 1984 to 2019.                      #
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
        c_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'c_bar'].values
        c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'c_bar'].values
        ell_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'ell_bar'].values
        ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'ell_bar'].values
        T = year - years[years.index(year) - 1]
        S_intercept = dignity_intercept.loc[:, 'S'].values
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
        df.loc[(df.year == year) & (df.race == race), 'log_lambda'] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                                 S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal,
                                                                                 inequality=True, c_i_bar_nd=c_i_bar_nd, c_j_bar_nd=c_j_bar_nd, Elog_of_c_i=Elog_of_c_i, Elog_of_c_j=Elog_of_c_j, Elog_of_c_i_nd=Elog_of_c_i_nd, Elog_of_c_j_nd=Elog_of_c_j_nd, Ev_of_ell_i=Ev_of_ell_i, Ev_of_ell_j=Ev_of_ell_j)['log_lambda']

# Cumulate the growth rates
df.loc[:, 'log_lambda'] = df.groupby('race', as_index=False).log_lambda.transform(lambda x: np.exp(np.cumsum(x))).log_lambda.values

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, np.log(np.append(1, df.loc[df.race == 1, 'log_lambda'])), color=colors[0], linewidth=2.5)
ax.fill_between(years, np.log(np.append(1, df_bs.loc[df_bs.race == 1, 'lb'])), y2=np.log(np.append(1, df_bs.loc[df_bs.race == 1, 'ub'])), color=colors[0], alpha=0.2, linewidth=0)
ax.annotate(str(np.round_(df.loc[df.race == 1, 'log_lambda'].iloc[-1], decimals=1)) + 'x', xy=(2019.25, np.log(df.loc[df.race == 1, 'log_lambda'].iloc[-1])), color='k', fontsize=10, va='center', annotation_clip=False)
ax.plot(years, np.log(np.append(1, df.loc[df.race == 2, 'log_lambda'])), color=colors[1], linewidth=2.5)
ax.fill_between(years, np.log(np.append(1, df_bs.loc[df_bs.race == 2, 'lb'])), y2=np.log(np.append(1, df_bs.loc[df_bs.race == 2, 'ub'])), color=colors[1], alpha=0.2, linewidth=0)
ax.annotate(str(np.round_(df.loc[df.race == 2, 'log_lambda'].iloc[-1], decimals=1)) + 'x', xy=(2019.25, np.log(df.loc[df.race == 2, 'log_lambda'].iloc[-1])), color='k', fontsize=10, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1984, 2019)
ax.set_xticks(np.append(np.linspace(1985, 2015, 7), 2019))

# Set the vertical axis
ax.set_ylim(np.log(1), np.log(3.5))
ax.set_yticks(np.log(np.linspace(1, 3.5, 6)))
ax.set_yticklabels(np.linspace(1, 3.5, 6))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(2012, np.log(2.12), 'White', fontsize=12, ha='center', color='k')
ax.text(2012, np.log(3.1), 'Black', fontsize=12, ha='center', color='k')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Cumulative welfare growth.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots cumulative consumption-equivalent welfare   #
# growth for Black and White Americans from 1940 to 2019.                      #
#                                                                              #
################################################################################

# Define a list of years
years = list(range(1940, 2010 + 1, 10)) + [2019]

# Load the dignity data
dignity = pd.read_csv(os.path.join(f_data, 'dignity.csv'))
dignity_intercept = dignity.loc[(dignity.historical == False) & (dignity.race == -1) & (dignity.latin == -1) & (dignity.year == 2006), :]
dignity = dignity.loc[(dignity.historical == True) & (dignity.latin == -1), :]

# Retrieve nominal consumption per capita in 2006
c_nominal = bea.data('nipa', tablename='t20405', frequency='a', year=2006).data.DPCERC
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=2006).data.B230RC
c_nominal = 1e6 * c_nominal / population

# Calculate the 95% confidence interval
#df_bs = pd.read_csv(os.path.join(f_data, 'cew_bootstrap.csv'))
#df_bs = df_bs.loc[(df_bs.description == 'Cumulative welfare growth historical'), :]
#df_bs = pd.merge(df_bs.groupby(['year', 'race'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.025), 'C': lambda x: x.quantile(q=0.025)}).rename(columns={'log_lambda': 'log_lambda_lb', 'C': 'C_lb'}),
#                 df_bs.groupby(['year', 'race'], as_index=False).agg({'log_lambda': lambda x: x.quantile(q=0.975), 'C': lambda x: x.quantile(q=0.975)}).rename(columns={'log_lambda': 'log_lambda_ub', 'C': 'C_ub'}), how='left')

# Calculate consumption-equivalent welfare growth
df = expand({'year': years[1:], 'race': [-1, 1, 2]})
df.loc[:, 'log_lambda'] = np.nan
df.loc[:, 'C'] = np.nan
for race in [-1, 1, 2]:
    for year in years[1:]:
        S_i = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'S'].values
        S_j = dignity.loc[(dignity.year == year) & (dignity.race == race), 'S'].values
        c_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'c_bar'].values
        c_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'c_bar'].values
        ell_i_bar = dignity.loc[(dignity.year == years[years.index(year) - 1]) & (dignity.race == race), 'ell_bar'].values
        ell_j_bar = dignity.loc[(dignity.year == year) & (dignity.race == race), 'ell_bar'].values
        T = year - years[years.index(year) - 1]
        S_intercept = dignity_intercept.loc[:, 'S'].values
        c_intercept = dignity_intercept.loc[:, 'c_bar'].values
        ell_intercept = dignity_intercept.loc[:, 'ell_bar'].values
        for i in ['log_lambda', 'C']:
            df.loc[(df.year == year) & (df.race == race), i] = cew_growth(S_i=S_i, S_j=S_j, c_i_bar=c_i_bar, c_j_bar=c_j_bar, ell_i_bar=ell_i_bar, ell_j_bar=ell_j_bar, T=T,
                                                                          S_intercept=S_intercept, c_intercept=c_intercept, ell_intercept=ell_intercept, c_nominal=c_nominal)[i]

# Cumulate the growth rates
df.loc[:, ['log_lambda', 'C']] = df.groupby('race', as_index=False)[['log_lambda', 'C']].transform(lambda x: np.exp(np.cumsum(x * np.diff(years)))).values

# Initialize the figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(years, np.log(np.append(1, df.loc[df.race == 1, 'log_lambda'])), color=colors[0], linewidth=2.5)
#ax.fill_between(years, np.log(np.append(1, df_bs.loc[df_bs.race == 1, 'log_lambda_lb'])), y2=np.log(np.append(1, df_bs.loc[df_bs.race == 1, 'log_lambda_ub'])), color=colors[0], alpha=0.2, linewidth=0)
ax.annotate(str(df.loc[df.race == 1, 'log_lambda'].iloc[-1].astype('int')) + 'x', xy=(2019.25, np.log(df.loc[df.race == 1, 'log_lambda'].iloc[-1])), color='k', fontsize=10, va='center', annotation_clip=False)
ax.plot(years, np.log(np.append(1, df.loc[df.race == 2, 'log_lambda'])), color=colors[1], linewidth=2.5)
#ax.fill_between(years, np.log(np.append(1, df_bs.loc[df_bs.race == 2, 'log_lambda_lb'])), y2=np.log(np.append(1, df_bs.loc[df_bs.race == 2, 'log_lambda_ub'])), color=colors[1], alpha=0.2, linewidth=0)
ax.annotate(str(df.loc[df.race == 2, 'log_lambda'].iloc[-1].astype('int')) + 'x', xy=(2019.25, np.log(df.loc[df.race == 2, 'log_lambda'].iloc[-1])), color='k', fontsize=10, va='center', annotation_clip=False)
ax.plot(years, np.log(np.append(1, df.loc[df.race == -1, 'C'])), color='k', linewidth=2.5)
#ax.fill_between(years, np.log(np.append(1, df_bs.loc[df_bs.race == -1, 'C_lb'])), y2=np.log(np.append(1, df_bs.loc[df_bs.race == -1, 'C_ub'])), color='k', alpha=0.2, linewidth=0)
ax.annotate(str(df.loc[df.race == -1, 'C'].iloc[-1].astype('int')) + 'x', xy=(2019.25, np.log(df.loc[df.race == -1, 'C'].iloc[-1])), color='k', fontsize=10, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1940, 2019)
ax.set_xticks(np.append(np.linspace(1940, 2010, 8), 2019))

# Set the vertical axis
ax.set_ylim(np.log(0.9), np.log(32))
ax.set_yticks(np.log(np.array([2**n for n in range(6)])))
ax.set_yticklabels(np.array([2**n for n in range(6)]))

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set the figure's text
ax.text(2010, np.log(7.5), 'White welfare', fontsize=12, ha='center', color='k')
ax.text(1991, np.log(17), 'Black welfare', fontsize=12, ha='center', color='k')
ax.text(2000, np.log(2.3), 'Consumption (all races)', fontsize=12, ha='center', color='k')

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Cumulative welfare growth historical.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script plots the consumption to disposable income ratio  #
# by year for Black and White Americans from 1991 to 2019.                     #
#                                                                              #
################################################################################

# Define a list of years
years = range(1991, 2019 + 1)

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
ax.annotate('White', xy=(2009.5, 82.5), color='k', fontsize=12, va='center', annotation_clip=False)
ax.plot(years, 100 * cex.loc[cex.race == 2, 'consumption'].values / cps.loc[cps.race == 2, 'income'].values, color=colors[1], linewidth=2.5)
ax.annotate('Black', xy=(2009.5, 90.5), color='k', fontsize=12, va='center', annotation_clip=False)

# Set the horizontal axis
ax.set_xlim(1991, 2019)
ax.set_xticks(np.append(1991, np.append(np.linspace(1995, 2015, 5), 2019)))

# Set the vertical axis
ax.set_ylim(75, 95)
ax.set_yticks(np.linspace(75, 95, 5))
ax.set_ylabel('$\%$', fontsize=12, rotation=0, ha='center', va='center')
ax.yaxis.set_label_coords(0, 1.1)

# Remove the top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(figures, 'Consumption to disposable income ratio.pdf'), format='pdf')
plt.close()
