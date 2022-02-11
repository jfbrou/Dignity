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

################################################################################
#                                                                              #
# This section of the script processes the CEX FMLI data files.                #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2020 + 1)

# Define the variable columns of the FMLI data files
columns = [None] * len(years)
for year in years:
    if (year >= 1984) & (year <= 1985):
        columns[years.index(year)] = [(0, 8),     # NEWID
                                      (296, 298), # FAM_SIZE
                                      (605, 616)] # FINLWT21
    elif (year >= 1986) & (year <= 1994):
        columns[years.index(year)] = [(0, 8),     # NEWID
                                      (334, 336), # FAM_SIZE
                                      (423, 434)] # FINLWT21
    else:
        columns[years.index(year)] = ['NEWID',
                                      'FAM_SIZE',
                                      'FINLWT21']

# Define the variable names of the FMLI data files
names = ['NEWID',
         'FAM_SIZE',
         'FINLWT21']

# Define the variable types of the FMLI data files
types = {'NEWID':    'str',
         'FAM_SIZE': 'int',
         'FINLWT21': 'float'}

# Initialize a data frame
cex_fmli = pd.DataFrame()

# Process the FMLI data files
for year in years:
    # Initialize a data frame
    df = pd.DataFrame()

    # Load the data
    for interview in range(1, 5 + 1):
        if (year >= 1984) & (year <= 1994):
            df_fmli = pd.read_fwf(os.path.join(cex_r_data, 'CEX' + str(year), 'fmli' + str(interview) + '.txt'), colspecs=columns[years.index(year)], names=names, dtype=types)
        else:
            df_fmli = pd.read_csv(os.path.join(cex_r_data, 'CEX' + str(year), 'fmli' + str(interview) + '.csv'), usecols=columns[years.index(year)], dtype=types)

        # Append the data frames for all interviews
        df = df.append(df_fmli, ignore_index=True)

    # Recode the NEWID variable
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))

    # Create the year variable and append the data frames for all years
    df.loc[:, 'year'] = year
    cex_fmli = cex_fmli.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script processes the CEX MTBI and ITBI data files.       #
#                                                                              #
################################################################################

# Define the variable columns of the MTBI and ITBI data files
mtbi_columns = [None] * len(years)
itbi_columns = [None] * len(years)
for year in years:
    if (year >= 1984) & (year <= 1994):
        mtbi_columns[years.index(year)] = [(0, 8), (8, 14), (14, 26)]
        itbi_columns[years.index(year)] = [(0, 8), (12, 18), (19, 31)]
    else:
        mtbi_columns[years.index(year)] = ['NEWID', 'UCC', 'COST']
        itbi_columns[years.index(year)] = ['NEWID', 'UCC', 'VALUE']

# Define the variable names of the MTBI and ITBI data files
mtbi_names = ['NEWID', 'UCC', 'COST']
itbi_names = ['NEWID', 'UCC', 'VALUE']

# Define the variable types of the MTBI and ITBI data files
mtbi_types = {'NEWID': 'str', 'UCC': 'int', 'COST': 'float'}
itbi_types = {'NEWID': 'str', 'UCC': 'int', 'VALUE': 'float'}

# Load the UCC dictionary
ucc = pd.read_csv(os.path.join(cex_r_data, 'ucc.csv'))

# Initialize a data frame
cex_expenditures = pd.DataFrame()

# Process the MTBI and ITBI data files
for year in years:
    # Initialize a data frame
    df = pd.DataFrame()

    # Load the data
    for interview in range(1, 5 + 1):
        if (year >= 1984) & (year <= 1994):
            df_mtbi = pd.read_fwf(os.path.join(cex_r_data, 'CEX' + str(year), 'mtbi' + str(interview) + '.txt'), colspecs=mtbi_columns[years.index(year)], names=mtbi_names, dtype=mtbi_types)
            df_itbi = pd.read_fwf(os.path.join(cex_r_data, 'CEX' + str(year), 'itbi' + str(interview) + '.txt'), colspecs=itbi_columns[years.index(year)], names=itbi_names, dtype=itbi_types).rename(columns={'VALUE': 'COST'})
            df_expenditures = df_mtbi.append(df_itbi, ignore_index=True)
        else:
            df_mtbi = pd.read_csv(os.path.join(cex_r_data, 'CEX' + str(year), 'mtbi' + str(interview) + '.csv'), usecols=mtbi_columns[years.index(year)], dtype=mtbi_types)
            df_itbi = pd.read_csv(os.path.join(cex_r_data, 'CEX' + str(year), 'itbi' + str(interview) + '.csv'), usecols=itbi_columns[years.index(year)], dtype=itbi_types)
            df_expenditures = df_mtbi.append(df_itbi, ignore_index=True)

        # Append the data frames for all interviews
        df = df.append(df_expenditures, ignore_index=True)

    # Recode the NEWID variable
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))

    # Merge the expenditures data frame with the UCC dictionary
    df = pd.merge(df, ucc.loc[:, ['UCC', 'consumption', 'factor', 'category']], how='left')
    df.loc[:, 'COST'] = df.COST * df.factor

    # Re-scale food at home expenditures before 1987
    if year <= 1987:
        df.loc[df.category == 'FDHOME', 'COST'] = df.COST / np.exp(-0.10795)

    # Aggregate expenditures, create the year variable and append the data frames for all years
    dftemp = df.groupby(['NEWID', 'UCC'], as_index=False).agg({'COST': 'sum'})
    df.loc[:, 'year'] = year
    cex_expenditures = cex_expenditures.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script merges the FMLI, MTBI and ITBI data files, and    #
# aggregates the CEX expenditures.                                             #
#                                                                              #
################################################################################

# Aggregate the CEX expenditures by year and UCC
cex = pd.merge(cex_expenditures, cex_fmli, how='left')
cex.loc[:, 'COST'] = cex.COST * cex.FINLWT21
cex = cex.groupby(['year', 'UCC'], as_index=False).agg({'COST': 'sum'})

# Load the CEX to NIPA PCE crosswalk and aggregate CEX consumption with respect to NIPA PCE categories
cw = pd.read_csv(os.path.join(cex_r_data, 'crosswalk.csv'))
cw = cw.loc[cw.scale != 0, :]
cw.loc[:, 'series'] = cw.series.replace({'DCHCRC': 'C1', 'DNSCRC': 'C1',
                                         'DCOSRC': 'C2', 'DOPHRC': 'C2',
                                         'DWSMRC': 'C3', 'DELCRC': 'C3', 'DGHERC': 'C3', 'DREFRC': 'C3',
                                         'DMOVRC': 'C4', 'DLIGRC': 'C4', 'DMUSRC': 'C4',
                                         'DTHERC': 'C5', 'DTAPRC': 'C5'})
cex = pd.merge(cex, cw, how='inner')
cex.loc[:, 'COST'] = cex.COST * cex.scale
cex = cex.groupby(['year', 'series'], as_index=False).agg({'COST': 'sum'})

################################################################################
#                                                                              #
# This section of the script processes the NIPA PCE data.                      #
#                                                                              #
################################################################################

# Load and reshape the NIPA PCE data
nipa_pce = bea.data('underlying', tablename='u20405', frequency='a', year=range(1984, 2020 + 1)).data.reset_index().rename(columns={'': 'year'})
nipa_pce = pd.melt(nipa_pce, id_vars=['year'], value_vars=[column for column in nipa_pce.columns if column not in 'year'], var_name='series', value_name='expenditures')

# Calculate the NIPA PCE aggregates
nipa_pce.loc[:, 'series'] = nipa_pce.series.replace({'DCHCRC': 'C1', 'DNSCRC': 'C1',
                                                     'DCOSRC': 'C2', 'DOPHRC': 'C2',
                                                     'DWSMRC': 'C3', 'DELCRC': 'C3', 'DGHERC': 'C3', 'DREFRC': 'C3',
                                                     'DMOVRC': 'C4', 'DLIGRC': 'C4', 'DMUSRC': 'C4',
                                                     'DTHERC': 'C5', 'DTAPRC': 'C5'})
nipa_pce = nipa_pce.groupby(['year', 'series'], as_index=False).agg({'expenditures': 'sum'})
nipa_pce.loc[:, 'expenditures'] = 1e6 * nipa_pce.expenditures
nipa_pce = nipa_pce.astype({'year': 'int', 'series': 'str'})

################################################################################
#                                                                              #
# This section of the script merges the CEX and NIPA PCE data frames.          #
#                                                                              #
################################################################################

# Merge the CEX and NIPA PCE aggregates
cex = pd.merge(cex, nipa_pce, how='inner')
cex.loc[:, 'ratio'] = cex.COST / cex.expenditures
cex = cex.drop(['COST', 'expenditures'], axis=1)

# Merge back with the crosswalk
cex = pd.merge(cw.loc[:, ['UCC', 'series', 'share']], cex, how='inner')
cex = cex.groupby(['year', 'UCC'], as_index=False).agg({'ratio': lambda x: weighted_average(x, data=cex, weights='share')})

# Eliminate ratios of zero or infinity
cex = cex.loc[(cex.ratio != np.inf) & (cex.ratio != 0), :]

# Save the data
cex.to_csv(os.path.join(cex_f_data, 'nipa_pce.csv'), index=False)
