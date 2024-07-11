# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import os

# Import functions and directories
from functions import *
from directories import *

# Start the BEA client
bea = beapy.BEA(key=bea_api_key)

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
                                      (605, 616)] # FINLWT21
    elif (year >= 1986) & (year <= 1995):
        columns[years.index(year)] = [(0, 8),     # NEWID
                                      (423, 434)] # FINLWT21
    else:
        columns[years.index(year)] = ['NEWID',
                                      'FINLWT21']

# Define the variable names of the FMLI data files
names = ['NEWID',
         'FINLWT21']

# Define the variable types of the FMLI data files
types = {'NEWID':    'str',
         'FINLWT21': 'float'}

# Initialize a data frame
cex_fmli = pd.DataFrame()

# Process the FMLI data files
for year in years:
    # Initialize a data frame
    df = pd.DataFrame()

    # Load the data
    for interview in range(2, 5 + 1):
        if (year >= 1984) & (year <= 1995):
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".txt"
            else:
                suffix = str(year)[2:] + str(interview) + ".txt"
            df_fmli = pd.read_fwf(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'fmlyi' + suffix), colspecs=columns[years.index(year)], names=names, dtype=types)
        elif (year == 2004) | (year == 2011) | (year == 2014) | (year == 2015):
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".dta"
            else:
                suffix = str(year)[2:] + str(interview) + ".dta"
            df_fmli = pd.read_stata(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "fmli" + suffix), columns=[string.lower() for string in columns[years.index(year)]]).rename(columns=dict(zip([string.lower() for string in columns[years.index(year)]], columns[years.index(year)])))
            df_fmli = df_fmli.astype({key: types[key] for key in df_fmli.columns})
        else:
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".csv"
            else:
                suffix = str(year)[2:] + str(interview) + ".csv"
            df_fmli = pd.read_csv(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "fmli" + suffix), usecols=columns[years.index(year)])
            df_fmli = df_fmli.replace(".", np.nan)
            df_fmli = df_fmli.astype({key: types[key] for key in df_fmli.columns})

        # Append the data frames for all interviews
        df = pd.concat([df, df_fmli], ignore_index=True)

    # Recode the NEWID variable
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))

    # Create the year variable and append the data frames for all years
    df.loc[:, 'year'] = year
    cex_fmli = pd.concat([cex_fmli, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script processes the CEX MTBI and ITBI data files.       #
#                                                                              #
################################################################################

# Define the variable columns of the MTBI and ITBI data files
mtbi_columns = [None] * len(years)
itbi_columns = [None] * len(years)
for year in years:
    if (year >= 1984) & (year <= 1995):
        mtbi_columns[years.index(year)] = [(0, 8), (8, 14), (14, 26), (27, 28)]
        itbi_columns[years.index(year)] = [(0, 8), (12, 18), (19, 31)]
    else:
        mtbi_columns[years.index(year)] = ['NEWID', 'UCC', 'COST', 'GIFT']
        itbi_columns[years.index(year)] = ['NEWID', 'UCC', 'VALUE']

# Define the variable names of the MTBI and ITBI data files
mtbi_names = ['NEWID', 'UCC', 'COST', 'GIFT']
itbi_names = ['NEWID', 'UCC', 'VALUE']

# Define the variable types of the MTBI and ITBI data files
mtbi_types = {'NEWID': 'str', 'UCC': 'int', 'COST': 'float', 'GIFT': 'int'}
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
    for interview in range(2, 5 + 1):
        if (year >= 1984) & (year <= 1995):
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".txt"
            else:
                suffix = str(year)[2:] + str(interview) + ".txt"
            df_mtbi = pd.read_fwf(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'mtabi' + suffix), colspecs=mtbi_columns[years.index(year)], names=mtbi_names, dtype=mtbi_types)
            df_itbi = pd.read_fwf(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'itabi' + suffix), colspecs=itbi_columns[years.index(year)], names=itbi_names, dtype=itbi_types).rename(columns={'VALUE': 'COST'})
            df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)
        elif (year == 2004) | (year == 2011) | (year == 2014) | (year == 2015):
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".dta"
            else:
                suffix = str(year)[2:] + str(interview) + ".dta"
            df_mtbi = pd.read_stata(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "mtbi" + suffix), columns=[string.lower() for string in mtbi_columns[years.index(year)]]).rename(columns=dict(zip([string.lower() for string in mtbi_columns[years.index(year)]], mtbi_columns[years.index(year)])))
            df_itbi = pd.read_stata(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "itbi" + suffix), columns=[string.lower() for string in itbi_columns[years.index(year)]]).rename(columns=dict(zip([string.lower() for string in itbi_columns[years.index(year)]], itbi_columns[years.index(year)])))
            df_mtbi = df_mtbi.astype({key: mtbi_types[key] for key in df_mtbi.columns})
            df_itbi = df_itbi.astype({key: itbi_types[key] for key in df_itbi.columns})
            df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)
        else:
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".csv"
            else:
                suffix = str(year)[2:] + str(interview) + ".csv"
            df_mtbi = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'mtbi' + suffix), usecols=mtbi_columns[years.index(year)], dtype=mtbi_types)
            df_itbi = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'itbi' + suffix), usecols=itbi_columns[years.index(year)], dtype=itbi_types)
            df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)

        # Append the data frames for all interviews
        df = pd.concat([df, df_expenditures], ignore_index=True)

    # Recode the NEWID variable
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))

    # Merge the expenditures data frame with the UCC dictionary
    df = pd.merge(df, ucc, how='left')
    df.loc[:, 'COST'] = df.COST * df.factor

    # Only keep non-gift consumption expenditures
    df = df.loc[(df.GIFT != 1) & (df.consumption == 1), :]

    # Re-scale food at home expenditures before 1987
    if year <= 1987:
        df.loc[df.level2 == 'FDHOME', 'COST'] = df.COST / np.exp(-0.10795)

    # Aggregate expenditures, create the year variable and append the data frames for all years
    df = df.groupby(['NEWID', 'UCC'], as_index=False).agg({'COST': 'sum'})
    df.loc[:, 'year'] = year
    cex_expenditures = pd.concat([cex_expenditures, df], ignore_index=True)

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
cex = pd.merge(cex, cw, how='inner')
cex.loc[:, 'COST'] = cex.COST * cex.scale
cex = cex.groupby(['year', 'series'], as_index=False).agg({'COST': 'sum'})

################################################################################
#                                                                              #
# This section of the script processes the NIPA PCE data.                      #
#                                                                              #
################################################################################

# Load and reshape the NIPA PCE data
nipa_pce = bea.data('underlying', tablename='u20405', frequency='a', year=range(1984, 2022 + 1)).data.reset_index().rename(columns={'': 'year'})
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
cex = cex.groupby(['year', 'UCC'], as_index=False).apply(lambda x: pd.Series({'ratio': np.average(x.ratio, weights=x.share)}))

# Eliminate ratios of zero or infinity
cex = cex.loc[(cex.ratio != np.inf) & (cex.ratio != 0), :]

# Save the data
cex.to_csv(os.path.join(cex_f_data, 'nipa_pce.csv'), index=False)