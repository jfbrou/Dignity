# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import statsmodels.formula.api as smf
import os
import calendar

# Import functions and directories
from functions import *
from directories import *

# Define a list of years
years = range(1984, 2022 + 1)

################################################################################
#                                                                              #
# This section of the script defines the properties of the CEX MTBI and ITBI   #
# data files.                                                                  #
#                                                                              #
################################################################################

# Define the variable columns of the MTBI and ITBI data files
mtbi_columns = [None] * len(years)
itbi_columns = [None] * len(years)
for year in years:
    if (year >= 1984) & (year <= 1989):
        mtbi_columns[years.index(year)] = [(0, 8), (8, 14), (14, 26), (27, 28)]
        itbi_columns[years.index(year)] = [(0, 8), (12, 18), (19, 31)]
    else:
        mtbi_columns[years.index(year)] = ["NEWID", "UCC", "COST", "GIFT"]
        itbi_columns[years.index(year)] = ["NEWID", "UCC", "VALUE"]

# Define the variable names of the MTBI and ITBI data files
mtbi_names = ["NEWID", "UCC", "COST", "GIFT"]
itbi_names = ["NEWID", "UCC", "VALUE"]

# Define the variable types of the MTBI and ITBI data files
mtbi_types = {"NEWID": "str", "UCC": "int", "COST": "float", "GIFT": "int"}
itbi_types = {"NEWID": "str", "UCC": "int", "VALUE": "float"}

# Load the UCC dictionary
ucc = pd.read_csv(os.path.join(cex_r_data, "ucc.csv"))

################################################################################
#                                                                              #
# This section of the script determines whether there are differences in the   #
# coverage of the "ucc.csv" file across years.                                 #
#                                                                              #
################################################################################

# Initialize a data frame
df = pd.DataFrame()

# Load the data
year = 1984
for interview in range(1, 5 + 1):
    if interview == 5:
        suffix = str(year + 1)[2:] + "1" + ".txt"
    else:
        suffix = str(year)[2:] + str(interview) + ".txt"
    df_mtbi = pd.read_fwf(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "mtbi" + suffix), colspecs=mtbi_columns[years.index(year)], names=mtbi_names, dtype=mtbi_types)
    df_itbi = pd.read_fwf(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "itbi" + suffix), colspecs=itbi_columns[years.index(year)], names=itbi_names, dtype=itbi_types).rename(columns={"VALUE": "COST"})
    df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)

    # Append the data frames for all interviews
    df = pd.concat([df, df_expenditures], ignore_index=True)

# Recode the NEWID variable
df.loc[:, "NEWID"] = df.NEWID.apply(lambda x: x.zfill(8))

# Merge the expenditures data frame with the UCC dictionary
df = pd.merge(df, ucc, how="left", indicator=True)
prev_ucc_set = set(df.loc[df._merge == "left_only", "UCC"].unique())

# Process the MTBI and ITBI data files
diff_set = [] * len(range(1985, 2022 + 1, 1))
for year in range(1985, 2022 + 1, 1):
    # Initialize a data frame
    df = pd.DataFrame()

    # Load the data
    for interview in range(1, 5 + 1):
        if (year >= 1984) & (year <= 1989):
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".txt"
            else:
                suffix = str(year)[2:] + str(interview) + ".txt"
            df_mtbi = pd.read_fwf(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "mtbi" + suffix), colspecs=mtbi_columns[years.index(year)], names=mtbi_names, dtype=mtbi_types)
            df_itbi = pd.read_fwf(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "itbi" + suffix), colspecs=itbi_columns[years.index(year)], names=itbi_names, dtype=itbi_types).rename(columns={"VALUE": "COST"})
            df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)
        else:
            if interview == 5:
                suffix = str(year + 1)[2:] + "1" + ".csv"
            else:
                suffix = str(year)[2:] + str(interview) + ".csv"
            df_mtbi = pd.read_csv(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "mtbi" + suffix), usecols=mtbi_columns[years.index(year)], dtype=mtbi_types)
            df_itbi = pd.read_csv(os.path.join(cex_r_data, "intrvw" + str(year)[2:], "itbi" + suffix), usecols=itbi_columns[years.index(year)], dtype=itbi_types).rename(columns={"VALUE": "COST"})
            df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)

        # Append the data frames for all interviews
        df = pd.concat([df, df_expenditures], ignore_index=True)

    # Recode the NEWID variable
    df.loc[:, "NEWID"] = df.NEWID.apply(lambda x: x.zfill(8))

    # Merge the expenditures data frame with the UCC dictionary
    df = pd.merge(df, ucc, how="left", indicator=True)
    ucc_set = set(df.loc[df._merge == "left_only", "UCC"].unique())
    diff_set[range(1985, 2022 + 1, 1).index(year)] = ucc_set - prev_ucc_set
    prev_ucc_set = ucc_set