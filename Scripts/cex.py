# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import beapy
import statsmodels.formula.api as smf
from dotenv import load_dotenv
load_dotenv()
import os
import calendar

# Import functions and directories
from functions import *
from directories import *

# Start the BEA client
bea = beapy.BEA(key=os.getenv('bea_api_key'))

################################################################################
#                                                                              #
# This section of the script processes the CEX MEMI data files.                #
#                                                                              #
################################################################################

# Define a list of years
years = range(1984, 2020 + 1)

# Define the variable columns of the MEMI data files
columns = [None] * len(years)
for year in years:
    if year == 1984:
        columns[years.index(year)] = [(0, 8),     # NEWID
                                      (8, 11),    # AGE
                                      (82, 84),   # EDUCA
                                      (122, 125), # INC_HRSQ
                                      (136, 138), # INCWEEKQ
                                      (159, 161), # MEMBNO
                                      (177, 178), # ORIGINR
                                      (194, 195), # RACE
                                      (236, 237), # SEX
                                      (74, 75)]   # CU_CODE
    elif year == 1985:
        columns[years.index(year)] = [(0, 8),     # NEWID
                                      (8, 11),    # AGE
                                      (84, 86),   # EDUCA
                                      (124, 127), # INC_HRSQ
                                      (140, 142), # INCWEEKQ
                                      (165, 167), # MEMBNO
                                      (185, 186), # ORIGINR
                                      (202, 203), # RACE
                                      (244, 245), # SEX
                                      (74, 75)]   # CU_CODE
    elif (year >= 1986) & (year <= 1994):
        columns[years.index(year)] = [(0, 8),     # NEWID
                                      (8, 11),    # AGE
                                      (82, 84),   # EDUCA
                                      (122, 125), # INC_HRSQ
                                      (136, 138), # INCWEEKQ
                                      (161, 163), # MEMBNO
                                      (179, 180), # ORIGINR
                                      (196, 197), # RACE
                                      (238, 239), # SEX
                                      (74, 75)]   # CU_CODE
    elif ((year >= 1995) & (year <= 2002)):
        columns[years.index(year)] = ['NEWID',
                                      'AGE',
                                      'EDUCA',
                                      'INCWEEKQ',
                                      'INC_HRSQ',
                                      'MEMBNO',
                                      'ORIGINR',
                                      'RACE',
                                      'SEX',
                                      'CU_CODE']
    else:
        columns[years.index(year)] = ['NEWID',
                                      'AGE',
                                      'EDUCA',
                                      'INCWEEKQ',
                                      'INC_HRSQ',
                                      'MEMBNO',
                                      'HORIGIN',
                                      'MEMBRACE',
                                      'RC_WHITE',
                                      'RC_BLACK',
                                      'RC_NATAM',
                                      'RC_ASIAN',
                                      'RC_PACIL',
                                      'SEX',
                                      'CU_CODE']

# Define the variable names of the MEMI data files
names = ['NEWID',
         'AGE',
         'EDUCA',
         'INC_HRSQ',
         'INCWEEKQ',
         'MEMBNO',
         'ORIGINR',
         'RACE',
         'SEX',
         'CU_CODE']

# Define the variable types of the MEMI data files
types = [None] * len(years)
for year in years:
    if year <= 2002:
        types[years.index(year)] = {'NEWID':    'str',
                                    'AGE':      'int',
                                    'EDUCA':    'float',
                                    'INC_HRSQ': 'str',
                                    'INCWEEKQ': 'str',
                                    'MEMBNO':   'int',
                                    'ORIGINR':  'int',
                                    'RACE':     'int',
                                    'SEX':      'int',
                                    'CU_CODE':  'int'}
    elif year == 2015:
        types[years.index(year)] = {'NEWID':    'str',
                                    'AGE':      'str',
                                    'EDUCA':    'float',
                                    'INC_HRSQ': 'str',
                                    'INCWEEKQ': 'str',
                                    'MEMBNO':   'int',
                                    'HORIGIN':  'int',
                                    'MEMBRACE': 'int',
                                    'RC_WHITE': 'float',
                                    'RC_BLACK': 'float',
                                    'RC_NATAM': 'float',
                                    'RC_ASIAN': 'float',
                                    'RC_PACIL': 'float',
                                    'SEX':      'int',
                                    'CU_CODE':  'int'}
    else:
        types[years.index(year)] = {'NEWID':    'str',
                                    'AGE':      'int',
                                    'EDUCA':    'float',
                                    'INC_HRSQ': 'str',
                                    'INCWEEKQ': 'str',
                                    'MEMBNO':   'int',
                                    'HORIGIN':  'int',
                                    'MEMBRACE': 'int',
                                    'RC_WHITE': 'float',
                                    'RC_BLACK': 'float',
                                    'RC_NATAM': 'float',
                                    'RC_ASIAN': 'float',
                                    'RC_PACIL': 'float',
                                    'SEX':      'int',
                                    'CU_CODE':  'int'}

# Define a race encoding
race_map = [None] * len(years)
for year in years:
    if year <= 1987:
        race_map[years.index(year)] = {1: 1,      # White
                                       2: 2,      # Black
                                       3: 4,      # Asian or Pacific Islander
                                       4: 3,      # Native American
                                       5: np.nan} # Other
    elif (year >= 1988) & (year <= 2002):
        race_map[years.index(year)] = {1: 1,      # White
                                       2: 2,      # Black
                                       3: 3,      # Native American
                                       4: 4,      # Asian or Pacific Islander
                                       5: np.nan} # Other
    else:
        race_map[years.index(year)] = {1: 1,      # White
                                       2: 2,      # Black
                                       3: 3,      # Native American
                                       4: 4,      # Asian
                                       5: 4,      # Pacific Islander
                                       6: 5,      # Multi-race
                                       7: np.nan} # Other

# Define a latin origin encoding
latin_map = [None] * len(years)
for year in years:
    if year <= 2002:
        latin_map[years.index(year)] = {1: 0, # European
                                        2: 1, # Spanish
                                        3: 0, # African American
                                        4: 0} # Other or unknown
    else:
        latin_map[years.index(year)] = {1: 1, # Hispanic
                                        2: 0} # Not Hispanic

# Define an education encoding
education_map = [None] * len(years)
for year in years:
    if year <= 2012:
        education_map[years.index(year)] = {0:  1, # Never attended
                                            1:  1, # 1st grade
                                            2:  1, # 2nd grade
                                            3:  1, # 3rd grade
                                            4:  1, # 4th grade
                                            5:  1, # 5th grade
                                            6:  1, # 6th grade
                                            7:  1, # 7th grade
                                            8:  1, # 8th grade
                                            9:  1, # 9th grade
                                            10: 1, # 10th grade
                                            11: 1, # 11th grade
                                            12: 1, # 12th grade
                                            38: 1, # 12th grade (no diploma)
                                            39: 1, # 12th grade (high school graduate)
                                            13: 2, # 1 year of college
                                            21: 2, # 1 year of college
                                            40: 2, # Some college (no degree)
                                            14: 2, # 2 years of college
                                            22: 2, # 2 years of college
                                            41: 2, # Associate degree (vocational)
                                            42: 2, # Associate degree (academic)
                                            15: 2, # 3 years of college
                                            23: 2, # 3 years of college
                                            16: 3, # 4 years of college
                                            24: 3, # 4 years of college
                                            43: 3, # Bachelor's degree
                                            17: 3, # 1 year of graduate school
                                            25: 3, # 1 year of graduate school
                                            31: 3, # 1 year of graduate school
                                            18: 3, # 2+ years of graduate school
                                            26: 3, # 2+ years of graduate school
                                            32: 3, # 2+ years of graduate school
                                            44: 3, # Master's degree
                                            45: 3, # Professional degree
                                            46: 3} # Doctorate degree
    else:
        education_map[years.index(year)] = {1: 1, # Less than 1 year of schooling
                                            2: 1, # 1st to 8th grade
                                            3: 1, # 9th to 12th grade (no degree)
                                            4: 1, # High school graduate
                                            5: 2, # Some college (no degree)
                                            6: 2, # Associate degree
                                            7: 3, # Bachelor's degree
                                            8: 3} # Master's, professional or doctorate degree

# Initialize a data frame
cex_memi = pd.DataFrame()

# Process the MEMI data files
for year in years:
    # Initialize a data frame
    df = pd.DataFrame()

    # Load the data
    for interview in range(1, 5 + 1):
        if (year >= 1984) & (year <= 1994):
            df_memi = pd.read_fwf(os.path.join(cex_r_data, 'CEX' + str(year), 'memi' + str(interview) + '.txt'), colspecs=columns[years.index(year)], names=names, dtype=types[years.index(year)])
            df_memi.loc[:, 'interview'] = interview
        elif (year == 2003) & (interview == 1):
            df_memi = pd.read_csv(os.path.join(cex_r_data, 'CEX' + str(year), 'memi' + str(interview) + '.csv'), usecols=columns[years.index(year - 1)], dtype=types[years.index(year - 1)]).rename(columns={'RACE': 'MEMBRACE', 'ORIGINR': 'HORIGIN'})
            df_memi.loc[:, 'interview'] = interview
        else:
            df_memi = pd.read_csv(os.path.join(cex_r_data, 'CEX' + str(year), 'memi' + str(interview) + '.csv'), usecols=columns[years.index(year)], dtype=types[years.index(year)])
            df_memi.loc[:, 'interview'] = interview

        # Append the data frames for all interviews
        df = df.append(df_memi, ignore_index=True)

    # Create a unique identifier for each family and family member since the last digit of NEWID encodes the interview number
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))
    df.loc[:, 'family_id'] = df.NEWID.str[:-1]
    df.loc[:, 'member_id'] = df.NEWID.str[:-1] + df.MEMBNO.astype('str')
    df.loc[:, 'member_id'] = df.member_id.apply(lambda x: x.zfill(9))

    # Count the number of interviews in which each family member has participated
    df.loc[:, 'interviews'] = df.loc[:, 'member_id'].map(df.loc[:, 'member_id'].value_counts())

    # Redefine the type of the age variable in 2015
    if year == 2015:
        df.loc[:, 'AGE'] = pd.to_numeric(df.loc[:, 'AGE'], errors='coerce')
        df = df.loc[df.AGE.notna(), :]

    # Rename the race and latin origin variables
    if year <= 2002:
        df = df.rename(columns={'ORIGINR': 'HORIGIN'})
    else:
        df = df.rename(columns={'MEMBRACE': 'RACE'})

    # Recode the race variable
    df.loc[:, 'race_weight'] = 1
    df.loc[:, 'RACE'] = df.RACE.map(race_map[years.index(year)])
    if year >= 2003:
        # Recode the White observations
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & df.RC_NATAM.isna() & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 1

        # Recode the Black observations
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 2

        # Recode the Native American observations
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 3

        # Recode the Asian or Pacific Islander observations
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 4

        # Split the White and Black observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        second.loc[:, 'RACE'] = 2
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 1
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 2
        df = df.append(second, ignore_index=True)

        # Split the White and Native American observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 1
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 2
        df = df.append(second, ignore_index=True)

        # Split the White and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 4
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 2
        df = df.append(second, ignore_index=True)

        # Split the Black and Native American observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 2
        df = df.append(second, ignore_index=True)

        # Split the Black and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 4
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 2
        df = df.append(second, ignore_index=True)

        # Split the Native American and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 4
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 3
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 2
        df = df.append(second, ignore_index=True)

        # Split the White, Black and Native American observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        second.loc[:, 'RACE'] = 2
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        third.loc[:, 'RACE'] = 3
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 1
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 3
        df = df.append(second.append(third, ignore_index=True), ignore_index=True)

        # Split the White, Black and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 2
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        third.loc[:, 'RACE'] = 4
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 3
        df = df.append(second.append(third, ignore_index=True), ignore_index=True)

        # Split the White, Native American and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        third.loc[:, 'RACE'] = 4
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 3
        df = df.append(second.append(third, ignore_index=True), ignore_index=True)

        # Split the Black, Native American and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        third.loc[:, 'RACE'] = 4
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 3
        df = df.append(second.append(third, ignore_index=True), ignore_index=True)

        # Split the White, Black, Native American and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 2
        second.loc[:, 'race_weight'] = second.race_weight / 4
        third = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        third.loc[:, 'RACE'] = 3
        third.loc[:, 'race_weight'] = third.race_weight / 4
        fourth = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        fourth.loc[:, 'RACE'] = 4
        fourth.loc[:, 'race_weight'] = fourth.race_weight / 4
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 4
        df = df.append(second.append(third.append(fourth, ignore_index=True), ignore_index=True), ignore_index=True)

        # Drop the unknown race observations and multi-race variables
        df = df.loc[df.RACE != 5, :]
        df = df.drop(['RC_WHITE', 'RC_BLACK', 'RC_NATAM', 'RC_ASIAN', 'RC_PACIL'], axis=1)

    # Adjust the family member identifier for the multi-race observations
    df.loc[:, 'member_id'] = df.member_id + df.RACE.astype('int').astype('str')

    # Recode the latin origin variable
    if year == 2003:
        df.loc[df.interview == 1, 'HORIGIN'] = df.HORIGIN.map(latin_map[years.index(year - 1)])
        df.loc[df.interview != 1, 'HORIGIN'] = df.HORIGIN.map(latin_map[years.index(year)])
    else:
        df.loc[:, 'HORIGIN'] = df.HORIGIN.map(latin_map[years.index(year)])

    # Recode the education variable
    df.loc[:, 'EDUCA'] = df.EDUCA.map(education_map[years.index(year)])

    # Recode the hours worked per week and weeks worked per year variables
    df.loc[:, 'INC_HRSQ'] = pd.to_numeric(df.INC_HRSQ, errors='coerce').fillna(0)
    df.loc[:, 'INCWEEKQ'] = pd.to_numeric(df.INCWEEKQ, errors='coerce').fillna(0)

    # Create the hours worked per year variable
    df.loc[:, 'hours'] = df.INC_HRSQ * df.INCWEEKQ
    df = df.drop(['INC_HRSQ', 'INCWEEKQ'], axis=1)

    # Split hours worked per year evenly among family members between 25 and 64
    df = pd.merge(df, df.loc[(df.AGE >= 25) & (df.AGE <= 64), :].groupby(['interview', 'family_id'], as_index=False).agg({'hours': 'mean'}).rename(columns={'hours': 'split'}), how='left')
    df.loc[(df.AGE >= 25) & (df.AGE < 65), 'hours'] = df.split
    df = df.drop(['split', 'interview'], axis=1)

    # Create the leisure variable
    if calendar.isleap(year):
        df.loc[:, 'leisure'] = (16 * 366 - df.hours) / (16 * 366)
    else:
        df.loc[:, 'leisure'] = (16 * 365 - df.hours) / (16 * 365)
    df = df.drop('hours', axis=1)

    # Create the year variable and append the data frames for all years
    df.loc[:, 'year'] = year
    cex_memi = cex_memi.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script processes the CEX FMLI data files.                #
#                                                                              #
################################################################################

# Define the variable columns of the FMLI data files
columns = [None] * len(years)
for year in years:
    if (year >= 1984) & (year <= 1985):
        columns[years.index(year)] = [(0, 8),       # NEWID
                                      (255, 264),   # EARNINCX
                                      (296, 298),   # FAM_SIZE
                                      (605, 616),   # FINLWT21
                                      (671, 679),   # FSALARYX
                                      (1028, 1029)] # RESPSTAT
    elif (year >= 1986) & (year <= 1994):
        columns[years.index(year)] = [(0, 8),     # NEWID
                                      (273, 282), # EARNINCX
                                      (334, 336), # FAM_SIZE
                                      (423, 434), # FINLWT21
                                      (489, 497), # FSALARYX
                                      (839, 840)] # RESPSTAT
    elif (year >= 2004) & (year <= 2005):
        columns[years.index(year)] = ['NEWID',
                                      'FAM_SIZE',
                                      'FINLWT21',
                                      'FSALARYM',
                                      'FNONFRMM',
                                      'FFRMINCM',
                                      'RESPSTAT']
    elif (year >= 2006) & (year <= 2012):
        columns[years.index(year)] = ['NEWID',
                                      'FAM_SIZE',
                                      'FINLWT21',
                                      'FSALARYX',
                                      'FNONFRMX',
                                      'FFRMINCX',
                                      'RESPSTAT']
    elif year >= 2013:
        columns[years.index(year)] = ['NEWID',
                                      'FAM_SIZE',
                                      'FINLWT21',
                                      'ROOMSQ']
    else:
        columns[years.index(year)] = ['NEWID',
                                      'EARNINCX',
                                      'FAM_SIZE',
                                      'FINLWT21',
                                      'FSALARYX',
                                      'RESPSTAT']

# Define the variable names of the FMLI data files
names = ['NEWID',
         'EARNINCX',
         'FAM_SIZE',
         'FINLWT21',
         'FSALARYX',
         'RESPSTAT']

# Define the variable types of the FMLI data files
types = {'NEWID':    'str',
         'EARNINCX': 'float',
         'FAM_SIZE': 'int',
         'FINLWT21': 'float',
         'FSALARYX': 'float',
         'FSALARYM': 'float',
         'RESPSTAT': 'int',
         'FNONFRMX': 'float',
         'FFRMINCX': 'float',
         'FNONFRMM': 'float',
         'FFRMINCM': 'float'}

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

    # Create a unique identifier for each family since the last digit of NEWID encodes the interview number
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))
    df.loc[:, 'family_id'] = df.NEWID.str[:-1]

    # Create the earnings variable for the years between 2004 and 2012
    if (year >= 2004) & (year <= 2005):
        df.loc[:, 'EARNINCX'] = df.FSALARYM + df.FNONFRMM + df.FFRMINCM
        df = df.rename(columns={'FSALARYM': 'FSALARYX'}).drop(['FNONFRMM', 'FFRMINCM'], axis=1)
    elif (year >= 2006) & (year <= 2012):
        df.loc[:, 'EARNINCX'] = df.FSALARYX + df.FNONFRMX + df.FFRMINCX
        df = df.drop(['FNONFRMX', 'FFRMINCX'], axis=1)

    # Only keep the number of rooms variable in 2019
    if year != 2019:
        df.loc[:, 'ROOMSQ'] = np.nan
    else:
        df.loc[:, 'ROOMSQ'] = pd.to_numeric(df.loc[:, 'ROOMSQ'], errors='coerce')

    # Create the year variable and append the data frames for all years
    df.loc[:, 'year'] = year
    cex_fmli = cex_fmli.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script combines features of the MEMI and FMLI data       #
# frames and aggregates the MEMI data frame over interviews.                   #
#                                                                              #
################################################################################

# Merge the FMLI data frame with the reference observations of the MEMI data file
cex_fmli = pd.merge(cex_memi.loc[cex_memi.CU_CODE == 1, ['year', 'NEWID', 'member_id', 'RACE', 'interviews', 'race_weight']], cex_fmli, how='left', indicator=True)
cex_fmli = cex_fmli.loc[cex_fmli._merge == 'both', :].drop('_merge', axis=1)

# Merge the MEMI data frame with the sampling weights in the FMLI data frame
cex_memi = pd.merge(cex_memi, cex_fmli.loc[:, ['year', 'NEWID', 'FINLWT21']], how='left', indicator=True)
cex_memi = cex_memi.loc[cex_memi._merge == 'both', :].drop('_merge', axis=1)

# Adjust the sampling weights in both data frames
cex_fmli.loc[:, 'FINLWT21'] = cex_fmli.loc[:, 'race_weight'] * cex_fmli.loc[:, 'FINLWT21']
cex_memi.loc[:, 'FINLWT21'] = cex_memi.loc[:, 'race_weight'] * cex_memi.loc[:, 'FINLWT21']
cex_fmli = cex_fmli.drop('race_weight', axis=1)
cex_memi = cex_memi.drop('race_weight', axis=1)

# Aggregate variables over interviews in the MEMI data frame
cex_memi = cex_memi.groupby(['year', 'member_id'], as_index=False).agg({'FINLWT21':  'mean',
                                                                        'leisure':   'mean',
                                                                        'family_id': lambda x: x.iloc[0],
                                                                        'SEX':       lambda x: x.iloc[0],
                                                                        'RACE':      lambda x: x.iloc[0],
                                                                        'HORIGIN':   lambda x: x.iloc[0],
                                                                        'EDUCA':     lambda x: x.iloc[0],
                                                                        'AGE':       lambda x: x.iloc[0],
                                                                        'CU_CODE':   lambda x: x.iloc[0]}).drop('member_id', axis=1)

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

# Load the NIPA PCE aggregation data file
nipa = pd.read_csv(os.path.join(cex_f_data, 'nipa_pce.csv'))

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
    df = pd.merge(df, ucc, how='left')
    df.loc[:, 'COST'] = df.COST * df.factor

    # Only keep non-gift consumption expenditures
    df = df.loc[(df.GIFT != 1) & (df.consumption == 1), :]

    # Re-scale food at home expenditures before 1987
    if year <= 1987:
        df.loc[df.category == 'FDHOME', 'COST'] = df.COST / np.exp(-0.10795)

    # Calculate the average NIPA PCE scaling ratio
    df_nipa = df.groupby(['NEWID', 'UCC'], as_index=False).agg({'COST': 'sum'})
    df_nipa = pd.merge(df_nipa, cex_fmli.loc[cex_fmli.year == year, ['NEWID', 'FINLWT21']], how='left')
    df_nipa.loc[:, 'COST'] = df_nipa.COST * df_nipa.FINLWT21
    df_nipa = df_nipa.groupby('UCC', as_index=False).agg({'COST': 'sum'})
    df_nipa = pd.merge(df_nipa, nipa.loc[nipa.year == year, ['UCC', 'ratio']], how='inner')
    df_nipa.loc[:, 'COST'] = df_nipa.COST / df_nipa.COST.sum()
    ratio_average = np.sum(df_nipa.ratio * df_nipa.COST)

    # Merge the expenditures data frame with the NIPA PCE aggregation data and re-scale CEX expenditures
    df = pd.merge(df, nipa.loc[nipa.year == year, ['UCC', 'ratio']], how='left')
    if year == 2019:
        df.loc[df.UCC == 600141, 'ratio'] = 1
    df.loc[df.ratio.isna(), 'ratio'] = ratio_average
    df.loc[:, 'COST_nipa'] = df.COST / df.ratio

    # Aggregate expenditures, create the year variable and append the data frames for all years
    df_aggregate = df.groupby('NEWID', as_index=False).agg({'COST': 'sum', 'COST_nipa': 'sum'}).rename(columns={'COST': 'consumption', 'COST_nipa': 'consumption_nipa'})
    df_aggregate_nd = df.loc[df.durable == 0, :].groupby('NEWID', as_index=False).agg({'COST': 'sum', 'COST_nipa': 'sum'}).rename(columns={'COST': 'consumption_nd', 'COST_nipa': 'consumption_nipa_nd'})
    df_aggregate_nh = df.loc[df.health == 0, :].groupby('NEWID', as_index=False).agg({'COST': 'sum'}).rename(columns={'COST': 'consumption_nh'})
    df_aggregate_nh_nd = df.loc[(df.health == 0) & (df.durable == 0), :].groupby('NEWID', as_index=False).agg({'COST': 'sum'}).rename(columns={'COST': 'consumption_nh_nd'})
    df_aggregate_rent = df.loc[df.rent == 1, :].groupby('NEWID', as_index=False).agg({'COST': 'sum'}).rename(columns={'COST': 'consumption_rent'})
    df = pd.merge(df_aggregate, df_aggregate_nd, how='outer')
    df = pd.merge(df, df_aggregate_nh, how='outer')
    df = pd.merge(df, df_aggregate_nh_nd, how='outer')
    df = pd.merge(df, df_aggregate_rent, how='outer')
    df.loc[:, 'year'] = year
    cex_expenditures = cex_expenditures.append(df, ignore_index=True)

################################################################################
#                                                                              #
# This section of the script merges the CEX MEMI, FMLI and expenditures data   #
# frames.                                                                      #
#                                                                              #
################################################################################

# Merge the FMLI and expenditures data frames
cex = pd.merge(cex_fmli, cex_expenditures, how='left').fillna(0)

# Aggregate variables over interviews
cex = cex.groupby(['year', 'member_id'], as_index=False).agg({'consumption':         'sum',
                                                              'consumption_nipa':    'sum',
                                                              'consumption_nd':      'sum',
                                                              'consumption_nipa_nd': 'sum',
                                                              'consumption_nh':      'sum',
                                                              'consumption_nh_nd':   'sum',
                                                              'consumption_rent':    'sum',
                                                              'FINLWT21':            'mean',
                                                              'EARNINCX':            'mean',
                                                              'FSALARYX':            'mean',
                                                              'ROOMSQ':              lambda x: x.iloc[0],
                                                              'RESPSTAT':            lambda x: x.iloc[0],
                                                              'family_id':           lambda x: x.iloc[0],
                                                              'FAM_SIZE':            lambda x: x.iloc[0],
                                                              'interviews':          lambda x: x.iloc[0],
                                                              'RACE':                lambda x: x.iloc[0]}).drop('member_id', axis=1)

# Drop observations with nonpositve consumption
cex = cex.loc[(cex.consumption > 0) & \
              (cex.consumption_nd > 0) & \
              (cex.consumption_nipa > 0) & \
              (cex.consumption_nipa_nd > 0) & \
              (cex.consumption_nh > 0) & \
              (cex.consumption_nh_nd > 0), :]

# Calculate the rent share of consumption
cex.loc[:, 'rent_share'] = cex.consumption_rent / cex.consumption
cex.loc[cex.rent_share < 0, 'rent_share'] = 0
cex.loc[cex.rent_share > 1, 'rent_share'] = 1
cex = cex.drop('consumption_rent', axis=1)

# Divide the consumption measures by the number of family members or its square root
for column in [column for column in cex.columns if column.startswith('consumption')]:
    cex.loc[:, column + '_sqrt'] = cex.loc[:, column] / np.sqrt(cex.FAM_SIZE)
    cex.loc[:, column] = cex.loc[:, column] / cex.FAM_SIZE

# Divide earnings by the number of family members
for column in ['EARNINCX', 'FSALARYX']:
    cex.loc[:, column] = cex.loc[:, column] / cex.FAM_SIZE

# Recode the complete income respondent variable
cex.loc[:, 'RESPSTAT'] = cex.RESPSTAT.map({1: 1, 2: 0}).fillna(0)

# Re-scale consumption by the number of interviews and take its logarithm for the nondurable component of consumption
for column in [column for column in cex.columns if column.startswith('consumption') and column.find('_nd') == -1]:
    cex.loc[:, column] = 4 * cex.loc[:, column] / cex.interviews
for column in [column for column in cex.columns if column.startswith('consumption') and column.find('_nd') != -1]:
    cex.loc[:, column] = np.log(4 * cex.loc[:, column] / cex.interviews)

# Define a function to calculate the weighted standard deviation of log nondurable consumption for all families by the number of interviews in which they participated in
def f(x):
    d = {}
    columns = [column for column in x.columns if column.startswith('consumption') and column.find('_nd') != -1]
    for column in columns:
        d[column.replace('consumption', 'scale')] = np.sqrt(np.average((x.loc[:, column] - np.average(x.loc[:, column], weights=x.FINLWT21))**2, weights=x.FINLWT21))
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate the weighted standard deviation of log nondurable consumption for all families by the number of interviews in which they participated in
d_rename = dict(zip([column.replace('consumption', 'scale') for column in cex.columns if column.startswith('consumption') and column.find('_nd') != -1],
                    [column.replace('consumption', 'scale_4') for column in cex.columns if column.startswith('consumption') and column.find('_nd') != -1]))
cex = pd.merge(cex, cex.groupby(['year', 'interviews'], as_index=False).apply(f), how='left')
cex = pd.merge(cex, cex.loc[cex.interviews == 4, :].groupby('year', as_index=False).apply(f).rename(columns=d_rename), how='left')

# Calculate the ratio of those standard deviations relative to the four-interviews households
for column in [column for column in cex.columns if column.startswith('scale') and column.find('_4') == -1]:
    cex.loc[:, column] = cex.loc[:, column.replace('scale', 'scale_4')] / cex.loc[:, column]
cex = cex.drop(['interviews'] + [column for column in cex.columns if column.startswith('scale_4')], axis=1)

# Define a function to calculate the average log nondurable consumption by year and race
def f(x):
    d = {}
    columns = [column for column in x.columns if column.startswith('consumption') and column.find('_nd') != -1]
    for column in columns:
        d[column + '_average'] = np.average(x.loc[:, column], weights=x.FINLWT21)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate average log nondurable consumption by year and race
cex = pd.merge(cex, cex.groupby(['year', 'RACE'], as_index=False).apply(f), how='left')

# Re-scale log nondurable consumption to adjust for differences in standard deviations across the number of interviews
for column in [column for column in cex.columns if column.startswith('consumption') and not column.endswith('average') and column.find('_nd') != -1]:
    cex.loc[:, column] = np.exp(cex.loc[:, column + '_average'] + cex.loc[:, column.replace('consumption', 'scale')] * (cex.loc[:, column] - cex.loc[:, column + '_average']))
cex = cex.drop([column for column in cex.columns if column.endswith('average') or column.startswith('scale')], axis=1)

# Only keep the relevant variables
mean_columns = [column for column in cex.columns if column.startswith('consumption')] + ['rent_share', 'EARNINCX', 'FSALARYX']
mean_functions = ['mean'] * len(mean_columns)
first_columns = ['ROOMSQ', 'FAM_SIZE', 'RESPSTAT']
first_functions = [lambda x: x.iloc[0]] * len(first_columns)
d_functions = dict(zip(mean_columns, mean_functions))
d_functions.update(dict(zip(first_columns, first_functions)))
cex = cex.groupby(['year', 'family_id'], as_index=False).agg(d_functions)

# Merge the member data frame with the family and expenditures one
cex = pd.merge(cex_memi, cex, how='inner').drop('family_id', axis=1)

# Calculate total NIPA PCE, nondurable PCE and the poverty threshold in each year, which corresponds to 2000 USD in 2012
pce = bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DPCERC.values.squeeze() - bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DINSRC.values.squeeze()
pce_nd = pce - bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DDURRC.values.squeeze()
pce_nh = pce - bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DTAERC.values.squeeze() - bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DHLCRC.values.squeeze()
pce_nh_nd = pce_nd - bea.data('nipa', tablename='t20405', frequency='a', year=years).data.DHLCRC.values.squeeze()
population = 1e3 * bea.data('nipa', tablename='t20100', frequency='a', year=years).data.B230RC.values.squeeze()
deflator = 1e2 / bea.data('nipa', tablename='t10104', frequency='a', year=years).data.DPCERG.values.squeeze()
pce = 1e6 * deflator * pce / population
pce_nd = 1e6 * deflator * pce_nd / population
pce_nh = 1e6 * deflator * pce_nh / population
pce_nh_nd = 1e6 * deflator * pce_nh_nd / population
poverty = 2000 * deflator

# Store the above two series in the data frame
cex = pd.merge(cex, pd.DataFrame(data={'year':      years,
                                       'pce':       pce,
                                       'pce_nd':    pce_nd,
                                       'pce_nh':    pce_nh,
                                       'pce_nh_nd': pce_nh_nd,
                                       'poverty':   poverty}), how='left')

# Define a function to calculate the average consumption by year
def f(x):
    d = {}
    columns = [column for column in x.columns if column.startswith('consumption')]
    for column in columns:
        d[column + '_average'] = np.average(x.loc[:, column], weights=x.FINLWT21)
    return pd.Series(d, index=[key for key, value in d.items()])

# Re-scale consumption such that it aggregates to the NIPA personal consumption expenditures
cex = pd.merge(cex, cex.groupby('year', as_index=False).apply(f), how='left')
for column in [column for column in cex.columns if column.startswith('consumption') and not column.endswith('average')]:
    if column.find('_nd') == -1:
        if column.find('_nh') == -1:
            cex.loc[:, column] = cex.pce + cex.pce * (cex.loc[:, column] - cex.loc[:, column + '_average']) / cex.loc[:, column + '_average']
        else:
            cex.loc[:, column] = cex.pce_nh + cex.pce_nh * (cex.loc[:, column] - cex.loc[:, column + '_average']) / cex.loc[:, column + '_average']
    else:
        if column.find('_nh') == -1:
            cex.loc[:, column] = cex.pce_nd + cex.pce_nd * (cex.loc[:, column] - cex.loc[:, column + '_average']) / cex.loc[:, column + '_average']
        else:
            cex.loc[:, column] = cex.pce_nh_nd + cex.pce_nh_nd * (cex.loc[:, column] - cex.loc[:, column + '_average']) / cex.loc[:, column + '_average']
    cex = cex.drop(column + '_average', axis=1)
cex = cex.loc[(cex.consumption > 0) & \
              (cex.consumption_nd > 0) & \
              (cex.consumption_nipa > 0) & \
              (cex.consumption_nipa_nd > 0) & \
              (cex.consumption_nh > 0) & \
              (cex.consumption_nh_nd > 0), :]

# Enforce the consumption floor on total consumption
for column in [column for column in cex.columns if column.startswith('consumption') and column.find('_nd') == -1]:
    cex.loc[cex.loc[:, column] <= cex.poverty, column] = cex.poverty
cex = cex.drop([column for column in cex.columns if column.startswith('pce')] + ['poverty'], axis=1)

# Set the values of the rooms variable to missing values for years other than 2019
cex.loc[cex.year != 2019, 'ROOMSQ'] = np.nan

# Recode the CU_CODE variable
cex.loc[cex.CU_CODE != 1, 'CU_CODE'] = 0

# Rename variables
cex = cex.rename(columns={'SEX':      'gender',
                          'RACE':     'race',
                          'HORIGIN':  'latin',
                          'EDUCA':    'education',
                          'AGE':      'age',
                          'FAM_SIZE': 'family_size',
                          'EARNINCX': 'earnings',
                          'FSALARYX': 'salary',
                          'RESPSTAT': 'complete',
                          'CU_CODE':  'respondent',
                          'ROOMSQ':   'rooms',
                          'FINLWT21': 'weight'})

# Redefine the types of all variables
cex = cex.astype({'year':                     'int',
                  'gender':                   'int',
                  'race':                     'int',
                  'latin':                    'float',
                  'education':                'float',
                  'age':                      'int',
                  'family_size':              'int',
                  'consumption':              'float',
                  'consumption_sqrt':         'float',
                  'consumption_nd':           'float',
                  'consumption_nd_sqrt':      'float',
                  'consumption_nipa':         'float',
                  'consumption_nipa_sqrt':    'float',
                  'consumption_nipa_nd':      'float',
                  'consumption_nipa_nd_sqrt': 'float',
                  'consumption_nh':           'float',
                  'consumption_nh_sqrt':      'float',
                  'consumption_nh_nd':        'float',
                  'consumption_nh_nd_sqrt':   'float',
                  'rent_share':               'float',
                  'rooms':                    'float',
                  'earnings':                 'float',
                  'salary':                   'float',
                  'leisure':                  'float',
                  'complete':                 'float',
                  'respondent':               'int',
                  'weight':                   'float'})

# Sort and save the data
cex = cex.sort_values(by='year')
cex.to_csv(os.path.join(cex_f_data, 'cex.csv'), index=False)

# Keep the imputation sample
df = cex.loc[cex.complete == 1, :]

# Create race binary variables
df = pd.concat([df, pd.get_dummies(df.race.astype('int'), prefix='race')], axis=1)

# Create education binary variables
df.loc[df.education.isna() | (df.age < 30), 'education'] = 4
df = pd.concat([df, pd.get_dummies(df.education.astype('int'), prefix='education')], axis=1)

# Recode the gender variable
df.loc[:, 'gender'] = df.gender.replace({1: 1, 2: 0})

# Define a function to calculate the average of consumption, income and demographics by year
def f(x):
    d = {}
    columns = ['consumption', 'earnings', 'salary'] + ['race_' + str(i) for i in range(1, 4 + 1)] \
                                                    + ['education_' + str(i) for i in range(1, 4 + 1)] \
                                                    + ['family_size', 'latin', 'gender', 'age']
    for column in columns:
        d[column + '_average'] = np.average(x.loc[:, column], weights=x.weight)
    return pd.Series(d, index=[key for key, value in d.items()])

# Calculate the average of consumption, income and demographics by year
df = pd.merge(df, df.groupby('year', as_index=False).apply(f), how='left')

# Calculate the percentage deviation of consumption, income and demographics from their annual average
columns = ['consumption', 'earnings', 'salary'] + ['race_' + str(i) for i in range(1, 4 + 1)] \
                                                + ['education_' + str(i) for i in range(1, 4 + 1)] \
                                                + ['family_size', 'latin', 'gender', 'age']
for column in columns:
    df.loc[:, column + '_deviation'] = df.loc[:, column] / df.loc[:, column + '_average'] - 1

# Fit and save the OLS models for consumption
earnings_formula = 'consumption_deviation ~ ' + ' + '.join([column for column in df.columns if column.endswith('deviation') and not column.startswith('consumption') and not column.startswith('salary')])
salary_formula = 'consumption_deviation ~ ' + ' + '.join([column for column in df.columns if column.endswith('deviation') and not column.startswith('consumption') and not column.startswith('earnings')])
earnings_model = smf.wls(formula=earnings_formula, data=df, weights=df.weight.to_numpy()).fit()
salary_model = smf.wls(formula=salary_formula, data=df, weights=df.weight.to_numpy()).fit()
earnings_model.save(os.path.join(cex_f_data, 'earnings.pickle'))
salary_model.save(os.path.join(cex_f_data, 'salary.pickle'))
