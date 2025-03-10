# Import libraries
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats.mstats import winsorize
import calendar

# Import functions and directories
from functions import *
from directories import *

# Define a list of years
years = range(1984, 2022 + 1)

################################################################################
#                                                                              #
# This section of the script processes the CEX MEMI data files.                #
#                                                                              #
################################################################################

# Define the variable columns of the MEMI data files
columns = [None] * len(years)
for year in years:
    if year == 1984:
        columns[years.index(year)] = [
            (0, 8),     # NEWID
            (8, 11),    # AGE
            (82, 84),   # EDUCA
            (122, 125), # INC_HRSQ
            (136, 138), # INCWEEKQ
            (159, 161), # MEMBNO
            (177, 178), # ORIGINR
            (194, 195), # RACE
            (236, 237), # SEX
            (74, 75)    # CU_CODE
        ]   
    elif year == 1985:
        columns[years.index(year)] = [
            (0, 8),     # NEWID
            (8, 11),    # AGE
            (84, 86),   # EDUCA
            (124, 127), # INC_HRSQ
            (140, 142), # INCWEEKQ
            (165, 167), # MEMBNO
            (185, 186), # ORIGINR
            (202, 203), # RACE
            (244, 245), # SEX
            (74, 75)    # CU_CODE
        ]   
    elif (year >= 1986) & (year <= 1992):
        columns[years.index(year)] = [
            (0, 8),     # NEWID
            (8, 11),    # AGE
            (82, 84),   # EDUCA
            (122, 125), # INC_HRSQ
            (136, 138), # INCWEEKQ
            (161, 163), # MEMBNO
            (179, 180), # ORIGINR
            (196, 197), # RACE
            (238, 239), # SEX
            (74, 75)    # CU_CODE
        ]   
    elif ((year >= 1990) & (year <= 2002)):
        columns[years.index(year)] = [
            'NEWID',
            'AGE',
            'EDUCA',
            'INCWEEKQ',
            'INC_HRSQ',
            'MEMBNO',
            'ORIGINR',
            'RACE',
            'SEX',
            'CU_CODE'
        ]
    else:
        columns[years.index(year)] = [
            'NEWID',
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
            'CU_CODE'
        ]

# Define the variable names of the MEMI data files
names = [None] * len(years)
for year in years:
    if year <= 2002:
        names[years.index(year)] = [
            'NEWID',
            'AGE',
            'EDUCA',
            'INC_HRSQ',
            'INCWEEKQ',
            'MEMBNO',
            'ORIGINR',
            'RACE',
            'SEX',
            'CU_CODE'
        ]
    else:
        names[years.index(year)] = [
            'NEWID',
            'AGE',
            'EDUCA',
            'INC_HRSQ',
            'INCWEEKQ',
            'MEMBNO',
            'HORIGIN',
            'MEMBRACE',
            'RC_WHITE',
            'RC_BLACK',
            'RC_NATAM',
            'RC_ASIAN',
            'RC_PACIL',
            'SEX',
            'CU_CODE'
        ]

# Define the variable types of the MEMI data files
types = [None] * len(years)
for year in years:
    if year <= 2002:
        types[years.index(year)] = {
            'NEWID':    'str',
            'AGE':      'float',
            'EDUCA':    'float',
            'INC_HRSQ': 'str',
            'INCWEEKQ': 'str',
            'MEMBNO':   'int',
            'ORIGINR':  'int',
            'RACE':     'int',
            'SEX':      'int',
            'CU_CODE':  'int'
        }
    elif year == 2015:
        types[years.index(year)] = {
            'NEWID':    'str',
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
            'CU_CODE':  'int'
        }
    else:
        types[years.index(year)] = {
            'NEWID':    'str',
            'AGE':      'float',
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
            'CU_CODE':  'int'
        }

# Define a race encoding
race_map = [None] * len(years)
for year in years:
    if year <= 1987:
        race_map[years.index(year)] = {
            1: 1,      # White
            2: 2,      # Black
            3: 4,      # Asian or Pacific Islander
            4: 3,      # Native American
            5: np.nan  # Other
        } 
    elif (year >= 1988) & (year <= 2002):
        race_map[years.index(year)] = {
            1: 1,      # White
            2: 2,      # Black
            3: 3,      # Native American
            4: 4,      # Asian or Pacific Islander
            5: np.nan  # Other
        } 
    else:
        race_map[years.index(year)] = {
            1: 1,      # White
            2: 2,      # Black
            3: 3,      # Native American
            4: 4,      # Asian
            5: 4,      # Pacific Islander
            6: 5,      # Multi-race
            7: np.nan  # Other
        } 

# Define a latin origin encoding
latin_map = [None] * len(years)
for year in years:
    if year <= 2002:
        latin_map[years.index(year)] = {
            1: 0, # European
            2: 1, # Spanish
            3: 0, # African American
            4: 0  # Other or unknown
        } 
    else:
        latin_map[years.index(year)] = {
            1: 1, # Hispanic
            2: 0  # Not Hispanic
        } 

# Define an education encoding
education_map = [None] * len(years)
for year in years:
    if year <= 2012:
        education_map[years.index(year)] = {
            0:  1, # Never attended
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
            46: 3  # Doctorate degree
        } 
    else:
        education_map[years.index(year)] = {
            1: 1, # Less than 1 year of schooling
            2: 1, # 1st to 8th grade
            3: 1, # 9th to 12th grade (no degree)
            4: 1, # High school graduate
            5: 2, # Some college (no degree)
            6: 2, # Associate degree
            7: 3, # Bachelor's degree
            8: 3  # Master's, professional or doctorate degree
        } 

# Initialize a data frame
cex_memi = pd.DataFrame()

# Process the MEMI data files
for year in years:
    # Initialize a data frame
    df = pd.DataFrame()

    for interview in range(1, 5 + 1):
        if (year >= 1984) & (year <= 1992):
            if interview == 5:
                suffix = 'memi' + str(year + 1)[2:] + '1' + '.txt'
            else:
                suffix = 'memi' + str(year)[2:] + str(interview) + '.txt'
            df_memi = pd.read_fwf(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], suffix), colspecs=columns[years.index(year)], names=names[years.index(year)])
            df_memi = df_memi.astype(types[years.index(year)])
            df_memi.loc[:, 'interview'] = interview
        else:
            if interview == 5:
                suffix = 'memi' + str(year + 1)[2:] + '1' + '.csv'
            else:
                suffix = 'memi' + str(year)[2:] + str(interview) + '.csv'
            if (year == 2003) & (interview == 1):
                df_memi = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], suffix), usecols=columns[years.index(year - 1)], dtype=types[years.index(year - 1)]).rename(columns={'RACE': 'MEMBRACE', 'ORIGINR': 'HORIGIN'})
            else:
                df_memi = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], suffix), usecols=columns[years.index(year)], dtype=types[years.index(year)])
            df_memi.loc[:, 'interview'] = interview
        df = pd.concat([df, df_memi], ignore_index=True)

    # Create a unique identifier for each family and family member since the last digit of NEWID encodes the interview number
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))
    df.loc[:, 'family_id'] = df.NEWID.str[:-1]
    df.loc[:, 'member_id'] = df.NEWID.str[:-1] + df.MEMBNO.astype('str')
    df.loc[:, 'member_id'] = df.member_id.apply(lambda x: x.zfill(9))
    
    # Count the number of interviews in which each family member has participated
    df.loc[:, 'interviews'] = df.loc[:, 'member_id'].map(df.loc[:, 'member_id'].value_counts())
    df = df.loc[df.interviews <= 4, :]

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
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 1
        df = pd.concat([df, second], ignore_index=True)

        # Split the White and Native American observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 1
        df = pd.concat([df, second], ignore_index=True)

        # Split the White and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 4
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 2
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df = pd.concat([df, second], ignore_index=True)

        # Split the Black and Native American observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 2
        df = pd.concat([df, second], ignore_index=True)

        # Split the Black and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 4
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 2
        df = pd.concat([df, second], ignore_index=True)

        # Split the Native American and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 4
        second.loc[:, 'race_weight'] = second.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 2
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 3
        df = pd.concat([df, second], ignore_index=True)

        # Split the White, Black and Native American observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        second.loc[:, 'RACE'] = 2
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), :].copy(deep=True)
        third.loc[:, 'RACE'] = 3
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'race_weight'] = df.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & df.RC_ASIAN.isna() & df.RC_PACIL.isna(), 'RACE'] = 1
        df = pd.concat([df, second, third], ignore_index=True)

        # Split the White, Black and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 2
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        third.loc[:, 'RACE'] = 4
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & df.RC_NATAM.isna() & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df = pd.concat([df, second, third], ignore_index=True)

        # Split the White, Native American and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        third.loc[:, 'RACE'] = 4
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 3
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & df.RC_BLACK.isna() & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df = pd.concat([df, second, third], ignore_index=True)

        # Split the Black, Native American and Asian or Pacific Islander observations in each category
        second = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        second.loc[:, 'RACE'] = 3
        second.loc[:, 'race_weight'] = second.race_weight / 3
        third = df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), :].copy(deep=True)
        third.loc[:, 'RACE'] = 4
        third.loc[:, 'race_weight'] = third.race_weight / 3
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 3
        df.loc[(df.RACE == 5) & df.RC_WHITE.isna() & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 2
        df = pd.concat([df, second, third], ignore_index=True)

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
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'race_weight'] = df.race_weight / 4
        df.loc[(df.RACE == 5) & (df.RC_WHITE == 1) & (df.RC_BLACK == 2) & (df.RC_NATAM == 3) & ((df.RC_ASIAN == 4) | (df.RC_PACIL == 5)), 'RACE'] = 1
        df = pd.concat([df, second, third, fourth], ignore_index=True)

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
    cex_memi = pd.concat([cex_memi, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script processes the CEX FMLI data files.                #
#                                                                              #
################################################################################

# Define the variable columns of the FMLI data files
columns = [None] * len(years)
for year in years:
    if (year >= 1984) & (year <= 1985):
        columns[years.index(year)] = [
            (0, 8),       # NEWID
            (255, 264),   # EARNINCX
            (296, 298),   # FAM_SIZE
            (605, 616),   # FINLWT21
            (671, 679),   # FSALARYX
            (1028, 1029), # RESPSTAT
            (1020, 1021)  # REGION
        ]
    elif (year >= 1986) & (year <= 1992):
        columns[years.index(year)] = [
            (0, 8),     # NEWID
            (273, 282), # EARNINCX
            (334, 336), # FAM_SIZE
            (423, 434), # FINLWT21
            (489, 497), # FSALARYX
            (839, 840), # RESPSTAT
            (831, 832)  # REGION
        ] 
    elif (year >= 2004) & (year <= 2005):
        columns[years.index(year)] = [
            'NEWID',
            'FAM_SIZE',
            'FINLWT21',
            'FSALARYM',
            'FNONFRMM',
            'FFRMINCM',
            'RESPSTAT',
            'REGION'
        ]
    elif (year >= 2006) & (year <= 2012):
        columns[years.index(year)] = [
            'NEWID',
            'FAM_SIZE',
            'FINLWT21',
            'FSALARYX',
            'FNONFRMX',
            'FFRMINCX',
            'RESPSTAT',
            'REGION'
        ]
    elif year >= 2013:
        columns[years.index(year)] = [
            'NEWID',
            'FAM_SIZE',
            'FINLWT21',
            'REGION'
        ]
    else:
        columns[years.index(year)] = [
            'NEWID',
            'EARNINCX',
            'FAM_SIZE',
            'FINLWT21',
            'FSALARYX',
            'RESPSTAT',
            'REGION'
        ]

# Define the variable names of the FMLI data files
names = [
    'NEWID',
    'EARNINCX',
    'FAM_SIZE',
    'FINLWT21',
    'FSALARYX',
    'RESPSTAT',
    'REGION'
]

# Define the variable types of the FMLI data files
types = {
    'NEWID':    'str',
    'EARNINCX': 'float',
    'FAM_SIZE': 'int',
    'FINLWT21': 'float',
    'FSALARYX': 'float',
    'FSALARYM': 'float',
    'RESPSTAT': 'int',
    'FNONFRMX': 'float',
    'FFRMINCX': 'float',
    'FNONFRMM': 'float',
    'FFRMINCM': 'float',
    'REGION':   'float'
}

# Initialize a data frame
cex_fmli = pd.DataFrame()

# Process the FMLI data files
for year in years:
    # Initialize a data frame
    df = pd.DataFrame()

    # Load the data
    for interview in range(1, 5 + 1):
        if (year >= 1984) & (year <= 1992):
            if interview == 5:
                suffix = 'fmli' + str(year + 1)[2:] + '1' + '.txt'
            else:
                suffix = 'fmli' + str(year)[2:] + str(interview) + '.txt'
            df_fmli = pd.read_fwf(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], suffix), colspecs=columns[years.index(year)], names=names, dtype=types)
        else:
            if interview == 5:
                suffix = 'fmli' + str(year + 1)[2:] + '1' + '.csv'
            else:
                suffix = 'fmli' + str(year)[2:] + str(interview) + '.csv'
            df_fmli = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], suffix), usecols=columns[years.index(year)], dtype=types)

        # Append the data frames for all interviews
        df = pd.concat([df, df_fmli], ignore_index=True)

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

    # Create the year variable and append the data frames for all years
    df.loc[:, 'year'] = year
    cex_fmli = pd.concat([cex_fmli, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script combines features of the MEMI and FMLI data       #
# frames and aggregates the MEMI data frame over interviews.                   #
#                                                                              #
################################################################################

# Merge the FMLI data frame with the reference observations of the MEMI data file
cex_fmli_copy = cex_fmli.copy(deep=True)
cex_fmli = pd.merge(cex_memi.loc[cex_memi.CU_CODE == 1, ['year', 'NEWID', 'member_id', 'RACE', 'interviews', 'race_weight']], cex_fmli, how='left', indicator=True)
cex_fmli = cex_fmli.loc[cex_fmli._merge == 'both', :].drop('_merge', axis=1)

# Merge the MEMI data frame with the sampling weights in the FMLI data frame
cex_memi = pd.merge(cex_memi, cex_fmli_copy.loc[:, ['year', 'NEWID', 'FINLWT21']], how='left', indicator=True)
cex_memi = cex_memi.loc[cex_memi._merge == 'both', :].drop('_merge', axis=1)

# Adjust the sampling weights in both data frames
cex_fmli.loc[:, 'FINLWT21'] = cex_fmli.loc[:, 'race_weight'] * cex_fmli.loc[:, 'FINLWT21']
cex_memi.loc[:, 'FINLWT21'] = cex_memi.loc[:, 'race_weight'] * cex_memi.loc[:, 'FINLWT21']
cex_fmli = cex_fmli.drop('race_weight', axis=1)
cex_memi = cex_memi.drop('race_weight', axis=1)

# Aggregate variables over interviews in the MEMI data frame
cex_memi = cex_memi.groupby(['year', 'member_id'], as_index=False).agg({
    'FINLWT21':  'mean',
    'leisure':   'mean',
    'family_id': lambda x: x.iloc[0],
    'SEX':       lambda x: x.iloc[0],
    'RACE':      lambda x: x.iloc[0],
    'HORIGIN':   lambda x: x.iloc[0],
    'EDUCA':     lambda x: x.iloc[0],
    'AGE':       lambda x: x.iloc[0],
    'CU_CODE':   lambda x: x.iloc[0]
}).drop('member_id', axis=1)

################################################################################
#                                                                              #
# This section of the script processes the CEX MTBI and ITBI data files.       #
#                                                                              #
################################################################################

# Define the variable columns of the MTBI and ITBI data files
mtbi_columns = [None] * len(years)
itbi_columns = [None] * len(years)
for year in years:
    if (year >= 1984) & (year <= 1992):
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

# Initialize a data frame
cex_expenditures = pd.DataFrame()

# Process the MTBI and ITBI data files
for year in years:
    # Initialize a data frame
    df = pd.DataFrame()

    # Load the data
    for interview in range(1, 5 + 1):
        if (year >= 1984) & (year <= 1992):
            if interview == 5:
                suffix = str(year + 1)[2:] + '1' + '.txt'
            else:
                suffix = str(year)[2:] + str(interview) + '.txt'
            df_mtbi = pd.read_fwf(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'mtbi' + suffix), colspecs=mtbi_columns[years.index(year)], names=mtbi_names, dtype=mtbi_types)
            df_itbi = pd.read_fwf(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'itbi' + suffix), colspecs=itbi_columns[years.index(year)], names=itbi_names, dtype=itbi_types).rename(columns={'VALUE': 'COST'})
            df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)
        else:
            if interview == 5:
                suffix = str(year + 1)[2:] + '1' + '.csv'
            else:
                suffix = str(year)[2:] + str(interview) + '.csv'
            df_mtbi = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'mtbi' + suffix), usecols=mtbi_columns[years.index(year)], dtype=mtbi_types)
            df_itbi = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'itbi' + suffix), usecols=itbi_columns[years.index(year)], dtype=itbi_types).rename(columns={'VALUE': 'COST'})
            df_expenditures = pd.concat([df_mtbi, df_itbi], ignore_index=True)

        # Append the data frames for all interviews
        df = pd.concat([df, df_expenditures], ignore_index=True)

    # Recode the NEWID variable
    df.loc[:, 'NEWID'] = df.NEWID.apply(lambda x: x.zfill(8))

    # Load the UCC dictionary
    ucc = pd.read_csv(os.path.join(cex_r_data, 'intrvw' + str(year)[2:], 'codebook.csv'))

    # Merge the expenditures data frame with the UCC dictionary
    df = pd.merge(df, ucc, how='left')
    df.loc[:, 'COST'] = df.COST * df.factor

    # Only keep non-gift consumption expenditures
    df = df.loc[(df.GIFT != 1) & (df.consumption == 1), :]

    # Re-scale food at home expenditures before 1987
    if year <= 1987:
        df.loc[df.UCC.isin([790220, 790230, 790240, 190904]), 'COST'] = df.COST / np.exp(-0.10795)

    # Set the negative medical expenditures (reimbursements) to zero
    df.loc[(df.UCC >= 540000) & (df.UCC < 590000) & (df.COST < 0), 'COST'] = 0
    df.loc[(df.UCC == 340906) & (df.COST < 0), 'COST'] = 0

    # Aggregate expenditures, create the year variable and append the data frames for all years
    df_aggregate = df.groupby('NEWID', as_index=False).agg({'COST': 'sum'}).rename(columns={'COST': 'consumption'})
    df_aggregate_nd = df.loc[df.durable == 0, :].groupby('NEWID', as_index=False).agg({'COST': 'sum'}).rename(columns={'COST': 'consumption_nd'})
    df = pd.merge(df_aggregate, df_aggregate_nd, how='outer')
    df.loc[:, 'year'] = year
    cex_expenditures = pd.concat([cex_expenditures, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script merges the CEX MEMI, FMLI, and expenditures data  #
# frames.                                                                      #
#                                                                              #
################################################################################

# Merge the FMLI and expenditures data frames
cex = pd.merge(cex_fmli, cex_expenditures, how='left').fillna(0)

# Aggregate variables over interviews
cex = cex.groupby(['year', 'family_id'], as_index=False).agg({
    'consumption':    'sum',
    'consumption_nd': 'sum',
    'FINLWT21':       'mean',
    'EARNINCX':       'mean',
    'FSALARYX':       'mean',
    'RESPSTAT':       lambda x: x.iloc[0],
    'FAM_SIZE':       lambda x: x.iloc[0],
    'interviews':     lambda x: x.iloc[0],
    'RACE':           lambda x: x.iloc[0],
    'REGION':         lambda x: x.iloc[0]
})

# Drop observations with nonpositve consumption
cex = cex.loc[(cex.consumption > 0) & (cex.consumption_nd > 0), :]

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

# Define a function to calculate weighted percentiles
def weighted_percentile(data, weights, percentile):
    sorted_data, sorted_weights = map(np.array, zip(*sorted(zip(data, weights))))
    cumulative_weight = np.cumsum(sorted_weights)
    cutoff = percentile * cumulative_weight[-1]
    return sorted_data[np.searchsorted(cumulative_weight, cutoff)]

# Winsorize consumption to the 1st and 99th percentiles
for column in [column for column in cex.columns if column.startswith('consumption')]:
    cex.loc[:, column] = cex.groupby('year', as_index=False).apply(lambda x: np.clip(x.loc[:, column], a_min=weighted_percentile(x.loc[:, column], x.FINLWT21, 0.01), a_max=weighted_percentile(x.loc[:, column], x.FINLWT21, 0.99))).values

# Define a function to calculate the weighted standard deviation of log nondurable consumption for all families by the number of interviews in which they participated
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

# Merge the member data frame with the family and expenditures one
cex = pd.merge(cex_memi, cex, how='inner').drop('family_id', axis=1)

# Calculate total NIPA PCE, nondurable PCE and the poverty threshold in each year, which corresponds to 2000 USD in 2012
bea_20405 = pd.read_csv(os.path.join(bea_r_data, 'table_20405.csv'), skiprows=[0, 1, 2, 4], skipfooter=5, header=0, engine='python').rename(columns={'Unnamed: 1': 'series'}).iloc[:, 1:]
bea_20405['series'] = bea_20405['series'].str.strip()
bea_20405 = bea_20405.melt(id_vars='series', var_name='year', value_name='value').dropna()
bea_20405 = bea_20405[bea_20405['value'] != '---']
bea_20405['value'] = pd.to_numeric(bea_20405['value'])
pce = bea_20405.loc[bea_20405['series'] == 'Personal consumption expenditures', 'value'].values - bea_20405.loc[bea_20405['series'] == 'Insurance', 'value'].values
pce_nd = pce - bea_20405.loc[bea_20405['series'] == 'Durable goods', 'value'].values
bea_20100 = pd.read_csv(os.path.join(bea_r_data, 'table_20100.csv'), skiprows=[0, 1, 2, 4], header=0).rename(columns={'Unnamed: 1': 'series'}).iloc[:, 1:]
bea_20100['series'] = bea_20100['series'].str.strip()
bea_20100 = bea_20100.melt(id_vars='series', var_name='year', value_name='value').dropna()
bea_20100['year'] = pd.to_numeric(bea_20100['year'])
population = 1e3 * bea_20100.loc[(bea_20100['series'] == 'Population (midperiod, thousands)6') & bea_20100['year'].isin(years), 'value'].values
bea_10104 = pd.read_csv(os.path.join(bea_r_data, 'table_10104.csv'), skiprows=[0, 1, 2, 4], header=0).rename(columns={'Unnamed: 1': 'series'}).iloc[:, 1:]
bea_10104['series'] = bea_10104['series'].str.strip()
bea_10104 = bea_10104.melt(id_vars='series', var_name='year', value_name='value').dropna()
bea_10104 = bea_10104[bea_10104['value'] != '---']
bea_10104['value'] = pd.to_numeric(bea_10104['value'])
bea_10104['year'] = pd.to_numeric(bea_10104['year'])
deflator = 1e2 / bea_10104.loc[(bea_10104['series'] == 'Personal consumption expenditures') & bea_10104['year'].isin(years), 'value'].values
deflator = deflator / deflator[years.index(2012)]
pce = 1e6 * deflator * pce / population
pce_nd = 1e6 * deflator * pce_nd / population
poverty = 2000 * deflator

# Store the above two series in the data frame
cex = pd.merge(cex, pd.DataFrame(data={
    'year':    years,
    'pce':     pce,
    'pce_nd':  pce_nd,
    'poverty': poverty
}), how='left')

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
        cex.loc[:, column] = cex.pce + cex.pce * (cex.loc[:, column] - cex.loc[:, column + '_average']) / cex.loc[:, column + '_average']
    else:
        cex.loc[:, column] = cex.pce_nd + cex.pce_nd * (cex.loc[:, column] - cex.loc[:, column + '_average']) / cex.loc[:, column + '_average']
    cex = cex.drop(column + '_average', axis=1)

# Enforce the consumption floor on total consumption
for column in [column for column in cex.columns if column.startswith('consumption') and column.find('_nd') == -1]:
    cex.loc[cex.loc[:, column] <= cex.poverty, column] = cex.poverty
cex = cex.drop([column for column in cex.columns if column.startswith('pce')] + ['poverty'], axis=1)

# Recode the CU_CODE variable
cex.loc[cex.CU_CODE != 1, 'CU_CODE'] = 0

# Recode the REGION variable
cex.loc[:, 'REGION'] = cex.REGION.map({1: 1, 2: 1, 3: 2, 4: 1, 0: np.nan})

# Rename variables
cex = cex.rename(columns={
    'SEX':      'gender',
    'RACE':     'race',
    'HORIGIN':  'latin',
    'EDUCA':    'education',
    'AGE':      'age',
    'FAM_SIZE': 'family_size',
    'EARNINCX': 'earnings',
    'FSALARYX': 'salary',
    'RESPSTAT': 'complete',
    'CU_CODE':  'respondent',
    'REGION':   'region',
    'FINLWT21': 'weight'
})

# Redefine the types of all variables
cex = cex.astype({
    'year':                'int',
    'gender':              'int',
    'race':                'int',
    'latin':               'float',
    'education':           'float',
    'age':                 'int',
    'family_size':         'int',
    'consumption':         'float',
    'consumption_sqrt':    'float',
    'consumption_nd':      'float',
    'consumption_nd_sqrt': 'float',
    'earnings':            'float',
    'salary':              'float',
    'leisure':             'float',
    'complete':            'float',
    'respondent':          'int',
    'region':              'float',
    'weight':              'float'
})

# Sort and save the data
cex = cex.sort_values(by='year')
cex.to_csv(os.path.join(cex_f_data, 'cex.csv'), index=False)