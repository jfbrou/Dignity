# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Map states to regions
state_to_region = {
    'AL': 'South',
    'AK': 'West',
    'AZ': 'West',
    'AR': 'South',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'South',
    'FL': 'South',
    'GA': 'South',
    'HI': 'West',
    'ID': 'West',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'South',
    'LA': 'South',
    'ME': 'Northeast',
    'MD': 'South',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'South',
    'MO': 'Midwest',
    'MT': 'West',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Northeast',
    'NM': 'West',
    'NY': 'Northeast',
    'NC': 'South',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'South',
    'OR': 'West',
    'PA': 'Northeast',
    'RI': 'Northeast',
    'SC': 'South',
    'SD': 'Midwest',
    'TN': 'South',
    'TX': 'South',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'South',
    'WA': 'West',
    'WV': 'South',
    'WI': 'Midwest',
    'WY': 'West'
}
fips_to_state = {
    1: 'AL',
    2: 'AK',
    4: 'AZ',
    5: 'AR',
    6: 'CA',
    8: 'CO',
    9: 'CT',
    10: 'DE',
    11: 'DC',
    12: 'FL',
    13: 'GA',
    15: 'HI',
    16: 'ID',
    17: 'IL',
    18: 'IN',
    19: 'IA',
    20: 'KS',
    21: 'KY',
    22: 'LA',
    23: 'ME',
    24: 'MD',
    25: 'MA',
    26: 'MI',
    27: 'MN',
    28: 'MS',
    29: 'MO',
    30: 'MT',
    31: 'NE',
    32: 'NV',
    33: 'NH',
    34: 'NJ',
    35: 'NM',
    36: 'NY',
    37: 'NC',
    38: 'ND',
    39: 'OH',
    40: 'OK',
    41: 'OR',
    42: 'PA',
    44: 'RI',
    45: 'SC',
    46: 'SD',
    47: 'TN',
    48: 'TX',
    49: 'UT',
    50: 'VT',
    51: 'VA',
    53: 'WA',
    54: 'WV',
    55: 'WI',
    56: 'WY',
    60: 'AS', 
    66: 'GU',  
    69: 'MP',  
    72: 'PR',  
    78: 'VI'   
}
state_to_abbreviation = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
state_to_abbreviation = {k.lower(): v for k, v in state_to_abbreviation.items()}
id_to_state = {
    1: 'AL',
    3: 'AZ',
    4: 'AR',
    5: 'CA',
    6: 'CO',
    7: 'CT',
    8: 'DE',
    9: 'DC',
    10: 'FL',
    11: 'GA',
    13: 'ID',
    14: 'IL',
    15: 'IN',
    16: 'IA',
    17: 'KS',
    18: 'KY',
    19: 'LA',
    20: 'ME',
    21: 'MD',
    22: 'MA',
    23: 'MI',
    24: 'MN',
    25: 'MS',
    26: 'MO',
    27: 'MT',
    28: 'NE',
    29: 'NV',
    30: 'NH',
    31: 'NJ',
    32: 'NM',
    33: 'NY',
    34: 'NC',
    35: 'ND',
    36: 'OH',
    37: 'OK',
    38: 'OR',
    39: 'PA',
    41: 'SC',
    42: 'SD',
    43: 'TN',
    44: 'TX',
    45: 'UT',
    47: 'VA',
    48: 'WA',
    49: 'WV',
    50: 'WI',
    51: 'WY'
}

# Load and process the NPS data
nps = pd.read_csv(os.path.join(nps_r_data, 'nps.tsv'), delimiter='\t', usecols=['YEAR', 'STATE', 'WHITEM', 'WHITEF', 'BLACKM', 'BLACKF'])
nps.loc[:, 'REGION'] = nps.loc[:, 'STATE'].map(state_to_region)
nps = nps.dropna(subset=['REGION'])
nps.loc[:, 'REGION'] = nps.loc[:, 'REGION'].map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1})
nps = nps.loc[nps.YEAR.isin(range(1984, 2022 + 1, 1)), :].drop('STATE', axis=1)
nps.loc[:, 'WHITE'] = nps.WHITEM + nps.WHITEF
nps.loc[:, 'BLACK'] = nps.BLACKM + nps.BLACKF
nps_white = nps.loc[:, ['YEAR', 'WHITE', 'REGION']].rename(columns={'WHITE': 'incarcerated_nps', 'REGION': 'region'})
nps_white.loc[:, 'race'] = 1
nps_black = nps.loc[:, ['YEAR', 'BLACK', 'REGION']].rename(columns={'BLACK': 'incarcerated_nps', 'REGION': 'region'})
nps_black.loc[:, 'race'] = 2
nps = pd.concat([nps_white, nps_black], ignore_index=True).rename(columns={'YEAR': 'year'})
nps = nps.groupby(['year', 'region', 'race'], as_index=False).agg({'incarcerated_nps': 'sum'})

# Load and process the ASJ data
asj = expand({'year': range(1984, 2022 + 1, 1), 'race': {1, 2}, 'region': {1, 2}})
for year in list(range(1985, 1987 + 1, 1)) + list(range(1989, 1992 + 1, 1)) + list(range(1994, 1998 + 1, 1)) + list(range(2000, 2003 + 1, 1)) + [2004, 2006] + list(range(2008, 2019, 1)) + list(range(2020, 2022 + 1, 1)):
    if (year >= 2020) | ((year >= 2013) & (year <= 2018)) | ((year >= 2008) & (year <= 2009)):
        df = pd.read_csv(os.path.join(asj_r_data, str(year) + '.tsv'), sep='\t')
        if year >= 2015:
            region_1 = df.STATEFIPS.map(fips_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
            region_2 = df.STATEFIPS.map(fips_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        elif (year >= 2013) & (year <= 2014):
            region_1 = df.STATE.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
            region_2 = df.STATE.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        else:
            region_1 = df.STATE.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
            region_2 = df.STATE.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'WHITE'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'BLACK'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'WHITE'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'BLACK'].sum()
    elif (year >= 2010) & (year <= 2012):
        df = pd.read_csv(os.path.join(asj_r_data, str(year) + '.tsv'), sep='\t')
        region_1 = df.state.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.state.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'white'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'black'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'white'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'black'].sum()
    elif ((year >= 2003) & (year <= 2004)) | (year == 2006):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        if year == 2004:
            region_1 = df.STATE.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
            region_2 = df.STATE.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        else:
            region_1 = df.STATE.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
            region_2 = df.STATE.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'WHITE'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'BLACK'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'WHITE'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'BLACK'].sum()
    elif (year >= 2000) & (year <= 2002):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        if year == 2002:
            region_1 = df.STATE.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
            region_2 = df.STATE.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        else:
            region_1 = pd.to_numeric(df.STATE, errors='coerce').map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
            region_2 = pd.to_numeric(df.STATE, errors='coerce').map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V51'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V53'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V51'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V53'].sum()
    elif year == 1998:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        region_1 = df.V4.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V4.map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V43'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V45'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V43'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V45'].sum()
    elif (year >= 1996) & (year <= 1997):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        region_1 = df.V5.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V5.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_1, 'V42'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_1, 'V44'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_2, 'V42'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_2, 'V44'], errors='coerce').sum()
    elif year == 1995:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        region_1 = df.V5.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V5.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_1, 'V36'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_1, 'V38'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_2, 'V36'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_2, 'V38'], errors='coerce').sum()
    elif year == 1994:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'))
        region_1 = df.V5.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V5.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_1, 'V26'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_1, 'V28'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_2, 'V26'], errors='coerce').sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = pd.to_numeric(df.loc[region_2, 'V28'], errors='coerce').sum()
    elif (year >= 1991) & (year <= 1992):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V65', 'V70', 'V6'])
        region_1 = df.V6.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V6.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V65'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V70'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V65'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V70'].sum()
    elif (year >= 1989) & (year <= 1990):
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V63', 'V68', 'V6'])
        region_1 = df.V6.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V6.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V63'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V68'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V63'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V68'].sum()
    elif year == 1987:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V53', 'V58', 'V6'])
        region_1 = df.V6.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V6.str.lower().map(state_to_abbreviation).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V53'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V58'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V53'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V58'].sum()
    elif year == 1986:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V50', 'V55', 'V6'])
        region_1 = df.V6.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 1
        region_2 = df.V6.map(id_to_state).map(state_to_region).map({'South': 2, 'West': 1, 'Midwest': 1, 'Northeast': 1}) == 2
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V50'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V55'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V50'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V55'].sum()
    elif year == 1985:
        df = pd.read_stata(os.path.join(asj_r_data, str(year) + '.dta'), columns=['V44', 'V49', 'V3'])
        region_1 = (df.V3 != 3)
        region_2 = (df.V3 == 3)
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V44'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 1), 'incarcerated_asj'] = df.loc[region_1, 'V49'].sum()
        asj.loc[(asj.year == year) & (asj.race == 1) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V44'].sum()
        asj.loc[(asj.year == year) & (asj.race == 2) & (asj.region == 2), 'incarcerated_asj'] = df.loc[region_2, 'V49'].sum()

# Merge the NPS and ASJ data
df = pd.merge(nps, asj)
df = df.loc[df.year.isin(range(1984, 2022 + 1, 1)), :]

# Interpolate the missing values in the ASJ by race and region
df = df.groupby(['region', 'race'], as_index=False).apply(lambda x: x.interpolate()).reset_index(drop=True)
for race in [1, 2]:
    for region in [1, 2]:
        df.loc[(df.year == 1984) & (df.race == race) & (df.region == region), 'incarcerated_asj'] = df.loc[(df.year == 1985) & (df.race == race) & (df.region == region), 'incarcerated_asj'].values * df.loc[(df.year == 1984) & (df.race == race) & (df.region == region), 'incarcerated_nps'].values / df.loc[(df.year == 1985) & (df.race == race) & (df.region == region), 'incarcerated_nps'].values

# Calculate the total number of incarcerated individuals
df.loc[:, 'incarcerated'] = df.loc[:, 'incarcerated_nps'] + df.loc[:, 'incarcerated_asj']
df = df.drop(['incarcerated_nps', 'incarcerated_asj'], axis=1)

# Load the adult population data
pop = pd.read_csv(os.path.join(pop_f_data, 'population.csv'))

# Merge the data and calculate the incarceration rate
df = pd.merge(df, pop)
df = pd.merge(expand({'year': range(1984, 2022 + 1, 1), 'race': [-1, 1, 2], 'region': [-1, 1, 2]}), df, how='left')
df.loc[df.race == -1, 'incarcerated'] = df.loc[df.race != -1, :].groupby('year').apply(lambda x: x.incarcerated.sum()).values
df.loc[df.race == -1, 'population'] = df.loc[df.race != -1, :].groupby('year').apply(lambda x: x.population.sum()).values
df.loc[:, 'incarceration_rate'] = df.loc[:, 'incarcerated'] / df.loc[:, 'population']
df = df.drop(['incarcerated', 'population'], axis=1)
df = pd.merge(expand({'year': range(1984, 2022 + 1, 1), 'race': [-1, 1, 2], 'age': range(101)}), df).reset_index(drop=True)
df.loc[(df.age < 18) | (df.age >= 85), 'incarceration_rate'] = 0

# Save the data
df.to_csv(os.path.join(incarceration_f_data, 'incarceration.csv'), index=False)