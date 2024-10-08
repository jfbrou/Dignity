# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Load and process the NPS data
nps = pd.read_csv(os.path.join(nps_r_data, 'nps.tsv'), delimiter='\t', usecols=['YEAR', 'STATE', 'WHITEM', 'WHITEF', 'BLACKM', 'BLACKF'])
nps = nps.loc[(nps.STATE == 'US') & nps.YEAR.isin(range(1984, 2022 + 1, 1)), :].drop('STATE', axis=1)
nps.loc[:, 'WHITE'] = nps.WHITEM + nps.WHITEF
nps.loc[:, 'BLACK'] = nps.BLACKM + nps.BLACKF
nps_white = nps.loc[:, ['YEAR', 'WHITE']].rename(columns={'WHITE': 'total'})
nps_white.loc[:, 'race'] = 1
nps_black = nps.loc[:, ['YEAR', 'BLACK']].rename(columns={'BLACK': 'total'})
nps_black.loc[:, 'race'] = 2
nps = pd.concat([nps_white, nps_black], ignore_index=True)

# Save the data
nps.to_csv(os.path.join(nps_f_data, 'nps.csv'), index=False)