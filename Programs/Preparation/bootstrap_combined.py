# Import libraries
import os
import sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as smf

# Import functions
sys.path.append(os.path.dirname(os.getcwd()))
from functions import *
from directories import *

# Append all bootstrap samples in a single data frame
cew_bootstrap = pd.DataFrame()
for b in range(1, 1000 + 1, 1):
    df = pd.read_csv(os.path.join(scratch, 'cew_bootstrap_' + str(b) + '.csv'))
    cew_bootstrap = pd.concat([cew_bootstrap, df], ignore_index=True)
    del df
cew_bootstrap.to_csv(os.path.join(scratch, 'cew_bootstrap.csv'), index=False)