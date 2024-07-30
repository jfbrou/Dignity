# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as smf
import os

# Set the Sherlock data directory
data = '/scratch/users/jfbrou/Dignity'

# Append all bootstrap samples in a single data frame
cew_bootstrap = pd.DataFrame()
for b in range(1, 1000 + 1, 1):
    df = pd.read_csv(os.path.join(data, 'cew_bootstrap_' + str(b) + '.csv'))
    cew_bootstrap = pd.concat([cew_bootstrap, df], ignore_index=True)
    del df
cew_bootstrap.to_csv(os.path.join(data, 'cew_bootstrap.csv'), index=False)