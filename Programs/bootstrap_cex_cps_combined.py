# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions
from functions import *

# Set the Sherlock data directory
data = '/scratch/users/jfbrou/Dignity'

# Append all bootstrap samples in a single data frame
dignity_cex_bootstrap = pd.DataFrame()
dignity_cps_bootstrap = pd.DataFrame()
for b in range(1, 1000 + 1, 1):
    df_cex = pd.read_csv(os.path.join(data, 'dignity_cex_bootstrap_' + str(b) + '.csv'))
    df_cps = pd.read_csv(os.path.join(data, 'dignity_cps_bootstrap_' + str(b) + '.csv'))
    dignity_cex_bootstrap = pd.concat([dignity_cex_bootstrap, df_cex], ignore_index=True)
    dignity_cps_bootstrap = pd.concat([dignity_cps_bootstrap, df_cps], ignore_index=True)
    del df_cex, df_cps
dignity_bootstrap = pd.merge(dignity_cex_bootstrap, dignity_cps_bootstrap, how='left')
dignity_bootstrap.to_csv(os.path.join(data, 'dignity_bootstrap.csv'), index=False)