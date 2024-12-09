# Import libraries
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as smf

# Load my environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), '.env'))

# Identify the storage directory
scratch = os.getenv('scratch')

# Append all bootstrap samples in a single data frame
cew_bootstrap = pd.DataFrame()
for b in range(1, 1000 + 1, 1):
    df = pd.read_csv(os.path.join(scratch, 'cew_bootstrap_' + str(b) + '.csv'))
    cew_bootstrap = pd.concat([cew_bootstrap, df], ignore_index=True)
    del df
cew_bootstrap.to_csv(os.path.join(scratch, 'cew_bootstrap.csv'), index=False)