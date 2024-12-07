# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Define variable columns
columns_1999_2020 = ["Single-Year Ages Code", "Census Region", "Race", "Year", "Deaths", "Population", "Hispanic Origin"]
columns_2018_2022 = ["Single-Year Ages Code", "Census Region", "Single Race 6", "Year", "Deaths", "Population", "Hispanic Origin"]

# Define variable names
names = ["age", "region", "race", "year", "deaths", "population"]

# Load the data
df_1999_2020 = pd.read_csv(os.path.join(cdc_r_data, "Multiple Cause of Death, 1999-2020.txt"), delimiter="\t", header=0, usecols=columns_1999_2020).rename(columns=dict(zip(columns_1999_2020, names)))
df_2018_2022 = pd.read_csv(os.path.join(cdc_r_data, "Multiple Cause of Death, 2018-2022, Single Race.txt"), delimiter="\t", header=0, usecols=columns_2018_2022).rename(columns=dict(zip(columns_2018_2022, names)))

# Only keep non missing values
df_1999_2020 = df_1999_2020.dropna()
df_2018_2022 = df_2018_2022.dropna()

# Drop observations where age is missing
df_1999_2020 = df_1999_2020.loc[(df_1999_2020.age != "NS") & (df_1999_2020.population != "Not Applicable"), :]
df_2018_2022 = df_2018_2022.loc[(df_2018_2022.age != "NS") & (df_2018_2022.population != "Not Applicable"), :]

# Recode the race variable
df_1999_2020.loc[:, "race"] = df_1999_2020.race.map({"White": 1, "Black or African American": 2})
df_2018_2022.loc[:, "race"] = df_2018_2022.race.map({"White": 1, "Black or African American": 2})

# Recode the region variable
df_1999_2020.loc[:, "region"] = df_1999_2020.region.map({"Census Region 1: Northeast": 1, "Census Region 2: Midwest": 1, "Census Region 3: South": 2, "Census Region 4: West": 1})
df_2018_2022.loc[:, "region"] = df_2018_2022.region.map({"Census Region 1: Northeast": 1, "Census Region 2: Midwest": 1, "Census Region 3: South": 2, "Census Region 4: West": 1})

# Adjust the types of the variables
df_1999_2020 = df_1999_2020.astype({"year": "int", "race": "int", "region": "int", "age": "int", "deaths": "int", "population": "int"})
df_2018_2022 = df_2018_2022.astype({"year": "int", "race": "int", "region": "int", "age": "int", "deaths": "int", "population": "int"})

# Group and save the data
df_1999_2020 = df_1999_2020.groupby(["year", "race", "region", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df_2018_2022 = df_2018_2022.groupby(["year", "race", "region", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df_1999_2020.to_csv(os.path.join(cdc_f_data, "cdc_1999_2020.csv"), index=False)
df_2018_2022.to_csv(os.path.join(cdc_f_data, "cdc_2018_2022.csv"), index=False)