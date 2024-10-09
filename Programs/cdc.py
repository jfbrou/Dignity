# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

# Import functions and directories
from functions import *
from directories import *

# Define variable columns
columns = ["Single-Year Ages Code", "Gender Code", "Hispanic Origin", "Single Race 6", "Year", "Deaths", "Population"]
columns_b = ["Single-Year Ages Code", "Gender Code", "Hispanic Origin", "Race", "Year", "Deaths", "Population"]
columns_region = ["Single-Year Ages Code", "Census Region", "Single Race 6", "Year", "Deaths", "Population"]
columns_b_region = ["Single-Year Ages Code", "Census Region", "Race", "Year", "Deaths", "Population"]

# Define variable names
names = ["age", "gender", "latin", "race", "year", "deaths", "population"]
names_region = ["age", "region", "race", "year", "deaths", "population"]

# Load the data
df = pd.read_csv(os.path.join(cdc_r_data, "deaths.txt"), delimiter="\t", header=0, usecols=columns).rename(columns=dict(zip(columns, names)))
df_b = pd.read_csv(os.path.join(cdc_r_data, "deaths_b.txt"), delimiter="\t", header=0, usecols=columns_b).rename(columns=dict(zip(columns_b, names)))
df_region = pd.read_csv(os.path.join(cdc_r_data, "deaths_region.txt"), delimiter="\t", header=0, usecols=columns_region).rename(columns=dict(zip(columns_region, names_region)))
df_b_region = pd.read_csv(os.path.join(cdc_r_data, "deaths_b_region.txt"), delimiter="\t", header=0, usecols=columns_b_region).rename(columns=dict(zip(columns_b_region, names_region)))

# Drop observations where age is missing
df = df.loc[(df.age != "NS") & (df.population != "Not Applicable"), :]
df_b = df_b.loc[(df_b.age != "NS") & (df_b.population != "Not Applicable"), :]
df_region = df_region.loc[(df_region.age != "NS") & (df_region.population != "Not Applicable"), :]
df_b_region = df_b_region.loc[(df_b_region.age != "NS") & (df_b_region.population != "Not Applicable"), :]

# Recode the gender variable
df.loc[:, "gender"] = df.gender.map({"M": 1, "F": 2})
df_b.loc[:, "gender"] = df_b.gender.map({"M": 1, "F": 2})

# Recode the latin origin variable
df.loc[:, "latin"] = df.latin.map({"Hispanic or Latino": 1, "Not Hispanic or Latino": 0, "Not Stated": 0})
df_b.loc[:, "latin"] = df_b.latin.map({"Hispanic or Latino": 1, "Not Hispanic or Latino": 0, "Not Stated": 0})

# Recode the race variable
df.loc[(df.race == "Asian") | (df.race == "Native Hawaiian or Other Pacific Islander"), "race"] = "Asian or Pacific Islander"
df.loc[:, "race"] = df.race.map({"White": 1, "Black or African American": 2, "American Indian or Alaska Native": 3, "Asian or Pacific Islander": 4, "More than one race": 5})
df_b.loc[:, "race"] = df_b.race.map({"White": 1, "Black or African American": 2, "American Indian or Alaska Native": 3, "Asian or Pacific Islander": 4, "More than one race": 5})
df_region.loc[:, "race"] = df_region.race.map({"White": 1, "Black or African American": 2})
df_b_region.loc[:, "race"] = df_b_region.race.map({"White": 1, "Black or African American": 2})

# Recode the region variable
df_region.loc[:, "region"] = df_region.region.map({"Census Region 1: Northeast": 1, "Census Region 2: Midwest": 2, "Census Region 3: South": 3, "Census Region 4: West": 4})
df_b_region.loc[:, "region"] = df_b_region.region.map({"Census Region 1: Northeast": 1, "Census Region 2: Midwest": 2, "Census Region 3: South": 3, "Census Region 4: West": 4})

# Adjust the types of the variables
df = df.astype({"year": "int", "race": "int", "gender": "int", "latin": "int", "age": "int", "deaths": "int", "population": "int"})
df_b = df_b.astype({"year": "int", "race": "int", "gender": "int", "latin": "int", "age": "int", "deaths": "int", "population": "int"})
df_region = df_region.astype({"year": "int", "race": "int", "region": "int", "age": "int", "deaths": "int", "population": "int"})
df_b_region = df_b_region.astype({"year": "int", "race": "int", "region": "int", "age": "int", "deaths": "int", "population": "int"})

# Group and save the data
df = df.groupby(["year", "race", "gender", "latin", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df_b = df_b.groupby(["year", "race", "gender", "latin", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df_region = df_region.groupby(["year", "race", "region", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df_b_region = df_b_region.groupby(["year", "race", "region", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df.to_csv(os.path.join(cdc_f_data, "cdc.csv"), index=False)
df_b.to_csv(os.path.join(cdc_f_data, "cdc_b.csv"), index=False)
df_region.to_csv(os.path.join(cdc_f_data, "cdc_region.csv"), index=False)
df_b_region.to_csv(os.path.join(cdc_f_data, "cdc_b_region.csv"), index=False)