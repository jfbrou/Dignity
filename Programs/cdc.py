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

# Define variable names
names = ["age", "gender", "latin", "race", "year", "deaths", "population"]

# Load the data
df = pd.read_csv(os.path.join(cdc_r_data, "deaths.txt"), delimiter="\t", header=0, usecols=columns).rename(columns=dict(zip(columns, names)))
df_b = pd.read_csv(os.path.join(cdc_r_data, "deaths_b.txt"), delimiter="\t", header=0, usecols=columns_b).rename(columns=dict(zip(columns_b, names)))

# Drop observations where age is missing
df = df.loc[(df.age != "NS") & (df.population != "Not Applicable"), :]
df_b = df_b.loc[(df_b.age != "NS") & (df_b.population != "Not Applicable"), :]

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

# Group and save the data
df = df.astype({"year": "int", "race": "int", "gender": "int", "latin": "int", "age": "int", "deaths": "int", "population": "int"})
df_b = df_b.astype({"year": "int", "race": "int", "gender": "int", "latin": "int", "age": "int", "deaths": "int", "population": "int"})
df = df.groupby(["year", "race", "gender", "latin", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df_b = df_b.groupby(["year", "race", "gender", "latin", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"}).astype({"deaths": "int", "population": "int"})
df.to_csv(os.path.join(cdc_f_data, "cdc.csv"), index=False)
df_b.to_csv(os.path.join(cdc_f_data, "cdc_b.csv"), index=False)