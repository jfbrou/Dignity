# Import libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
import os

# Import functions and directories
from functions import *
from directories import *

# Define the Gompertz function
def gompertz(x, data=None):
    # Estimate the model
    model = LinearRegression().fit(np.array(range(65, 84 + 1)).reshape(-1, 1), np.log(x.iloc[:20].to_numpy()))

    # Return the mortality rates for ages 85 to 99
    return np.append(x.iloc[:20], np.exp(model.intercept_ + model.coef_ * range(85, 100)))

# Create a cdc data frame
cdc_df = pd.DataFrame()
cdc_b_df = pd.DataFrame()

################################################################################
#                                                                              #
# This section of the script computes survival rates for all individuals.      #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))

# Aggregate deaths and population by year and age
df = df.groupby(["year", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [-1], "latin": [-1], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby("year", as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby("year", as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by gender.                #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))

# Aggregate mortality by year, gender, and age
df = df.groupby(["year", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [-1], "latin": [-1], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race.                  #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))

# Aggregate mortality by year, race and age
df = df.groupby(["year", "race", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [1, 2], "latin": [-1], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race and gender.       #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))

# Aggregate mortality by year, race, gender and age
df = df.groupby(["year", "race", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [1, 2], "latin": [-1], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for Latinos.              #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))
df = df.loc[df.latin == 1, :]

# Aggregate mortality by year and age
df = df.groupby(["year", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [-1], "latin": [1], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby("year", as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby("year", as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by gender for Latinos.    #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))
df = df.loc[df.latin == 1, :]

# Aggregate mortality by year, gender and age
df = df.groupby(["year", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [-1], "latin": [1], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for non-Latinos by race.  #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))
df = df.loc[df.latin == 0, :]

# Aggregate mortality by year, race and age
df = df.groupby(["year", "race", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [1, 2], "latin": [0], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for non-Latinos by race   #
# and gender.                                                                  #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc.csv"))
df = df.loc[df.latin == 0, :]

# Aggregate mortality by year, race, gender and age
df = df.groupby(["year", "race", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [1, 2], "latin": [0], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race and region.       #
#                                                                              #
################################################################################

# Load the cdc data from 2018 to 2022
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_region.csv"))

# Aggregate mortality by year, race, region, and age
df = df.groupby(["year", "race", "region", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(2018, 2022 + 1, 1), "age": range(101), "race": [1, 2], "latin": [-1], "gender": [-1], "region": [1, 2, 3, 4]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race", "region"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race", "region"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_df = pd.concat([cdc_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for all individuals.      #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))

# Aggregate deaths and population by year and age
df = df.groupby(["year", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [-1], "latin": [-1], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby("year", as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby("year", as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by gender.                #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))

# Aggregate mortality by year, gender, and age
df = df.groupby(["year", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [-1], "latin": [-1], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race.                  #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))

# Aggregate mortality by year, race and age
df = df.groupby(["year", "race", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [1, 2], "latin": [-1], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race and gender.       #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))

# Aggregate mortality by year, race, gender and age
df = df.groupby(["year", "race", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [1, 2], "latin": [-1], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for Latinos.              #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))
df = df.loc[df.latin == 1, :]

# Aggregate mortality by year and age
df = df.groupby(["year", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [-1], "latin": [1], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby("year", as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby("year", as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by gender for Latinos.    #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))
df = df.loc[df.latin == 1, :]

# Aggregate mortality by year, gender and age
df = df.groupby(["year", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [-1], "latin": [1], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for non-Latinos by race.  #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))
df = df.loc[df.latin == 0, :]

# Aggregate mortality by year, race and age
df = df.groupby(["year", "race", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [1, 2], "latin": [0], "gender": [-1], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates for non-Latinos by race   #
# and gender.                                                                  #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b.csv"))
df = df.loc[df.latin == 0, :]

# Aggregate mortality by year, race, gender and age
df = df.groupby(["year", "race", "gender", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [1, 2], "latin": [0], "gender": [1, 2], "region": [-1]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race", "gender"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race", "gender"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script computes survival rates by race and region.       #
#                                                                              #
################################################################################

# Load the cdc data from 1999 to 2020
df = pd.read_csv(os.path.join(cdc_f_data, "cdc_b_region.csv"))

# Aggregate mortality by year, race, region, and age
df = df.groupby(["year", "race", "region", "age"], as_index=False).agg({"deaths": "sum", "population": "sum"})

# Create a data frame with all levels of all variables
df = pd.merge(expand({"year": range(1999, 2020 + 1, 1), "age": range(101), "race": [1, 2], "latin": [-1], "gender": [-1], "region": [1, 2, 3, 4]}), df, how="left")

# Compute mortality rates
df.loc[:, "M"] = df.deaths / (df.population + df.deaths / 2)

# Calculate the survival rates
df.loc[df.age.isin(range(65, 100)), "S"] = 1 - df.loc[df.age.isin(range(65, 100)), :].groupby(["year", "race", "region"], as_index=False).M.transform(gompertz).values
df.loc[~df.age.isin(range(65, 100)), "S"] = 1 - df.loc[~df.age.isin(range(65, 100)), "M"]
df.loc[:, "S"] = df.groupby(["year", "race", "region"], as_index=False).S.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values

# Append the CDC data frame
df = df.drop(["deaths", "population", "M"], axis=1)
cdc_b_df = pd.concat([cdc_b_df, df], ignore_index=True)

################################################################################
#                                                                              #
# This section of the script processes the life tables.                        #
#                                                                              #
################################################################################

# Load the life tables data
lt = pd.read_csv(os.path.join(cdc_r_data, "lifetables.csv"))
lt85 = expand({"year": list(range(1984, 1989 + 1, 1)) + list(range(1991, 1996 + 1, 1)), "age": range(86, 100 + 1, 1), "race": [-1, 1, 2], "latin": [-1], "gender": [-1, 1, 2]})
lt = pd.concat([lt, lt85], ignore_index=True).sort_values(by=["year", "race", "latin", "gender", "age"])

# Compute the mortality rates
lt.loc[:, "M"] = lt.groupby(["year", "race", "latin", "gender"], as_index=False).S.transform(lambda x: 1 - x.shift(-1) / x).values

# Extrapolate the survival rates
lt.loc[lt.age.isin(range(65, 100)), "gompertz"] = 1 - lt.loc[lt.age.isin(range(65, 100)), :].groupby(["year", "race", "latin", "gender"], as_index=False).M.transform(gompertz).values
lt.loc[~lt.age.isin(range(65, 100)), "gompertz"] = 1 - lt.loc[~lt.age.isin(range(65, 100)), "M"]
lt.loc[:, "gompertz"] = lt.groupby(["year", "race", "latin", "gender"], as_index=False).gompertz.transform(lambda x: np.append(1, x.iloc[:-1].cumprod())).values
lt.loc[lt.S.isna(), "S"] = lt.gompertz
lt = lt.drop(["M", "gompertz"], axis=1)

# Adjust the 1950 and 1960 survival rates for Black Americans
adjustment = lt.loc[(lt.year == 1970) & (lt.race == 2), "S"].values / lt.loc[(lt.year == 1970) & (lt.race == 5), "S"].values
lt = lt.loc[(lt.year != 1970) | ((lt.year == 1970) & (lt.race != 5)), :]
lt.loc[(lt.year == 1950) & (lt.race == 5), "S"] = lt.loc[(lt.year == 1950) & (lt.race == 5), "S"].values * adjustment
lt.loc[(lt.year == 1960) & (lt.race == 5), "S"] = lt.loc[(lt.year == 1960) & (lt.race == 5), "S"].values * adjustment
lt.loc[lt.race == 5, "race"] = 2

# Append the life tables with the above data frames
lt.loc[:, "dataset"] = "lt"
lt = pd.concat([lt, cdc_df], ignore_index=True)
lt.loc[lt.dataset.isna(), "dataset"] = "cdc"
lt = pd.concat([lt, cdc_b_df], ignore_index=True)
lt.loc[lt.dataset.isna(), "dataset"] = "cdc_b"

# Adjust the post-2020 survival rates
for race in [1, 2]:
    for latin in [-1, 0]:
        for gender in [-1, 1, 2]:
            adjustment_2018 = lt.loc[(lt.year == 2018) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values / lt.loc[(lt.year == 2018) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"].values
            adjustment_2019 = lt.loc[(lt.year == 2019) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values / lt.loc[(lt.year == 2019) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"].values
            adjustment_2020 = lt.loc[(lt.year == 2020) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values / lt.loc[(lt.year == 2020) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"].values
            adjustment = (adjustment_2018 + adjustment_2019 + adjustment_2020) / 3
            lt.loc[(lt.year == 2021) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] = lt.loc[(lt.year == 2021) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] * adjustment
            lt.loc[(lt.year == 2022) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] = lt.loc[(lt.year == 2022) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] * adjustment
for gender in [-1, 1, 2]:
    adjustment_2018 = lt.loc[(lt.year == 2018) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values / lt.loc[(lt.year == 2018) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"].values
    adjustment_2019 = lt.loc[(lt.year == 2019) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values / lt.loc[(lt.year == 2019) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"].values
    adjustment_2020 = lt.loc[(lt.year == 2020) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values / lt.loc[(lt.year == 2020) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"].values
    adjustment = (adjustment_2018 + adjustment_2019 + adjustment_2020) / 3
    lt.loc[(lt.year == 2021) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] = lt.loc[(lt.year == 2021) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] * adjustment
    lt.loc[(lt.year == 2022) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] = lt.loc[(lt.year == 2022) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc"), "S"] * adjustment

# Adjust the 2018-2020 survival rates
for race in [1, 2]:
    for latin in [-1, 0]:
        for gender in [-1, 1, 2]:
            adjustment_2015 = lt.loc[(lt.year == 2015) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "lt"), "S"].values / lt.loc[(lt.year == 2015) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values
            adjustment_2016 = lt.loc[(lt.year == 2016) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "lt"), "S"].values / lt.loc[(lt.year == 2016) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values
            adjustment_2017 = lt.loc[(lt.year == 2017) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "lt"), "S"].values / lt.loc[(lt.year == 2017) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values
            adjustment = (adjustment_2015 + adjustment_2016 + adjustment_2017) / 3
            lt.loc[(lt.year == 2018) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] = lt.loc[(lt.year == 2018) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] * adjustment
            lt.loc[(lt.year == 2019) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] = lt.loc[(lt.year == 2019) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] * adjustment
            lt.loc[(lt.year == 2020) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] = lt.loc[(lt.year == 2020) & (lt.race == race) & (lt.latin == latin) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] * adjustment
for gender in [-1, 1, 2]:
    adjustment_2015 = lt.loc[(lt.year == 2015) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "lt"), "S"].values / lt.loc[(lt.year == 2015) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values
    adjustment_2016 = lt.loc[(lt.year == 2016) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "lt"), "S"].values / lt.loc[(lt.year == 2016) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values
    adjustment_2017 = lt.loc[(lt.year == 2017) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "lt"), "S"].values / lt.loc[(lt.year == 2017) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"].values
    adjustment = (adjustment_2015 + adjustment_2016 + adjustment_2017) / 3
    lt.loc[(lt.year == 2018) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] = lt.loc[(lt.year == 2018) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] * adjustment
    lt.loc[(lt.year == 2019) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] = lt.loc[(lt.year == 2019) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] * adjustment
    lt.loc[(lt.year == 2020) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] = lt.loc[(lt.year == 2020) & (lt.race == -1) & (lt.latin == 1) & (lt.gender == gender) & (lt.dataset == "cdc_b"), "S"] * adjustment

# Drop the unused tables
drop_sample_1 = lt.year.isin(range(2018, 2020 + 1, 1)) & (lt.dataset == "cdc")
lt = lt.loc[~drop_sample_1, :]
drop_sample_2 = lt.year.isin(range(1999, 2017 + 1, 1)) & (lt.dataset == "cdc_b")
lt = lt.loc[~drop_sample_2, :]
lt = lt.drop("dataset", axis=1)

# Save the data
lt.to_csv(os.path.join(cdc_f_data, "survival.csv"), index=False)