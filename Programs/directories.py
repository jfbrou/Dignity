# Import libraries
import os

# Identify the storage directory
path = "/Users/jfbrou/Library/CloudStorage/Dropbox/Dignity/Github"

# Define the BEA API key
bea_api_key = "59DDFDFD-09EB-4529-B4DB-08880867FAEB"

# Identify the data directory
r_data = os.path.join(path, "Data/Raw")
f_data = os.path.join(path, "Data/Final")

# Identify the directory containing the data files from the American Community Surveys (ACS) and U.S. censuses
acs_r_data = os.path.join(r_data, "ACS")
acs_f_data = os.path.join(f_data, "ACS")

# Identify the directory containing the data files from the Bureau of Justice Statistics' (BJS) National Corrections Reporting Program (NCR)
ncr_r_data = os.path.join(r_data, "NCR")
ncr_f_data = os.path.join(f_data, "NCR")

# Identify the directory containing the data files from the Bureau of Justice Statistics' (BJS) National Prisoner Statistics (NPS)
nps_r_data = os.path.join(r_data, "NPS")
nps_f_data = os.path.join(f_data, "NPS")

# Identify the directory containing the data files from the Center for Disease Control and Prevention (CDC)
cdc_r_data = os.path.join(r_data, "CDC")
cdc_f_data = os.path.join(f_data, "CDC")

# Identify the directory containing the data files from the National Health Interview Survey (NHIS)
nhis_r_data = os.path.join(r_data, "NHIS")
nhis_f_data = os.path.join(f_data, "NHIS")

# Identify the directory containing the data files from the Consumer Expenditure Surveys (CEX)
cex_r_data = os.path.join(r_data, "CEX")
cex_f_data = os.path.join(f_data, "CEX")

# Identify the directory containing the data files from the CPS
cps_r_data = os.path.join(r_data, "CPS")
cps_f_data = os.path.join(f_data, "CPS")

# Identify the directory containing the population estimates data files
population_r_data = os.path.join(r_data, "Population")
population_f_data = os.path.join(f_data, "Population")

# Identify the directory containing all figures
figures = os.path.join(path, "Figures")

# Identify the directory containing all tables
tables = os.path.join(path, "Tables")
