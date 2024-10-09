# Import libraries
import os

# Identify the storage directory
path = "/Users/jfbrou/Library/CloudStorage/Dropbox/GitHub/Dignity"

# Define the BEA API key
bea_api_key = "59DDFDFD-09EB-4529-B4DB-08880867FAEB"

# Define the CDC API keys
cdc_api_key = "e6wdrl8voxmffwf07k3g4qfb2"
cdc_api_key_secret = "d532p6ylwb6c6k5i5g9ojnjulxhi4kk024a7wl0a9cdtl8n9x"

# Identify the data directory
r_data = os.path.join(path, "Data/Raw")
f_data = os.path.join(path, "Data/Final")

# Identify the directory containing the data files from the American Community Surveys (ACS) and U.S. censuses
acs_r_data = os.path.join(r_data, "ACS")
acs_f_data = os.path.join(f_data, "ACS")

# Identify the directory containing the data files from the Center for Disease Control and Prevention (CDC)
cdc_r_data = os.path.join(r_data, "CDC")
cdc_f_data = os.path.join(f_data, "CDC")

# Identify the directory containing the data files from the Consumer Expenditure Surveys (CEX)
cex_r_data = os.path.join(r_data, "CEX")
cex_f_data = os.path.join(f_data, "CEX")

# Identify the directory containing the data files from the CPS
cps_r_data = os.path.join(r_data, "CPS")
cps_f_data = os.path.join(f_data, "CPS")

# Identify the directory containing the data files from the NPS
nps_r_data = os.path.join(r_data, "NPS")

# Identify the directory containing the data files from the ASJ
asj_r_data = os.path.join(r_data, "ASJ")

# Identify the directory containing the population data files
pop_r_data = os.path.join(r_data, "POP")
pop_f_data = os.path.join(f_data, "POP")

# Identify the directory containing the incarceration rate data files
incarceration_f_data = os.path.join(f_data, "Incarceration")

# Identify the directory containing all figures
figures = os.path.join(path, "Figures")

# Identify the directory containing all tables
tables = os.path.join(path, "Tables")
