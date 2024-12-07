# Import libraries
import os
from dotenv import load_dotenv

# Load my environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.getcwd()), '.env'))

# Identify the storage directory
path = os.getenv('path')

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

# Identify the directory containing the population data files
nhis_r_data = os.path.join(r_data, "NHIS")
nhis_f_data = os.path.join(f_data, "NHIS")

# Identify the directory containing all figures
figures = os.path.join(path, "Figures")

# Identify the directory containing all tables
tables = os.path.join(path, "Tables")
