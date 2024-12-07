---
contributors:
  - Jean-Félix Brouillette
  - Charles I. Jones
  - Peter J. Klenow
---

# Replication package for: Race and Economic Well-Being in the United States

## Overview

This replication package contains two Python programs. The program `data.py`, prepares the data for analysis and the program `analysis.py` produces the figures and tables in the paper. The replicator should expect the programs to run for about 2 hours.

## Data Availability Statement

### Statement about Rights

- I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript. 
- I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package.

### Summary of Availability

- All data are publicly available.

### Details on each Data Source

#### U.S. Census Bureau's Current Population Survey (CPS) [1]

>To replicate our extract of the CSP data, follow these steps:
>1. **Create/Log into an IPUMS Account**:  
>Go to [IPUMS CPS](https://cps.ipums.org/cps/) and sign in with your account. If you do not have an account, you will need to register for one.
>2. **Obtain an API key**:  
>Obtain an IPUMS API key [here](https://account.ipums.org/api_keys).
>3. **Paste your API key**:  
>Paste your API key in line 20 of the `cps.py` program instead of the string `'ipums_api_key'`.

#### Bureau of Justice Statistics' National Prisoner Statistics [2]

>The data can be downloaded in `.tsv` format [here](https://www.icpsr.umich.edu/web/NACJD/studies/38871) and a dictionary is available [here](https://www.icpsr.umich.edu/web/NACJD/studies/38871/datadocumentation#). A copy of the data file (`nps.tsv`) is provided in `.tsv` format as part of this archive in the `Data/Raw/NPS` directory of this replication package. The data is in the public domain. To download the data, follow these steps:
>1. **Create/Log into an ICPSR Account**:  
>Go to [ICPSR](https://login.icpsr.umich.edu/realms/icpsr/protocol/openid-connect/auth?client_id=icpsr-web-prod&response_type=code&login=true&redirect_uri=https://www.icpsr.umich.edu/web/oauth/callback) and sign in with your account. If you do not have an account, you will need to register for one.
>2. **Download the data**:  
>Download the data [here](https://www.icpsr.umich.edu/web/NACJD/studies/38871/datadocumentation#) under "Download" and choose the "Delimited" format.

#### Bureau of Justice Statistics' Annual Survey of Jails [3]

>The data can be downloaded in `.tsv` and `.dta` format [here](https://www.icpsr.umich.edu/web/NACJD/series/7) for each available year between 1985 and 2022. Copies of the data files (`"year".tsv` for "year" taking values between 1985 and 2022) are provided in `.tsv` and `.dta` format (depending on the year) as part of this archive in the `Data/Raw/ASJ` directory of this replication package. The data is in the public domain. To download the data, follow these steps:
>1. **Create/Log into an ICPSR Account**:  
>Go to [ICPSR](https://login.icpsr.umich.edu/realms/icpsr/protocol/openid-connect/auth?client_id=icpsr-web-prod&response_type=code&login=true&redirect_uri=https://www.icpsr.umich.edu/web/oauth/callback) and sign in with your account. If you do not have an account, you will need to register for one.
>2. **Download the data**:  
>Click [here](https://www.icpsr.umich.edu/web/NACJD/series/7), select a specific survey year, and under "Download", choose the "Delimited" format.

#### Centers for Disease Control (CDC) and Prevention's National Center for Health Statistics (NCHS) [4], [5], [6]

>The data we use to calculate survival rates by age can be obtained from three locations. For the years 1984 to 1998, the authors collected the data from the publicly available PDF files of the [U.S. life tables](https://www.cdc.gov/nchs/products/life_tables.htm). This data file (`lifetables.csv`) is provided in `.csv` format as part of this archive in the `Data/Raw/CDC/` directory of this replication package. For the years 1999 to 2020, we obtained the data [here](https://wonder.cdc.gov/mcd-icd10.html). For the years 2018 to 2022, we obtained the data [here](https://wonder.cdc.gov/mcd-icd10-expanded.html). Copies of these data files (`Multiple Cause of Death, 1999-2020.txt` and `Multiple Cause of Death, 2018-2022, Single Race.txt`) are provided in `.txt` format as part of this archive in the `Data/Raw/CDC` directory of this replication package. All of this data is in the public domain. To replicate our extracts of the CDC NCHS data, follow these steps:
>1. **Click on the links provided above**:  
>Scroll down and agree to abide by the terms of data use.
>2. **Organize table layout**:  
>In section 1 named "organize table layout", select to group results by "Census Region", "Single-Year Ages", "Hispanic Origin", "Race" (or "Single Race 6" for the 2018-2022 data), and "Year".
>3. **Select demographics**:  
>In the "Race" (or "Single Race 6" for the 2018-2022 data) tab of section 3 named "select demographics", click on "Black or African American" and "White".
>4. **Other options**:
>In section 8 named "other options", click on "export results" and press "send" at the bottom of the page.

#### Bureau of Labor Statistics' Consumer Expenditure Survey (CEX) [7]

>The data can be downloaded in `.csv` format [here](https://www.bls.gov/cex/pumd_data.htm#csv), its documentation is available [here](https://www.bls.gov/cex/pumd_doc.htm), and a dictionary is available [here](https://www.bls.gov/cex/pumd/ce-pumd-interview-diary-dictionary.xlsx). Copies of the data files are provided as part of this archive in the `Data/Raw/CEX` directory of this replication package. The data is in the public domain. In each directory `Data/Raw/CEX/intrvw"yy"` where "yy" stands for the last two digits of each year between 1984 and 2022, the authors created files named `codebook.csv` linking expenditure categories to their [UCC code](https://www.bls.gov/cex/pumd/stubs.zip), a [weight adjustment](https://www.bls.gov/cex/cecomparison/pce_concordance.xlsx), whether they should be considered as consumption, and whether they should be considered as durable consumption.

#### U.S. Census Bureau's Population Estimates Program (PEP) [8]

>The data we use to calculate population estimates can be obtained from several locations. For the years 1984 to 1989, the data can be downloaded in `.txt` format [here](https://www2.census.gov/programs-surveys/popest/datasets/1980-1990/state/asrh/st_int_asrh.txt) and a dictionary is available [here](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/1980-1990/st_int_asrh_doc.txt). For the years 1990 to 1999, the data can be downloaded in `.txt` format [here](https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-1990-2000-state-and-county-characteristics.html) and a dictionary is available [here](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/1990-2000/stch-intercensal_layout.txt). Copies of the data files are provided as part of this archive in the `Data/Raw/POP` directory of this replication package. The data is in the public domain.

#### Centers for Disease Control (CDC) and Prevention's National Health Interview Survey (NHIS) [9]

>To replicate our extract of the NHIS data, follow these steps:
>1. **Create/Log into an IPUMS Account**:  
>Go to [IPUMS NHIS](https://nhis.ipums.org/nhis/) and sign in with your account. If you do not have an account, you will need to register for one.
>2. **Obtain an API key**:  
>Obtain an IPUMS API key [here](https://account.ipums.org/api_keys).
>3. **Paste your API key**:  
>Paste your API key in line 18 of the `nhis.py` program instead of the string `'ipums_api_key'`.

### Preliminary code during the editorial process

> Code for data cleaning and analysis is provided as part of the replication package. It is available [here](https://github.com/jfbrou/Dignity) for review. It will be uploaded to the [AEA Data and Code Repository](https://www.openicpsr.org/openicpsr/aea) once the paper has been accepted.

## References

[1] **Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry.**  
*IPUMS CPS: Version 12.0 [dataset].*  
Minneapolis, MN: IPUMS, 2024.  
[https://doi.org/10.18128/D030.V12.0](https://doi.org/10.18128/D030.V12.0)

[2] **United States. Bureau of Justice Statistics.**  
*National Prisoner Statistics, [United States], 1978-2022.*  
Inter-university Consortium for Political and Social Research [distributor], 2024-01-10.  
[https://doi.org/10.3886/ICPSR38871.v1](https://doi.org/10.3886/ICPSR38871.v1)

[3] **United States Department of Justice. Office of Justice Programs. Bureau of Justice Statistics.**  
*Annual Survey of Jails: Jurisdiction-Level Data, 1989.*  
[distributor], 2005-11-04.  
[https://doi.org/10.3886/ICPSR09373.v2](https://doi.org/10.3886/ICPSR09373.v2)

[4] **United States Department of Health and Human Services (US DHHS),  
Centers for Disease Control and Prevention (CDC),  
National Center for Health Statistics (NCHS).**  
*Multiple Cause of Death 1999-2020 on CDC WONDER Online Database,* released 2021.  
Data are compiled from data provided by the 57 vital statistics jurisdictions through the Vital Statistics Cooperative Program.

[5] **United States Department of Health and Human Services (US DHHS),  
Centers for Disease Control and Prevention (CDC),  
National Center for Health Statistics (NCHS).**  
*Multiple Cause of Death by Single Race 2018-2022 on CDC WONDER Online Database,* released 2024.  
Data are compiled from data provided by the 57 vital statistics jurisdictions through the Vital Statistics Cooperative Program.

[6] **United States Department of Health and Human Services (US DHHS),  
Centers for Disease Control and Prevention (CDC),  
National Center for Health Statistics (NCHS).**  
*Life Tables.*  
Available at: [https://www.cdc.gov/nchs/products/life_tables.htm](https://www.cdc.gov/nchs/products/life_tables.htm)

[7] **U.S. Bureau of Labor Statistics.**  
*Consumer Expenditure Surveys (CEX).*  
Available at: [https://www.bls.gov/cex/](https://www.bls.gov/cex/)

[8] **U.S. Census Bureau.**  
*Population Estimates Data.*  
Available at: [https://www.census.gov/programs-surveys/popest/data/data-sets.html](https://www.census.gov/programs-surveys/popest/data/data-sets.html)


[9] **Lynn A. Blewett, Julia A. Rivera Drew, Miriam L. King, Kari C.W. Williams,  
Daniel Backman, Annie Chen, and Stephanie Richards.**  
*IPUMS Health Surveys: National Health Interview Survey, Version 7.4 [dataset].*  
Minneapolis, MN: IPUMS, 2024.  
[https://doi.org/10.18128/D070.V7.4](https://doi.org/10.18128/D070.V7.4)

---

## Acknowledgements