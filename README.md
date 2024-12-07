---
contributors:
  - Jean-Félix Brouillette
  - Charles I. Jones
  - Peter J. Klenow
---

# Replication Package for: "Race and Economic Well-Being in the United States"

## Overview

This replication package contains three main Python programs:

- **`Programs/directories.py`**: Contains the paths to the raw data files and the directories where the processed data will be saved.
- **`Programs/data.py`**: Prepares the datasets required for the analysis.
- **`Programs/analysis.py`**: Generates the figures and tables presented in the paper.

**Note:** Running these scripts end-to-end may take up to approximately two hours.

## Data Availability Statement

### Rights and Permissions

- The authors certify that they have legitimate access to, and permission to use, all datasets employed in the manuscript.
- The authors confirm that they hold the necessary permissions to redistribute and publish the data included in this replication package.

### Summary of Data Availability

- **All data are publicly available.**

### Data Sources and Instructions

Below we list the data sources used in this study, along with instructions on how to replicate the exact data extracts if needed. In most cases, this involves creating an account with the data provider (if not already done), obtaining an API key where relevant, and following the specified steps.

#### 1. U.S. Census Bureau's Current Population Survey (CPS) [1]

To replicate the CPS extracts:

1. **Create/Log into an IPUMS Account**:  
   Visit [IPUMS CPS](https://cps.ipums.org/cps/) and sign in or register.
2. **Obtain an API Key**:  
   Request an IPUMS API key [here](https://account.ipums.org/api_keys).
3. **Update `Programs/directories.py`**:  
     Insert your IPUMS API key at line 9, replacing `'ipums_api_key'`.

#### 2. Bureau of Justice Statistics' National Prisoner Statistics [2]

- Data in `.tsv` format is available [here](https://www.icpsr.umich.edu/web/NACJD/studies/38871) along with documentation.
- A copy of `nps.tsv` is provided in `Data/Raw/NPS`.
- To download directly:
  1. **Create/Log into an ICPSR Account**:  
     Visit [ICPSR](https://login.icpsr.umich.edu) to sign in or register.
  2. **Download the Data**:  
     Access [this page](https://www.icpsr.umich.edu/web/NACJD/studies/38871/datadocumentation#) and select "Delimited" format under "Download".

#### 3. Bureau of Justice Statistics' Annual Survey of Jails (ASJ) [3]

- Annual data files (`"year".tsv` or `.dta`) from 1985 to 2022 are available at [this link](https://www.icpsr.umich.edu/web/NACJD/series/7).
- Copies are included in `Data/Raw/ASJ`.
- To download:
  1. **Create/Log into an ICPSR Account**:  
     Visit [ICPSR](https://login.icpsr.umich.edu) to sign in or register.
  2. **Select the Year and Download**:  
     From [here](https://www.icpsr.umich.edu/web/NACJD/series/7), choose a survey year and select "Delimited" format.

#### 4. CDC/NCHS Mortality and Life Tables Data [4], [5], [6]

- For 1984–2017, life table data are from PDF files available [here](https://www.cdc.gov/nchs/products/life_tables.htm). A cleaned `.csv` file (`lifetables.csv`) created by the authors is in `Data/Raw/CDC/`.
- For 2018–2020, data come from [CDC WONDER](https://wonder.cdc.gov/mcd-icd10.html).
- For 2021–2022, data come from [CDC WONDER](https://wonder.cdc.gov/mcd-icd10-expanded.html).
- Copies of processed `.txt` files are in `Data/Raw/CDC`.
- To replicate these extracts:
  1. Visit the provided CDC WONDER links, agree to terms.
  2. Under "Organize Table Layout", select grouping by "Census Region", "Single-Year Ages", "Hispanic Origin", "Race"/"Single Race 6" (as appropriate), and "Year".
  3. In the "Race" section, select "Black or African American" and "White".
  4. Under "Other Options" choose "Export Results" and click "Send".

#### 5. Bureau of Labor Statistics' Consumer Expenditure Survey (CEX) [7]

- Data in `.csv` format is available [here](https://www.bls.gov/cex/pumd_data.htm#csv) with documentation [here](https://www.bls.gov/cex/pumd_doc.htm).
- A codebook (created by the authors) linking expenditure categories to UCC codes and other details is in each `Data/Raw/CEX/intrvw"yy"` directory.
- Data for all relevant years (1984–2022) are included in `Data/Raw/CEX`.

#### 6. U.S. Census Bureau's Population Estimates Program (PEP) [8]

- Data for 1984–1989: [.txt](https://www2.census.gov/programs-surveys/popest/datasets/1980-1990/state/asrh/st_int_asrh.txt) and documentation [.txt](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/1980-1990/st_int_asrh_doc.txt).
- Data for 1990–1999: [.txt](https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-1990-2000-state-and-county-characteristics.html) and documentation [.txt](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/1990-2000/stch-intercensal_layout.txt).
- All files are stored in `Data/Raw/POP`.

#### 7. CDC/NCHS National Health Interview Survey (NHIS) [9]

- To replicate the NHIS data extract:
  1. **Create/Log into an IPUMS Account**:  
     Visit [IPUMS NHIS](https://nhis.ipums.org/nhis/) and sign in or register.
  2. **Obtain an API Key**:  
     [Get an API key here](https://account.ipums.org/api_keys).
  3. **Update `Programs/directories.py`**:  
     Insert your IPUMS API key at line 9, replacing `'ipums_api_key'`.

#### 8. Bureau of Economic Analysis' (BEA) NIPA Tables [10], [11], [12]

- Data from NIPA tables (2.4.5, 2.1, 1.1.4) are sourced via the BEA API.
- To replicate:
  1. **Create/Log into a BEA Account**:  
     Sign up at [BEA](https://apps.bea.gov/API/signup/).
  2. **Obtain an API Key**:  
     Get a BEA API key [here](https://apps.bea.gov/API/signup/).
  3. **Update `Programs/directories.py`**:  
     Insert your BEA API key at line 10, replacing `'bea_api_key'`.

### Dataset Summary Table

| Dataset                                           | Source/Provider                | Provided in Package?        | Access Method                   | Format(s) Included      |
|---------------------------------------------------|--------------------------------|-----------------------------|---------------------------------|-------------------------|
| Current Population Survey (CPS)                   | U.S. Census Bureau / IPUMS CPS | Not directly (API required) | API via IPUMS                   | N/A (retrieved via API) |
| National Prisoner Statistics (NPS)                | U.S. BJS / ICPSR               | Yes (in `Data/Raw/NPS`)     | Direct download via ICPSR       | TSV                     |
| Annual Survey of Jails (ASJ)                      | U.S. BJS / ICPSR               | Yes (in `Data/Raw/ASJ`)     | Direct download via ICPSR       | TSV, DTA                |
| CDC WONDER (Mortality, 1999–2020, 2018–2022)      | CDC / NCHS, WONDER database    | Yes (in `Data/Raw/CDC`)     | Direct download via WONDER      | TXT                     |
| CDC NCHS Life Tables (1984–2017)                  | CDC / NCHS                     | Yes (in `Data/Raw/CDC`)     | PDF originals, processed CSV    | CSV                     |
| Consumer Expenditure Survey (CEX)                 | U.S. BLS                       | Yes (in `Data/Raw/CEX`)     | Direct download via BLS         | CSV                     |
| Population Estimates (1984–1999)                  | U.S. Census Bureau             | Yes (in `Data/Raw/POP`)     | Direct download via U.S. Census | TXT                     |
| National Health Interview Survey (NHIS)           | CDC NCHS / IPUMS NHIS          | Not directly (API required) | API via IPUMS                   | N/A (retrieved via API) |
| BEA NIPA Tables (2.4.5, 2.2.4, 2.1)               | U.S. BEA                       | Not directly (API required) | API via BEA                     | N/A (retrieved via API) |

## Computational Requirements

### Software Requirements

- Python 3.10.9
  - The file "`Programs/requirements.txt`" lists all dependencies, please run "`pip install -r requirements.txt`" as the first step.

### Controlled Randomness

- A random seed is set at line 8 of program `Programs/bootstrap.py`.
- A random seed is set at line 8 of program `Programs/bootstrap_cps.py`.
- A random seed is set at line 8 of program `Programs/bootstrap_cex.py`.

### Memory, Runtime, Storage Requirements

### Preliminary Code

Preliminary data cleaning and analysis code is provided in this package. A version for review is available on [GitHub](https://github.com/jfbrou/Dignity). Once the paper is accepted, code will be uploaded to the [AEA Data and Code Repository](https://www.openicpsr.org/openicpsr/aea).

## References

[1] **Sarah Flood et al. (2024).** *IPUMS CPS: Version 12.0 [dataset].* Minneapolis, MN: IPUMS.  
[https://doi.org/10.18128/D030.V12.0](https://doi.org/10.18128/D030.V12.0)

[2] **U.S. Bureau of Justice Statistics (2024).** *National Prisoner Statistics, [United States], 1978-2022.* ICPSR [distributor].  
[https://doi.org/10.3886/ICPSR38871.v1](https://doi.org/10.3886/ICPSR38871.v1)

[3] **U.S. Department of Justice, BJS (2005).** *Annual Survey of Jails: Jurisdiction-Level Data, 1989.*  
[https://doi.org/10.3886/ICPSR09373.v2](https://doi.org/10.3886/ICPSR09373.v2)

[4] **US DHHS, CDC, NCHS (2021).** *Multiple Cause of Death 1999-2020 on CDC WONDER.*  
Data from 57 vital statistics jurisdictions.

[5] **US DHHS, CDC, NCHS (2024).** *Multiple Cause of Death by Single Race 2018-2022 on CDC WONDER.*  
Data from 57 vital statistics jurisdictions.

[6] **US DHHS, CDC, NCHS.** *Life Tables.*  
[https://www.cdc.gov/nchs/products/life_tables.htm](https://www.cdc.gov/nchs/products/life_tables.htm)

[7] **U.S. Bureau of Labor Statistics.** *Consumer Expenditure Surveys (CEX).*  
[https://www.bls.gov/cex/](https://www.bls.gov/cex/)

[8] **U.S. Census Bureau.** *Population Estimates Data.*  
[https://www.census.gov/programs-surveys/popest/data/data-sets.html](https://www.census.gov/programs-surveys/popest/data/data-sets.html)

[9] **Lynn A. Blewett et al. (2024).** *IPUMS Health Surveys: NHIS, Version 7.4 [dataset].* Minneapolis, MN: IPUMS.  
[https://doi.org/10.18128/D070.V7.4](https://doi.org/10.18128/D070.V7.4)

[10] **U.S. Bureau of Economic Analysis.** “Table 2.4.5: Personal Consumption Expenditures by Type of Product”  
[https://apps.bea.gov/iTable/](https://apps.bea.gov/iTable/) (accessed December 6, 2024).

[11] **U.S. Bureau of Economic Analysis.** “Table 2.2.4: Personal Consumption Expenditures by Function”  
[https://apps.bea.gov/iTable/](https://apps.bea.gov/iTable/) (accessed December 6, 2024).

[12] **U.S. Bureau of Economic Analysis.** “Table 2.1: Personal Income and Its Disposition”  
[https://apps.bea.gov/iTable/](https://apps.bea.gov/iTable/) (accessed December 6, 2024).

## Acknowledgements

We gratefully acknowledge all contributors and the respective data providers for making these datasets publicly available and accessible.