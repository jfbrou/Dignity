---
contributors:
  - Jean-Félix Brouillette
  - Charles I. Jones
  - Peter J. Klenow
---

# Replication Package for: "Race and Economic Well-Being in the United States"

## Overview

This replication package contains two main Python programs:

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
3. **Update `.env`**:  
     Insert your IPUMS API key at line 1, replacing the value of `ipums_api_key`.

#### 2. Bureau of Justice Statistics' National Prisoner Statistics [2]

- Data in `.tsv` format is available [here](https://www.icpsr.umich.edu/web/NACJD/studies/38871) along with documentation.
- A copy is provided in `Data/Raw/NPS`.
- To download directly:
  1. **Create/Log into an ICPSR Account**:  
     Visit [ICPSR](https://login.icpsr.umich.edu) to sign in or register.
  2. **Download the Data**:  
     Access [this page](https://www.icpsr.umich.edu/web/NACJD/studies/38871/datadocumentation#) and select the "Delimited" format under "Download".

#### 3. Bureau of Justice Statistics' Annual Survey of Jails (ASJ) [3]

- Annual data files (in `.tsv` or `.dta` format) from 1985 to 2022 are available [here](https://www.icpsr.umich.edu/web/NACJD/series/7).
- Copies are provided in `Data/Raw/ASJ`.
- To download directly:
  1. **Create/Log into an ICPSR Account**:  
     Visit [ICPSR](https://login.icpsr.umich.edu) to sign in or register.
  2. **Download the Data**:  
     Access [this page](https://www.icpsr.umich.edu/web/NACJD/series/7), select the relevant survey years, and select the "Delimited" format under "Download".

#### 4. CDC/NCHS Mortality and Life Tables Data [4], [5], [6]

- For 1984–2017, life table data are from PDF files available [here](https://www.cdc.gov/nchs/products/life_tables.htm). A cleaned `.csv` file (`lifetables.csv`) created by the authors is provided in `Data/Raw/CDC`.
- For 2018–2020, data come from [CDC WONDER](https://wonder.cdc.gov/mcd-icd10.html).
   - A copy is provided in `Data/Raw/CDC`.
- For 2021–2022, data come from [CDC WONDER](https://wonder.cdc.gov/mcd-icd10-expanded.html).
   - A copy is provided in `Data/Raw/CDC`.
- To replicate these last two extracts:
  1. Visit the provided CDC WONDER links, agree to terms.
  2. Under "Organize Table Layout", select grouping by "Census Region", "Single-Year Ages", "Hispanic Origin", "Race"/"Single Race 6" (as appropriate), and "Year".
  3. In the "Race" section, select "Black or African American" and "White".
  4. Under "Other Options" choose "Export Results" and click "Send".

#### 5. Bureau of Labor Statistics' Consumer Expenditure Survey (CEX) [7]

- Data in `.csv` format is available [here](https://www.bls.gov/cex/pumd_data.htm#csv) with documentation [here](https://www.bls.gov/cex/pumd_doc.htm).
- A codebook (created by the authors) linking expenditure categories to UCC codes and other details is in each `Data/Raw/CEX/intrvw"yy"` directory.
- Copies are provided in `Data/Raw/CEX`.

#### 6. U.S. Census Bureau's Population Estimates Program (PEP) [8]

- Data for 1984–1989 is available [here](https://www2.census.gov/programs-surveys/popest/datasets/1980-1990/state/asrh/st_int_asrh.txt) and documentation is available [here](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/1980-1990/st_int_asrh_doc.txt).
- Data for 1990–1999 is available [here](https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-1990-2000-state-and-county-characteristics.html) and documentation is available [here](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/1990-2000/stch-intercensal_layout.txt).
- Copies are provided in `Data/Raw/POP`.

#### 7. CDC/NCHS National Health Interview Survey (NHIS) [9]

- To replicate the NHIS data extract:
  1. **Create/Log into an IPUMS Account**:  
     Visit [IPUMS NHIS](https://nhis.ipums.org/nhis/) and sign in or register.
  2. **Obtain an API Key**:  
     [Get an API key here](https://account.ipums.org/api_keys).
  3. **Update `.env`**:  
     Insert your IPUMS API key at line 1, replacing the value of `ipums_api_key`.

#### 8. Bureau of Economic Analysis' (BEA) NIPA Tables [10], [11], [12]

- Data from NIPA tables (2.4.5, 2.1, 1.1.4) are sourced via the BEA API.
- To replicate:
  1. **Create/Log into a BEA Account**:  
     Sign up at [BEA](https://apps.bea.gov/API/signup/).
  2. **Obtain an API Key**:  
     Get a BEA API key [here](https://apps.bea.gov/API/signup/).
  3. **Update `.env`**:  
     Insert your BEA API key at line 2, replacing the value of `bea_api_key`.

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
  - The file "`Programs/requirements.txt`" lists all dependencies.

### Controlled Randomness

- A random seed is set at line 9 of program `Programs/Preparation/bootstrap.py`.
- A random seed is set at line 9 of program `Programs/Preparation/bootstrap_cps.py`.
- A random seed is set at line 9 of program `Programs/Preparation/bootstrap_cex.py`.

### Memory, Runtime, Storage Requirements

## Description of programs/code

- `Programs/data.py` prepares the datasets required for the analysis and runs the programs below:
  - `Programs/Preparation/directories.py` defines the necessary directories.
  - `Programs/Preparation/functions.py` contains functions used in the preparation programs.
  - `Programs/Preparation/cdc.py` prepares the mortality data used to calculate survival rates.
  - `Programs/Preparation/population.py` prepares the population data used to calculate incarceration rates.
  - `Programs/Preparation/survival.py` calculates survival rates.
  - `Programs/Preparation/incarceration.py` calculates incarceration rates.
  - `Programs/Preparation/nhis.py` prepares the NHIS data used to calculate HALex scores.
  - `Programs/Preparation/cps.py` prepares the CPS data used to calculate leisure, earnings, and unemployment rates.
  - `Programs/Preparation/cex.py` prepares the CEX data used to calculate consumption.
  - `Programs/Preparation/dignity.py` combines the data prepared in the above programs to calculate the ingredients of our consumption-equivalent welfare metric.
- `Programs/analysis.py` generates the figures and tables presented in the paper and runs the programs below:
  - `Programs/Analysis/directories.py` defines the necessary directories.
  - `Programs/Analysis/functions.py` contains functions used in the analysis programs.
  - `Programs/Analysis/figures.py` produces the figures in the paper and online appendix.
  - `Programs/Analysis/tables.py` produces the tables in the paper.

## Instructions to Replicators

> INSTRUCTIONS: The first two sections ensure that the data and software necessary to conduct the replication have been collected. This section then describes a human-readable instruction to conduct the replication. This may be simple, or may involve many complicated steps. It should be a simple list, no excess prose. Strict linear sequence. If more than 4-5 manual steps, please wrap a main program/Makefile around them, in logical sequences. Examples follow.

- Edit `programs/config.do` to adjust the default path
- Run `programs/00_setup.do` once on a new system to set up the working environment. 
- Download the data files referenced above. Each should be stored in the prepared subdirectories of `data/`, in the format that you download them in. Do not unzip. Scripts are provided in each directory to download the public-use files. Confidential data files requested as part of your FSRDC project will appear in the `/data` folder. No further action is needed on the replicator's part.
- Run `programs/01_main.do` to run all steps in sequence.

### Details

- `programs/00_setup.do`: will create all output directories, install needed ado packages. 
   - If wishing to update the ado packages used by this archive, change the parameter `update_ado` to `yes`. However, this is not needed to successfully reproduce the manuscript tables. 
- `programs/01_dataprep`:  
   - These programs were last run at various times in 2018. 
   - Order does not matter, all programs can be run in parallel, if needed. 
   - A `programs/01_dataprep/main.do` will run them all in sequence, which should take about 2 hours.
- `programs/02_analysis/main.do`.
   - If running programs individually, note that ORDER IS IMPORTANT. 
   - The programs were last run top to bottom on July 4, 2019.
- `programs/03_appendix/main-appendix.do`. The programs were last run top to bottom on July 4, 2019.
- Figure 1: The figure can be reproduced using the data provided in the folder “2_data/data_map”, and ArcGIS Desktop (Version 10.7.1) by following these (manual) instructions:
  - Create a new map document in ArcGIS ArcMap, browse to the folder
“2_data/data_map” in the “Catalog”, with files  "provinceborders.shp", "lakes.shp", and "cities.shp". 
  - Drop the files listed above onto the new map, creating three separate layers. Order them with "lakes" in the top layer and "cities" in the bottom layer.
  - Right-click on the cities file, in properties choose the variable "health"... (more details)

## List of tables and programs


> INSTRUCTIONS: Your programs should clearly identify the tables and figures as they appear in the manuscript, by number. Sometimes, this may be obvious, e.g. a program called "`table1.do`" generates a file called `table1.png`. Sometimes, mnemonics are used, and a mapping is necessary. In all circumstances, provide a list of tables and figures, identifying the program (and possibly the line number) where a figure is created.
>
> NOTE: If the public repository is incomplete, because not all data can be provided, as described in the data section, then the list of tables should clearly indicate which tables, figures, and in-text numbers can be reproduced with the public material provided.

The provided code reproduces:

- [ ] All numbers provided in text in the paper
- [ ] All tables and figures in the paper
- [ ] Selected tables and figures in the paper, as explained and justified below.

| Figure/Table #    | Program                  | Line Number | Output file                      | Note                            |
|-------------------|--------------------------|-------------|----------------------------------|---------------------------------|
| Table 1           | 02_analysis/table1.do    |             | summarystats.csv                 ||
| Table 2           | 02_analysis/table2and3.do| 15          | table2.csv                       ||
| Table 3           | 02_analysis/table2and3.do| 145         | table3.csv                       ||
| Figure 1          | n.a. (no data)           |             |                                  | Source: Herodus (2011)          |
| Figure 2          | 02_analysis/fig2.do      |             | figure2.png                      ||
| Figure 3          | 02_analysis/fig3.do      |             | figure-robustness.png            | Requires confidential data      |


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