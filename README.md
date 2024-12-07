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
>In section 1 named "organize table layout", select to group results by "Census Region", "Single-Year Ages", "Race" (or "Single Race 6" for the 2018-2022 data), and "Year".
>3. **Select demographics**:  
>In the "Race" (or "Single Race 6" for the 2018-2022 data) tab of section 3 named "select demographics", click on "Black or African American" and "White".
>4. **Other options**:
>In section 8 named "other options", click on "export results" and press "send" at the bottom of the page.

#### Bureau of Labor Statistics' Consumer Expenditure Survey (CEX) [7]

>The data can be downloaded in `.csv` format [here](https://www.bls.gov/cex/pumd_data.htm#csv), its documentation is available [here](https://www.bls.gov/cex/pumd_doc.htm), and a dictionary is available [here](https://www.bls.gov/cex/pumd/ce-pumd-interview-diary-dictionary.xlsx). Copies of the data files are provided as part of this archive in the `Data/Raw/CEX` directory of this replication package. The data is in the public domain. In each directory `Data/Raw/CEX/intrvw"yy"` where "yy" stands for the last two digits of each year between 1984 and 2022, the authors created files named `codebook.csv` linking expenditure categories to their [UCC code](https://www.bls.gov/cex/pumd/stubs.zip), a [weight adjustment](https://www.bls.gov/cex/cecomparison/pce_concordance.xlsx), whether they should be considered as consumption, and whether they should be considered as durable consumption.

#### U.S. Census Bureau's Population Estimates Program (PEP)

#### Centers for Disease Control (CDC) and Prevention's National Health Interview Survey (NHIS)

>To replicate our extract of the NHIS data, follow these steps:
>1. **Create/Log into an IPUMS Account**:  
>Go to [IPUMS NHIS](https://nhis.ipums.org/nhis/) and sign in with your account. If you do not have an account, you will need to register for one.
>2. **Obtain an API key**:  
>Obtain an IPUMS API key [here](https://account.ipums.org/api_keys).
>3. **Paste your API key**:  
>Paste your API key in line ZZZ of the `nhis.py` program instead of the string `'ipums_api_key'`.

### Preliminary code during the editorial process

> Code for data cleaning and analysis is provided as part of the replication package. It is available [here](https://github.com/jfbrou/Dignity) for review. It will be uploaded to the [AEA Data and Code Repository](https://www.openicpsr.org/openicpsr/aea) once the paper has been accepted.

## Dataset list

| Data file | Source | Notes    |Provided |
|-----------|--------|----------|---------|
| `data/raw/lbd.dta` | LBD | Confidential | No |
| `data/raw/terra.dta` | IPUMS Terra | As per terms of use | Yes |
| `data/derived/regression_input.dta`| All listed | Combines multiple data sources, serves as input for Table 2, 3 and Figure 5. | Yes |


## Computational requirements

> INSTRUCTIONS: In general, the specific computer code used to generate the results in the article will be within the repository that also contains this README. However, other computational requirements - shared libraries or code packages, required software, specific computing hardware - may be important, and is always useful, for the goal of replication. Some example text follows. 

> INSTRUCTIONS: We strongly suggest providing setup scripts that install/set up the environment. Sample scripts for [Stata](https://github.com/gslab-econ/template/blob/master/config/config_stata.do),  [R](https://github.com/labordynamicsinstitute/paper-template/blob/master/programs/global-libraries.R), [Julia](https://github.com/labordynamicsinstitute/paper-template/blob/master/programs/packages.jl) are easy to set up and implement. Specific software may have more sophisticated tools: [Python](https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability), [Julia](https://julia.quantecon.org/more_julia/tools_editors.html#Package-Environments).

### Software Requirements

> INSTRUCTIONS: List all of the software requirements, up to and including any operating system requirements, for the entire set of code. It is suggested to distribute most dependencies together with the replication package if allowed, in particular if sourced from unversioned code repositories, Github repos, and personal webpages. In all cases, list the version *you* used. All packages should be listed in human-readable form in this README, but should also be included in a setup or install script.

- [X] The replication package contains one or more programs to install all dependencies and set up the necessary directory structure.

- Python 3.6.4
  - `pandas` 0.24.2
  - `numpy` 1.16.4
  - the file "`requirements.txt`" lists these dependencies, please run "`pip install -r requirements.txt`" as the first step. See [https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability](https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability) for further instructions on creating and using the "`requirements.txt`" file.

### Controlled Randomness

> INSTRUCTIONS: Some estimation code uses random numbers, almost always provided by pseudorandom number generators (PRNGs). For reproducibility purposes, these should be provided with a deterministic seed, so that the sequence of numbers provided is the same for the original author and any replicators. While this is not always possible, it is a requirement by many journals' policies. The seed should be set once, and not use a time-stamp. If using parallel processing, special care needs to be taken. If using multiple programs in sequence, care must be taken on how to call these programs, ideally from a main program, so that the sequence is not altered. If no PRNG is used, check the other box.

- [X] Random seed is set at line _____ of program ______
- [X] No Pseudo random generator is used in the analysis described here.

### Memory, Runtime, Storage Requirements

> INSTRUCTIONS: Memory and compute-time requirements may also be relevant or even critical. Some example text follows. It may be useful to break this out by Table/Figure/section of processing. For instance, some estimation routines might run for weeks, but data prep and creating figures might only take a few minutes. You should also describe how much storage is required in addition to the space visible in the typical repository, for instance, because data will be unzipped, data downloaded, or temporary files written.

#### Summary

Approximate time needed to reproduce the analyses on a standard (CURRENT YEAR) desktop machine:

- [ ] <10 minutes
- [ ] 10-60 minutes
- [ ] 1-2 hours
- [ ] 2-8 hours
- [ ] 8-24 hours
- [ ] 1-3 days
- [ ] 3-14 days
- [ ] > 14 days

Approximate storage space needed:

- [ ] < 25 MBytes
- [ ] 25 MB - 250 MB
- [ ] 250 MB - 2 GB
- [ ] 2 GB - 25 GB
- [ ] 25 GB - 250 GB
- [ ] > 250 GB

- [ ] Not feasible to run on a desktop machine, as described below.

#### Details

The code was last run on a **4-core Intel-based laptop with MacOS version 10.14.4 with 200GB of free space**. 

Portions of the code were last run on a **32-core Intel server with 1024 GB of RAM, 12 TB of fast local storage**. Computation took **734 hours**. 

Portions of the code were last run on a **12-node AWS R3 cluster, consuming 20,000 core-hours, with 2TB of attached storage**.  

> INSTRUCTIONS: Identifiying hardware and OS can be obtained through a variety of ways:
> Some of these details can be found as follows:
>
> - (Windows) by right-clicking on "This PC" in File Explorer and choosing "Properties"
> - (Mac) Apple-menu > "About this Mac"
> - (Linux) see code in [linux-system-info.sh](https://github.com/AEADataEditor/replication-template/blob/master/tools/linux-system-info.sh)`


## Description of programs/code

> INSTRUCTIONS: Give a high-level overview of the program files and their purpose. Remove redundant/ obsolete files from the Replication archive.

- Programs in `programs/01_dataprep` will extract and reformat all datasets referenced above. The file `programs/01_dataprep/main.do` will run them all.
- Programs in `programs/02_analysis` generate all tables and figures in the main body of the article. The program `programs/02_analysis/main.do` will run them all. Each program called from `main.do` identifies the table or figure it creates (e.g., `05_table5.do`).  Output files are called appropriate names (`table5.tex`, `figure12.png`) and should be easy to correlate with the manuscript.
- Programs in `programs/03_appendix` will generate all tables and figures  in the online appendix. The program `programs/03_appendix/main-appendix.do` will run them all. 
- Ado files have been stored in `programs/ado` and the `main.do` files set the ADO directories appropriately. 
- The program `programs/00_setup.do` will populate the `programs/ado` directory with updated ado packages, but for purposes of exact reproduction, this is not needed. The file `programs/00_setup.log` identifies the versions as they were last updated.
- The program `programs/config.do` contains parameters used by all programs, including a random seed. Note that the random seed is set once for each of the two sequences (in `02_analysis` and `03_appendix`). If running in any order other than the one outlined below, your results may differ.

### (Optional, but recommended) License for Code

> INSTRUCTIONS: Most journal repositories provide for a default license, but do not impose a specific license. Authors should actively select a license. This should be provided in a LICENSE.txt file, separately from the README, possibly combined with the license for any data provided. Some code may be subject to inherited license requirements, i.e., the original code author may allow for redistribution only if the code is licensed under specific rules - authors should check with their sources. For instance, some code authors require that their article describing the econometrics of the package be cited. Licensing can be complex. Some non-legal guidance may be found [here](https://social-science-data-editors.github.io/guidance/Licensing_guidance.html).

The code is licensed under a MIT/BSD/GPL [choose one!] license. See [LICENSE.txt](LICENSE.txt) for details.

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


---

## Acknowledgements