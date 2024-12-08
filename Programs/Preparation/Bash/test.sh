#!/bin/bash

# Source the .env file to get DB credentials
set -a
source ../../../.env
set +a

#SBATCH --job-name=test
#SBATCH --error=${error}errors_%x_%j.err
#SBATCH --output=${output}output_%x_%j.out
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=${email}

cd ${home}Programs/Preparation
ml python/3.10
python3 test.py