#!/bin/bash

# Get absolute path to directory
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the .env file to get DB credentials
set -a
source ${ABSOLUTE_PATH}/../../../.env
set +a

#SBATCH --job-name=bootstrap_cex
#SBATCH --error=${error}errors_%x_%j.err
#SBATCH --output=${output}output_%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=${email}
#SBATCH --array=1-200

cd ${home}Programs/Preparation
ml python/3.10
python3 bootstrap_cex.py