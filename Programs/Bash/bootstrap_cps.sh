#!/bin/bash
#
#SBATCH -p hns
#SBATCH --job-name=bootstrap_cps
#SBATCH --error=/home/users/jfbrou/Dignity/Programs/Data programs/Error files/errors_%x_%j.err
#SBATCH --output=/home/users/jfbrou/Dignity/Programs/Data programs/Output files/output_%x_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --array=1-200

cd /home/users/jfbrou/Dignity/Programs/Data programs
ml python/3.10.0
python3 bootstrap_cps.py
