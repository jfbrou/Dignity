#!/bin/bash
#
#SBATCH -p hns
#SBATCH --job-name=bootstrap_cex_cps_combined
#SBATCH --error=/home/jfbrou/Dignity/Programs/Data programs/Error files/errors_%x_%j.err
#SBATCH --output=/home/jfbrou/Dignity/Programs/Data programs/Output files/output_%x_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

cd /home/jfbrou/Dignity/Programs/Preparation
ml python/3.10
python3 bootstrap_cex_cps_combined.py
