#!/bin/bash
#
#SBATCH -p hns
#SBATCH --job-name=bootstrap_cex_cps_combined
#SBATCH --error=/home/users/jfbrou/Dignity_bash/errors_%x_%j.err
#SBATCH --output=/home/users/jfbrou/Dignity_bash/output_%x_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jfbrou@stanford.edu

cd /home/users/jfbrou/Dignity/Programs
ml python/3.9.0
python3 bootstrap_cex_cps_combined.py
