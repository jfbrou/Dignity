#!/bin/bash
#
#SBATCH --job-name=bootstrap_cps
#SBATCH --error=/home/jfbrou/Dignity_bash/errors_%x_%j.err
#SBATCH --output=/home/jfbrou/Dignity_bash/output_%x_%j.out
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jean-felix.brouillette@hec.ca
#SBATCH --array=1-200

cd /home/jfbrou/Dignity/Programs/Preparation
ml python/3.10
python3 bootstrap_cps.py
