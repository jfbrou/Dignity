#!/bin/bash
#
#SBATCH -p hns
#SBATCH --job-name=bootstrap
#SBATCH --error=/home/users/jfbrou/Dignity_bash/errors_%x_%j.err
#SBATCH --output=/home/users/jfbrou/Dignity_bash/output_%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfbrou@stanford.edu

cd /home/users/jfbrou/Dignity/Programs
ml python
python3 bootstrap.py