#!/bin/bash
#SBATCH --job-name=first-level-stressbuffer.job
#SBATCH --time=8-00:00
#SBATCH --mem=12000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irf823@stanford.edu
#SBATCH -c 8
#SBATCH -N 1

ml python/3.6.1
source ./level1/bin/activate

python3 firstlevel.py ./includes/ 'stressbuffer'
