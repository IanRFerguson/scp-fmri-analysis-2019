#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=install_mriqc
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=$USER@stanford.edu

singularity build                                               \
  /oak/stanford/groups/jzaki/zaki_images/mriqc-0.15.1.simg      \
  docker://poldracklab/mriqc:0.15.1
