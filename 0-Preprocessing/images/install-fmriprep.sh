#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=install_fmriprep
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=$USER@stanford.edu

singularity build                                                 \
  /oak/stanford/groups/jzaki/zaki_images/fmriprep-20.2.1.simg     \
  docker://nipreps/fmriprep:20.2.1
