#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=qc-group-level
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=irf823@stanford.edu
#SBATCH --output=group-level.out

# Produces automated QA reports after fmriprep pre-processing
# Modified script from Freeman Lab (New York University)
# IRF | Stanford University

# -------- Directory Variables
SCP="/oak/stanford/groups/jzaki/Social_Networks/project_export/includes"
IMAGE="/oak/stanford/groups/jzaki/zaki_images/mriqc-0.15.1.simg"
OUTPUT="${SCP}/derivatives/mriqc/GROUP"
FLOATER="/scratch/users/irf823/SCP/work"

mkdir -p $OUTPUT

# -------- Script
singularity exec --cleanenv $IMAGE              \
    mriqc $SCP $OUTPUT                          \
    group -w $FLOATER
