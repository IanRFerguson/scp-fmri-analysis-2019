#!/bin/bash

# Produces automated QA reports after fmriprep pre-processing
# Modified script from Freeman Lab (New York University)
# IRF | Stanford University

# -------- Directory Variables
SCP="/oak/stanford/groups/jzaki/Social_Networks/project_export/includes/"
IMAGE="/oak/stanford/groups/jzaki/zaki_images/mriqc-0.15.1.simg"
OUTPUT="${SCP}/derivatives/mriqc/sub-${SUBJ}"
FLOATER="/scratch/users/irf823/SCP/work"

mkdir -p $OUTPUT

# -------- Script
singularity exec --cleanenv $IMAGE              \
    mriqc $SCP $OUTPUT                          \
    participant                                 \
    -w $FLOATER
