#!/bin/bash

# This script loops through all subject IDs and runs fmriprep in parallel
# Change the SBATCH arguments as you see fit
#
# Ian Richard Ferguson | Stanford University

project_directory="${SCRATCH}/SCP"
mkdir -p $project_directory && cd $project_directory

job_directory="${project_directory}/.job"
data_dir=/oak/stanford/groups/jzaki/Social_Networks/project_export/includes/

subjects=("sub-XXXXX" "sub-YYYYY" "sub-ZZZZZ" ... etc)

for sub in ${subjects[@]}; do

    job_file="${job_directory}/${sub}.job"

    echo "#!/bin/bash
#SBATCH --job-name=${sub}.job
#SBATCH --output=.out/${sub}.out
#SBATCH --error=.out/${sub}.err
#SBATCH --time=2-00:00
#SBATCH --mem=12000
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@stanford.edu
#SBATCH -c 8
#SBATCH -N 1
bash $HOME/scripts/fmriprep_singleSubj.sh $sub" > $job_file
    sbatch $job_file

done
