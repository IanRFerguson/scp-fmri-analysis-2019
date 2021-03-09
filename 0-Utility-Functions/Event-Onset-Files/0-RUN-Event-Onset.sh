#! /usr/bin/env bash

# --------- Goals of this script
#     * Execute Python script to convert CSVs to BIDS TSVs
#     * Create new subdirectories to hide old CSVs
#     * Move CSVs to new subdirectories

python3.6 1-CSV-to-TSV.py
sleep 1

python3.6 2-Stress-Buffering.py
sleep 1

#events=('faces_task/' 'social_evaluation/' 'stress_buffering/')
events=('faces_task/' 'social_evaluation/' 'stress_buffering/')

for dir in ${events[@]}; do
  csv_dir=`echo "${dir[@]}"/00_CSVs`

  [[ -d $csv_dir ]] || mkdir $csv_dir
  mv  ${dir[@]}/*.csv $csv_dir/
done
