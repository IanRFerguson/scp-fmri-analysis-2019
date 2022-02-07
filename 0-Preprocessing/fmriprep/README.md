# fmriprep

These scripts set up your raw BIDS data for preprocessing with `fmriprep`

Run these in order...

* `derive_subjects.py`: creates a list of subjects to port into job script
* Add subjects from this list to `Job-Script.sh`
* `Job-Script.sh`: Loops through subject labels and deploys `fmriprep_singleSubject.sh` to SLURM manager