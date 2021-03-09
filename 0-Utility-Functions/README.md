# Preprocessing Scripts

We preprocessed the data using <a href="https://fmriprep.org/en/stable/" target=_blank>**fmriprep**</a>. This directory contains scripts to convert various data files to be BIDS-compliant.

### Event TSVs
**00-Run-Event-Onset.sh** parses through all CSVs for each task condition, converts them to BIDS format, and moves the CSV to a specific sub-directory.
