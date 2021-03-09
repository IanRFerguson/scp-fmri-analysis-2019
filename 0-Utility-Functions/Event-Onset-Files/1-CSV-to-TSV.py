#! usr/bin/env python3.6

"""
Parses through CSVs for two task conditions
Converts them to BIDS-compliant TSV files
"""

# ----- Imports + Setup
import os
import csv
import pandas as pd
from tqdm import tqdm

faces = os.listdir("faces_task/")                                               # List all files in each directory
soceval = os.listdir("social_evaluation/")

dirs = ["faces_task/", "social_evaluation/"]                                    # Task conditions
files = [faces, soceval]

all_files = []                                                                  # Empty list to append into
k = 0                                                                           # Iterator variable

for directory in dirs:
    for file in files[k]:
        if 'Icon' in file:                                                      # We want CSV files only
            continue                                                            # Would love to minimize this
        elif '.tsv' in file:                                                    # I'm only human :(
            continue
        else:
            all_files.append("{}{}".format(directory, file))                    # Add to list
    k += 1                                                                      # Iterate ++


# ----- Functions
def updateNamingConvention(IN, LOG):
    """
    Provides BIDS-compliant naming conventions to existing CSVs
    """

    directory = IN.split('/')[0]
    filename = IN.split('/')[1].split('_')

    if 'faces' in filename:
        try:
            output = directory + "/sub-{}_task-faces_run-1_events.tsv".format(filename[0])
        except Exception as e:
            LOG.write("Error @ {} ... {}\n".format(IN, e))

    elif 'eval' in filename:
        try:
            output = directory + "/sub-{}_task-socialeval_run-XYZ_events.tsv".format(filename[0])
        except Exception as e:
            LOG.write("Error @ {} ... {}\n".format(IN, e))

    elif 'memory' in filename:
        try:
            output = directory + "/sub-{}_task-stressbuffer_events.tsv".format(filename[0])
        except Exception as e:
            LOG.write("Error @ {} ... {}\n".format(IN, e))

    else:
        LOG.write("Error @ {} in else block...{}\n".format(IN))
        output = directory

    return output


def parseCSV(incoming, task, LOG):
    """
    * Faces task has one run only, it can be saved directly
    * Social evaluation has three runs...
        ** CSV needs to split into three TSVs w/ 100 rows each
        ** Naming convention for each file builds on helper function
    """

    if task != "social_eval":
        BIDS_name = updateNamingConvention(incoming, LOG)                       # Created BIDS compliant filename
        temp = pd.read_csv(incoming)                                            # Read CSV as Pandas DF
        temp.to_csv(BIDS_name, sep='\t', encoding='utf-8', index=False)         # Save unchanged file as TSV
    else:
        # Three runs == Three event onset files
        # ----- Run 1
        try:
            BIDS_name1 = updateNamingConvention(incoming, LOG)
            BIDS_name1 = BIDS_name1.replace('XYZ', '1')                         # run-XYZ -> run-1
            temp1 = pd.read_csv(incoming).iloc[:100]                            # Read in first 100 rows [0-99]
            temp1.to_csv(BIDS_name1, sep='\t', encoding='utf-8', index=False)
        except Exception as e:
            LOG.write("Error @ {} run-1...{}".format(incoming, e))

        # ----- Run 2
        try:
            BIDS_name2 = updateNamingConvention(incoming, LOG)
            BIDS_name2 = BIDS_name2.replace('XYZ', '2')                         # run-XYZ -> run-2
            temp2 = pd.read_csv(incoming).iloc[100:200]                         # Read in next 100 rows [100-199]
            temp2.to_csv(BIDS_name2, sep='\t', encoding='utf-8', index=False)
        except Exception as e:
            LOG.write("Error @ {} run-2...{}".format(incoming, e))

        # ----- Run 3
        try:
            BIDS_name3 = updateNamingConvention(incoming, LOG)
            BIDS_name3 = BIDS_name3.replace('XYZ', '3')                         # run-XYZ -> run-3
            temp3 = pd.read_csv(incoming).iloc[200:]                            # Read in next 100 rows [200-299]
            temp3.to_csv(BIDS_name3, sep='\t', encoding='utf-8', index=False)
        except Exception as e:
            LOG.write("Error @ {} run-3...{}".format(incoming, e))


def main():
    """
    Loops through all event CSVs and saves them to BIDS-formatted TSV files
    """

    log = open("Error-Log.txt", "w")                                            # Error log to keep track of exceptions

    for file in tqdm(all_files):
            try:
                if 'social_eval' not in file:
                    parseCSV(file, task="DeanBaltiTheHandsomeCowboy", LOG=log)  # Faces + Stress buffering tasks
                else:
                    parseCSV(file, task="social_eval", LOG=log)                 # Social evaluation task

            except Exception as e:
                log.write("Error @ {} ... {}\n".format(file, e))                # Write exception to error log
                continue

    log.close()
    print("All CSVs parsed ... see Error-Log.txt for details\n")

if __name__ == "__main__":
    print("Parsing faces && social-eval onsets...")
    main()
