#! usr/bin/env python3.6

"""
Parses through CSVs for three task conditions
Converts them to BIDS-compliant TSV files
"""

# ----- Imports + Setup

import os
import csv
import pandas as pd
from tqdm import tqdm

faces = os.listdir("faces_task/")                                               # List all files in each directory
soceval = os.listdir("social_evaluation/")
stressbuff = os.listdir("stress_buffering/")

dirs = ["faces_task/", "social_evaluation/", "stress_buffering/"]               # Task conditions
files = [faces, soceval, stressbuff]

all_files = []                                                                  # Empty list to append into
k = 0                                                                           # Iterator variable

for directory in dirs:
    for file in files[k]:
        if 'Icon' or '.tsv' in file:                                            # We want CSV files only
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
            output = directory + "/sub-{}_task-faces_events.tsv".format(filename[0])
        except Exception as e:
            LOG.write("Error @ {} ... {}\n".format(IN, e))

    elif 'eval' in filename:
        try:
            output = directory + "/sub-{}_task-socialeval_events.tsv".format(filename[0])
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


def main():
    """
    Loops through all event CSVs and saves them to BIDS-formatted TSV files
    """

    log = open("Error-Log.txt", "w")                                            # Error log to keep track of things

    for file in tqdm(all_files):
            try:
                BIDS_name = updateNamingConvention(file, log)                   # Create BIDS compliant filename
                temp = pd.read_csv(file)                                        # Read in CSV to Pandas DF
                temp.to_csv(BIDS_name, sep = '\t',
                encoding = 'utf-8', index = False)                              # Save as TSV

            except Exception as e:
                log.write("Error @ {} ... {}\n".format(file, e))                # Write exception to error log
                continue
                
    log.close()
    print("All CSVs parsed ... see Error-Log.txt for details")

if __name__ == "__main__":
    print("Let's rock...")
    main()
