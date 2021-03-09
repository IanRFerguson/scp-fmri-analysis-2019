#! /usr/bin/env python3.6

"""
Parses Stress-Buffering files only (stim files not onsets)
Matches to templates/stress_buffering_runwise_onsets file
"""

# ----- Imports + Setup
import os
import pandas as pd
from tqdm import tqdm

buff_template = pd.read_csv("templates/stress_buffering_runwise_onsets.csv")    # Read in template to match participant stim files


# ----- Format template
buff_template['run-block'] = buff_template.apply(lambda x: "{}-{}".format(x.run, x.block), axis=1)

# Long to wide
temp_onset = buff_template.pivot(index='run-block', columns='trial_type', values='onset').reset_index()
temp_duration = buff_template.pivot(index='run-block', columns='trial_type', values='duration').reset_index()

for var in temp_onset.columns:                                                  # Add onset tag to column names
    if var != "run-block":
        hold_please = var + "_onset"
        temp_onset.rename(columns={var:hold_please}, inplace=True)


for var in temp_duration.columns:                                               # Add duration tag to column names
    if var != "run-block":
        hold_please = var + "_duration"
        temp_duration.rename(columns={var:hold_please}, inplace=True)

template = temp_onset.merge(temp_duration, on="run-block", how="outer")         # Combine onset && duration dataframes


# ----- Helper functions
def parseOnset(DF):
    DF['run'] = [1,1,1,2,2,2,3,3,3]                                             # Convention...
    DF['block'] = [1,2,3,1,2,3,1,2,3]                                           # Run 1 Block 1, Run 1 Block 2, etc.
    DF['run-block'] = DF.apply(lambda x: "{}-{}".format(x.run, x.block), axis=1)
    DF.rename(columns={"condition":"block_type"}, inplace=True)

    keepers = [x for x in DF.columns if x not in ['cue', 'memory']]             # Drop cue and memory columns (not relevant to analysis)
    DF = DF[keepers]                                                            # Select these columns only
    return DF


print("\nMatching stress-buffering stim files to template...")

with open("Stress-Buffering-Error-Log.txt", "w") as log:
    for onset in tqdm(os.listdir("stress_buffering/")):
        if ".csv" in onset:
            try:
                part = pd.read_csv(os.path.join("stress_buffering/", onset))    # Read in file as Pandas DataFrame
            except Exception as e:
                log.write("\nError @ {} ... {}".format(onset, e))               # Write error to log if CSV can't be read
                continue

            if len(part) != 9:                                                  # 3 Runs w/ 3 Blocks == 3 * 3 == 9
                log.write("\n{} isn't the right size DF ({} not 9)".format(onset, len(part)))
                continue

            elif len(part) == 9:
                part = parseOnset(part)                                         # Clean up CSV w/ helper function defined above
                sub_name = onset.split('_')[0]                                  # Isolate subject number from filename
                output = "sub-{}_task-stress_run-{}_events.tsv"                 # BIDS-compliant file name

                # One TSV file per run
                for run in [1,2,3]:
                    reduced = part[part['run'] == run]
                    reduced_output = output.format(sub_name, run)
                    reduced.to_csv(os.path.join("stress_buffering/", reduced_output), sep="\t", index=False)

print("\nAll stress buffering event onsets generated...")
