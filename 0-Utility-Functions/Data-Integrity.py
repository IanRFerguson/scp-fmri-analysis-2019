#! /usr/bin/env python3.6

"""
This is a hygiene check for subjects' task completeness
Output == a CSV where each row is a subject
"""

# ------- Imports + Setup
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

folks = [  ...  ]

"""
Faces == One run / sub
Stress == Three runs / sub
Social == Three runs / sub
"""

# ------- Ripper

faces = os.listdir("faces_task/")                                               # Isolate list of TSVs / task conditions
stress_buff = os.listdir("stress_buffering/")
social_eval = os.listdir("social_evaluation/")

output = {}                                                                     # Empty dict to append into

for sub in tqdm(folks):

    """
    This loop isolates subject onset files and their lengths
    These values are stored in dict temporarily - converted to DF after
    """

    output[sub] = {'faces': 0, 'faces-run1':0,                                  # Default values
                   'stress-buff': 0, 'sb-run1':0, 'sb-run2':0, 'sb-run3':0,
                   'social-eval': 0, 'se-run1':0, 'se-run2':0, 'se-run3':0}

    for file in faces:
        if sub in file:
            output[sub]['faces'] += 1                                           # File exists

            temp = pd.read_csv(os.path.join("faces_task/", file), sep="\t")
            output[sub]['faces-run1'] = len(temp)                               # No. of trials in task run
            continue

    for file in stress_buff:
        if sub in file:
            output[sub]['stress-buff'] += 1                                     # File exists

            temp = pd.read_csv(os.path.join("stress_buffering/", file), sep="\t")

            if "run-1" in file:
                output[sub]['sb-run1'] = len(temp)                              # No. of trials in task run (for 3 runs)
                continue
            elif "run-2" in file:
                output[sub]['sb-run2'] = len(temp)
                continue
            elif "run-3" in file:
                output[sub]['sb-run3'] = len(temp)
                continue


    for file in social_eval:
        if sub in file:
            output[sub]['social-eval'] += 1                                     # File exists

            temp = pd.read_csv(os.path.join("social_evaluation/", file), sep="\t")

            if "run-1" in file:
                output[sub]['se-run1'] = len(temp)                              # No. of trials in task run (for 3 runs)
                continue
            elif "run-2" in file:
                output[sub]['se-run2'] = len(temp)
                continue
            elif "run-3" in file:
                output[sub]['se-run3'] = len(temp)
                continue


output = pd.DataFrame.from_dict(output, orient='index').reset_index()           # Flip dictionary to Pandas DataFrame (long - wide)

output['Check'] = ['Incomplete'] * len(output)                                  # Two dummy vars to denote complete case / participant
output['Check - Description'] = [''] * len(output)

for idx, sub in enumerate(output['index']):
    fv = output['faces'][idx]                                                   # Check values for each task (1/3/3)
    sb = output['stress-buff'][idx]
    se = output['social-eval'][idx]

    if fv == 1 and sb == 3 and se == 3:
        output['Check'][idx] = 'Complete'                                       # Complete case!

    else:
        unpack = ""                                                             # Append missing task info into empty string

        if fv == 0:
            unpack += "No faces onset... "
        if sb != 3:
            unpack += "{} stress buff onsets... ".format(sb)
        if se != 3:
            unpack += "{} social eval onsets... ".format(se)

        output['Check - Description'][idx] = unpack                             # Assign missing info to row/column index

# ------- Kick to CSV

output.rename(columns={'index': 'subID'}, inplace=True)
output.to_csv("Subject-Quality-Check.csv", index=False)
