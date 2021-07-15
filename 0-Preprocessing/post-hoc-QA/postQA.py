#!/bin/python3

"""
About this Script

* Used to visualize motion artifacts in functional scans
* Run after mriqc_groupLevel.sh && download group_bold.tsv file
* Outputs one PDF file with framewise displacment visualizations per subject

Ian Richard Ferguson | Stanford University
"""

# --------- Imports + Setup
import sys
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

here = os.getcwd()                                                                                      # Isolate current working dir
sns.set_style('whitegrid')                                                                              # Aesthetic defaults
sns.set_palette(sns.color_palette(["#2e4057", "#d1495b", "#edae49",
                                  "#00798c", "#66a182", "#8d96a3"]))

try:
    fileName = sys.argv[1]                                                                              # Read in TSV from command line
except:
    fileName = input("No file input!\n\nPlease point to group_bold.tsv\nRelative path:\t\t")

try:
    subs = pd.read_csv(os.path.join(here, fileName), sep="\t")                                          # Open TSV as DataFrame object
except:
    print("Ack! Sorry friend, {} is not a valid path".format(fileName))
    sys.exit(1)

for idx, var in zip([0,1,2], ['subID', 'task', 'run']):
    subs[var] = subs['bids_name'].apply(lambda x: x.split('_')[idx])                                    # Isolate file info from BIDS name

def taskRun(x):
    return "{}_{}".format(x['task'], x['run'])

subs['task-run'] = subs.apply(taskRun, axis=1)                                                          # Aggregate task and run (for plotting)
selects = ['bids_name', 'subID', 'task', 'run', 'task-run']                                             # Variables of interest

for var in subs.columns:
    if "fd_" in var:
        selects.append(var)

subs = subs[selects]                                                                                    # Reduce DataFrame

def generateSummaryPlot(METRIC):

    with PdfPages('./{}-summary.pdf'.format(METRIC)) as pdf_pages:                                      # Invoke PdfPages object

        if METRIC == "fd_num":
            y_limit = max(subs['fd_num']) * 1.2                                                         # y_limit == y-axis range
            summary = "aggregate"                                                                       # summary == FD metric
        elif METRIC == "fd_mean":
            y_limit = max(subs['fd_mean']) * 1.2
            summary = "average"
        else:
            y_limit = 100
            summary = "percent"


        for sub in tqdm(subs['subID'].unique()):
            temp = subs[subs['subID'] == sub].reset_index(drop=True)                                    # Reduce DF subwise

            plt.figure(figsize=(8,6))                                                                   # 8 x 6 plots / subject / metric
            sns.barplot(data=temp, x="task-run", y=METRIC, hue="task")
            plt.xlabel('')
            plt.xticks(rotation=45)
            plt.ylim(0, y_limit)
            plt.legend([],[], frameon=False)
            plt.title("{}_frame-displacement-{}-summary".format(sub,summary))
            pdf_pages.savefig(pad_inches=5)
            plt.close()

for key in ['fd_mean', 'fd_num', 'fd_perc']:
    print("\nRunning {} summary...".format(key))
    generateSummaryPlot(key)

print("\nAll plots saved!")
