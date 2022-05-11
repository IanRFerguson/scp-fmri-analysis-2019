#!/bin/python3

"""
ABOUT THIS SCRIPT

First-pass using RSA to compare voxel pattern similarity
across memory and support target trials. Rough script that will
be updated iteratively.

Ian Richard Ferguson | Stanford University
"""


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import glob, os, pathlib, nibabel
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from nilearn.image import math_img
from nltools.data import Brain_Data
from nltools.mask import expand_mask
from bids import BIDSLayout


def easy_RSA(condition_list, roi, sub_id, output=True, smoothing="8mm", threshold_mask=False):
    """
    Parameters
        condition_list: List of functional tasks / conditions to include
        roi: Relative path to ROI NifTi file
        sub_id: Subject's identifier in BIDS project
    """

    # GLM Express output
    sub_dir = os.path.join("./bids/derivatives/first-level-ian",
                           f"sub-{sub_id}", "task-stressbuffer/first-level-model/condition-maps/")

    # Throw error if there's a typo or missing directory
    if not os.path.exists(sub_dir):
        raise OSError("Invalid directory path")

    # --- Isolate conditions of interest
    keepers = []

    # Loop through all first-level beta maps
    for file in glob.glob(os.path.join(sub_dir, "**/*.nii.gz"), recursive=True):

        # Loop through conditions of interest
        for t in condition_list:

            # Add beta map to list if it matches our conditions of interest
            if t in file and smoothing in file:
                keepers.append(file)

    # Easily isolate condition names (thanks BIDS!)
    conditions = [os.path.basename(x).split(
        'condition-')[1].split('_smoothing')[0] for x in keepers]

    # Convert maps to Brain_Data instance
    keepers_x = Brain_Data(keepers)

    # --- Set up mask
    mask = nibabel.load(roi)

    # Threshold map if needed
    if threshold_mask:
        mask = math_img(f"img >= 1.", img=mask)

    # Expand mask into multi-index array (if binarize_mask this will be length == 1)
    mask = expand_mask(Brain_Data(mask))

    # Loop through masks and apply correlation distance
    out = []

    for m in mask:
        out.append(keepers_x.apply_mask(m).distance(metric='correlation'))

    # Plot output for each matrix
    for k in range(len(out)):
        current_roi = mask[k]
        current_matrix = out[k]
        current_matrix.labels = conditions

        if output:
            output_directory = f"./easy_RSA/sub-{sub_id}"

            if not os.path.exists(output_directory):
                pathlib.Path(output_directory).mkdir(
                    exist_ok=True, parents=True)

            plt.figure(figsize=(10, 10))
            current_matrix.plot(vmin=0, vmax=2, cmap="RdBu_r")
            plt.savefig(os.path.join(output_directory,
                        f"sub-{sub_id}_matrix-{k}.png"))
            plt.close()

    return out


def build_distance_matrix(nl_data, output_dictionary):
    """
    Parameters
        nl_data: Adjacency instance from above function
    """

    labels = nl_data[0].labels

    temp = pd.DataFrame(nl_data[0].squareform(),
                        index=labels,
                        columns=labels)

    for k in labels:
        for m in labels:
            if k != m:
                key = f"{k} x {m}"

                value = temp[k][m]

                try:
                    output_dictionary[key].append(value)
                except:
                    output_dictionary[key] = [value]


targets = ["condition-memory",
           "condition-self_perspective",
           "condition-high_trust_perspective",
           "condition-low_trust_perspective"]

distance_dictionary = {}

for sub in tqdm(BIDSLayout("./bids").get_subjects()):
    
    try:
        sub_output=easy_RSA(condition_list=targets,
                            roi="../ROIs/Saxe_TOM_all_combined_FINAL.nii",
                            sub_id=sub,
                            threshold_mask=True,
                            output=False)
        
    except Exception as e:
        print(f"sub-{sub}:\t\t{e}")
        
    try:
        build_distance_matrix(sub_output, distance_dictionary)
    
    except Exception as e:
        print(f"sub-{sub}:\t\t{e}")


def dedicated_test(a, b):

    t_a = distance_dictionary[a]
    t_b = distance_dictionary[b]

    test = ttest_rel(t_a, t_b)

    print(f"=== {a} vs. {b} ===\n\n")
    print(f"T-statistic:\t\t{test.statistic}\nP-Value:\t\t{test.pvalue}")


dedicated_test("memory x self_perspective", "memory x high_trust_perspective")
dedicated_test("memory x self_perspective", "memory x low_trust_perspective")
dedicated_test("memory x high_trust_perspective", "memory x low_trust_perspective")