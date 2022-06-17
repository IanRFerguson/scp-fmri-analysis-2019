#!/bin/python3

"""
About this Script

We want to get each subject's PINES expression at every
TR across all their functional runs.

This script does the following:
      * Reads in events and preprocessed BOLD for each subject for each run
      * Reduces events to high/low/self perspective
      * Calculates dot product for every TR

Ian Richard Fergsuon | Stanford University
"""

# --- Imports
from glm_express import Subject
import os, glob, pathlib, sys
from bids import BIDSLayout
import nibabel as nib
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Helpers
def get_longform_design(subject, run):
      """
      Creates long design matrix where every row is a
      TR collected by the scanner
      """

      # Columns to keep
      keys = ["onset", "duration", "trial_type", "block_type"]
      
      # Read in events and isolate keys
      event = subject.load_events(run=run).loc[:, keys]

      for ix, onset in enumerate(event["onset"]):

            # These match the given row value
            start_onset = onset
            start_duration = event["duration"][ix]
            block = event["block_type"][ix]
            trial = event["trial_type"][ix]

            # "Make long" e.g., add a row for every 1s of duration
            for k in range(start_duration):

                  start_onset += 1

                  # Create temp DataFrame (single row)
                  row_data = pd.DataFrame({
                        "onset": start_onset,
                        "duration": None,
                        "trial_type": trial,
                        "block_type": block
                  }, index=[0])

                  # Add to output DataFrame
                  event = event.append(row_data, ignore_index=True)

      # We only want memory and perspective trials
      event = event[event["trial_type"].isin(["memory", "perspective"])]

      # Drop duplicated onsets (this is happening in the for loop somewhere)
      event = event.sort_values(by="onset").drop_duplicates(subset=["onset"]).reset_index(drop=True)

      # We don't need duration columns moving forward
      event = event.drop(columns=["duration"])

      return event



def get_brain_data_vector(subject, run, relevant_onsets, pines_map):
      """
      For each TR, we'll vectorize the preprocessed data
      such that we return a list of 1D vectors for every
      TR in the given run
      """

      # Isolate relative path to preprocessed bold signal
      bold_run = subject.bids_container[f"run-{run}"]["func"]

      # Load NifTi image as numpy arrays
      image = nib.load(bold_run)

      # We want one np array for every TR ... hence, we transpose
      image_data = image.get_fdata().T

      # Flatten the PINES map prior to pattern expression
      flat_pines = pines_map.get_fdata().flatten()

      # Empty DF to append into
      output = pd.DataFrame()

      # Loop through memory and perspective onsets
      for tr in relevant_onsets:

            # Flatten NifTi map into 1D array
            flat_nifti = image_data[tr].flatten()
            
            # Calculate dot product (pattern expression)
            pines_expression = np.dot(flat_nifti, flat_pines)

            # Temporary DataFrame (single row)
            temp = pd.DataFrame({
                  "onset": tr,
                  "pines_expression": pines_expression
            }, index=[0])

            # Add to output DF
            output = output.append(temp, ignore_index=True)

      return output



def run_subject(sub_id, pines_map, bids_path="./bids/"):
      """
      Runs time series for a single subject
      """
      
      # Instantiate GLMX Subject
      subject = Subject(sub_id=sub_id, 
                        task="stressbuffer",
                        template_space="MNI152NLin6",
                        bids_root=bids_path,
                        suppress=True)

      # We want subject-specific output directories
      output_path = os.path.join(".",
                                 "pines_over_time",
                                 f"sub-{subject.sub_id}")

      pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)                        

      # Loop through functional runs
      for functional in range(1, subject.functional_runs+1):

            # Get reduced events file
            longform_design = get_longform_design(subject, run=functional)

            # List of onsets 
            relevant_onsets = list(longform_design["onset"])

            # Get dataframe of onsets / PINES pattern expressions
            vectorized_data = get_brain_data_vector(subject, 
                                                    run=functional,
                                                    relevant_onsets=relevant_onsets,
                                                    pines_map=pines_map)

            # Merge dataframes, keeping the onsets from the PINES PE dataframe
            composite = longform_design.merge(vectorized_data, on="onset", how="right").reset_index(drop=True)
            
            # Save locally
            output_name = os.path.join(output_path, 
                                       f"sub-{subject.sub_id}_run-{functional}_pines-over-time.csv")

            composite.to_csv(output_name, index=False)



def run():
      """
      This loops through every subject and calls the run_subject function
      """

      # User-provided rel path to BIDS
      bids_path = sys.argv[1]

      # Feed BIDS path to BIDSLayout object
      layout = BIDSLayout(bids_path)

      # Relative path to PINES weights
      pines_path = os.path.join(bids_path, 
                                "derivatives",
                                "masks",
                                "Rating_Weights_LOSO_2.nii")

      # Load PINES as NifTiImage
      pines_map = nib.load(pines_path)

      # Loop through subjects in BIDS project
      for sub_id in tqdm(layout.get_subjects()):

            try:
                  run_subject(sub_id=sub_id,
                                    pines_map=pines_map,
                                    bids_path=bids_path)

            except Exception as e:
                  print(f"{sub_id}:\t{e}")
                  


if __name__ == "__main__":
      run()