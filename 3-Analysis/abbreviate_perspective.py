#!/bin/python3

"""
ABOUT THIS SCRIPT

This script updates our approach to modeling by abbreviating
the duration of perspective trials to maximize hemodynamic response

Ian Richard Ferguson | Stanford University
"""

# --- Imports
import warnings
warnings.filterwarnings('ignore')

from glm_express import Subject
from nilearn.glm import first_level
import numpy as np
import pandas as pd
import sys


# --- Helpers
def custom_design_matrix(subject, run, perspective_cutoff=None):
      """
      This function creates a run-specific design matrix to our
      specifications (namely, with abbreviated perspective trials)

      Parameters
            subject: GLM Express Subject object
            run: int
            perspective_cutoff: if not None, should be numeric

      Returns
            Pandas DataFrame object
      """

      # Load events and drop fixation trials
      events = subject.load_events(run=run)
      events = events[events['trial_type'] != 'fixation'].reset_index(drop=True)
      
      # Load confound regressors derived from fmriprep
      #confounds = subject.load_confounds(run=run).loc[:, subject.confound_regressors]
      confounds = [x for x in subject.confounds if 'confounds_timeseries' in x if f"run-{run}" in x][0]
      confounds = pd.read_csv(confounds, sep="\t").loc[:, subject.confound_regressors]
      
      # Merge block and trial identifiers
      def custom_block_pairing(DF):
            block_ = DF['block_type']
            trial_ = DF['trial_type']

            if trial_ not in ['memory', 'spatial']:
                  return f"{block_}_{trial_}"
            
            return trial_
      

      # Merge block and trial identifiers
      events['trial_type'] = events.apply(custom_block_pairing, axis=1)
      events = events.loc[:, ['onset', 'duration', 'trial_type']]


      # Cutoff perpsective trials to our specifications
      if perspective_cutoff is not None:
            for ix, val in enumerate(events['trial_type']):
                  if "perspective" in val:

                        # -- Update discard trial
                        start_onset = events['onset'][ix]
                        new_onset = start_onset + float(perspective_cutoff)

                        start_duration = events['duration'][ix]
                        new_duration = start_duration - float(perspective_cutoff)

                        # -- Update trial of interest
                        events['duration'][ix] = float(perspective_cutoff)

                        events = events.append({'onset': new_onset,
                                                'duration': new_duration,
                                                'trial_type': f"{val}_discard"}, ignore_index=True)

      events = events.sort_values(by='onset').reset_index(drop=True)
      
      # Parameters for design matrix
      n_scans = len(confounds)
      t_r = subject.t_r
      frame_times = np.arange(n_scans) * t_r
      
      # Create temporary design matrix
      temp_dm = first_level.make_first_level_design_matrix(frame_times,
                                                            events,
                                                            hrf_model='spm')
      
      # drift_x and constant
      nilearn_derivs = [x for x in list(temp_dm.columns) if "drift" in x] + ["constant"]
      
      # Condition names
      functionals = [x for x in list(temp_dm.columns) if x not in nilearn_derivs]
      
      # Combine events and confounds
      temp_dm = temp_dm.join(confounds, how='outer')
      
      # Reorder columns
      clean_columns = functionals + list(confounds.columns) + nilearn_derivs
      
      # mean impute FD and DVARS
      for con in ['framewise_displacement', 'dvars']:
            temp_dm[con].fillna(temp_dm[con].mean(), inplace=True)

      # Reassign conditions to subject attribute
      if run == 1:
            subject.conditions = list(events['trial_type'].unique())
            
      return temp_dm.loc[:, clean_columns]


def main():

      subject_id = sys.argv[1]
      bids_root = sys.argv[2]

      try:
            perspective_cutoff = sys.argv[3]
      except:
            perspective_cutoff = None

      sub = Subject(subject_id, 
                   "stressbuffer", 
                   bids_root, 
                   suppress=True,
                   template_space="MNI152NLin6")

      keepers = []

      for run in range(1, sub.functional_runs+1):
            keepers.append(custom_design_matrix(sub, run=run, perspective_cutoff=perspective_cutoff))

      sub.run_first_level_glm(user_design_matrices=keepers, verbose=True)


if __name__ == "__main__":
      main()