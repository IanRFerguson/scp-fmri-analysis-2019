#!/bin/python3

"""
nicursor - A lightweight pointer for neuroimaging research

Ian Richard Ferguson | Stanford University
"""

# ---- Utility imports
import warnings
warnings.filterwarnings('ignore')

import os
import pathlib
import matplotlib.pyplot as plt
import json
import pandas as pd
from copy import Error


# ---- Neuroimports
import nilearn.plotting as nip
from bids.layout import BIDSLayout

# Project root
root = './bids'

try:
     layout = BIDSLayout(root)
except Exception as e:
     raise OSError("Deploy this script from the same level as your BIDS Project")

# Subject IDs and task labels derived from BIDS data
valid_subjects = layout.get_subjects()
valid_tasks = layout.get_tasks()


# Object Definition
class Subject:
      def __init__(self, subID, task, suppress=True):

            # ---- Gloabl
            if subID in valid_subjects:
                  self.subID = subID
            else:
                  raise ValueError(f'{subID} is invalid input. Valid options: {str(valid_subjects)}')

            if task in valid_tasks:
                 self.task = task
            else:
                 raise ValueError(f'{task} is an invalid input. Valid options: {str(valid_tasks)}')

            # Absolute path to output from first level GLM
            self.first_level_output = self._output_1L()

            # ---- Raw
            self.anat = self._pathToAnat()                              # Raw anatomical
            self.nifti = self._pathToNifti()                            # Raw functional
            self.events = self._pathToEvents()                          # TSV event onsets

            # ---- Derivatives
            self.dummy_scans = [2] * len(self.nifti)                    # Hard-coded but changeable
            self.confounds = self._pathToConfounds()                    # TSV confound timeseries
            self.preprocessed = self._pathToPreprocessed()              # Preprocessed NifTi Files (ALL)
            self.preprocessed_bold_only = self._BOLDOnly()              # Preprocessed BOLD files only (no masks)
            self.roi = self._setROI()                                   # Output name for FSL ExtractROI() function
            self.brain_mask = self._getBrainMask()                      # Preprocessed NifTi format brain map

            # ---- Analysis
            self.contrasts = []                                         # Will be a list of lists of contrasts

            if not suppress:
                  print(str(self))                                      # Print Subject info at __init__


      def __str__(self):
            """
            Prints a dictionary of this object's current attributes
            """

            container = {"Subject ID": self.subID,
                        "Task": self.task,
                        "# of Functional Runs": len(self.nifti),
                        "Functional Runs": self.nifti,
                        "Path to Output Directory": self.first_level_output}

            return json.dumps(container, indent=4)


      # -------- UTILITY FUNCTIONS

      def _isolateRun(self, X):
            """
            RegEx would be easier, but here we are

            Returns run-x from BIDS standard naming convention
            """

            target = X.split('_')                                        # String => List
            for node in target:
                 if 'run-' in node:
                        y = node.split('-')[1]                          # Separate 'run-' from string
                        return int(y)                                   # Return integer only

            return None                                                 # Empty return if no match


      def _output_1L(self):
            """
            Defines absolute path to output directory for first-level GLM
            """

            path = os.path.join(root, (f'derivatives/first-level/sub-{self.subID}/task-{self.task}'))

            if not os.path.exists(path):
                  pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            return path


      # -------- RAW BRAIN DATA

      def _all_functional(self):
            """
            List of all raw functional files (UNPROCESSED)
            """

            path = f'sub-{self.subID}/func/'
            return os.listdir(os.path.join(root, path))


      def _pathToNifti(self):
            """
            List of absolute paths to NifTi files for specified task
            """

            response = layout.get(task=self.task,
                                  subject=self.subID,
                                  extension='nii.gz')

            if len(response) > 0:
                 return [os.path.join(root, x.relpath) for x in response]
            else:
                raise Error(f'No {self.task} matches for {self.subID}')


      def _all_anat(self):
            """
            List of all raw anatomical files (UNPROCESSED)
            """

            path = f'sub-{self.subID}/anat/'
            return os.listdir(os.path.join(root, path))


      def _pathToAnat(self):
            """
            List of absolute paths to anatomical files
            """

            return [os.path.join(root, (f'sub-{self.subID}/anat/{x}')) for x in self._all_anat() if "nii.gz" in x]


      def _pathToEvents(self):
            """
            List of absolute paths to TSV files for specified task
            """

            response = layout.get(task=self.task,
                                  subject=self.subID,
                                  extension='tsv')

            if len(response) > 0:
                 return [os.path.join(root, x.relpath) for x in response]
            else:
                 raise Error(f'No {self.task} matches for {self.subID}')


      def load_events(self, trial_type='trial_type', run='ALL'):
            """
            trial_type => condition variable (e.g., 'friendly face', 'hostile face', etc)
            run => functional run to isolate

            * Standardizes 'trial_type' variable name
            * Reads in event onsets as Pandas DataFrame object
            * For tasks with multiple runs, this function concatenates events into one DF

            Returns Pandas DataFrame Object
            """

            # Case - One run per task
            if len(self.events) == 1:
                  output = pd.read_csv(self.events[0], sep='\t')        # Read in events as DataFrame object
                  output['run'] = [1] * len(output)
                  return output

            # Case - Multiple runs per task
            else:
                  keepers = []                                          # Container for run-wise DF objects
                  for iter in self.events:
                        temp_run = self._isolateRun(iter)               # Isolate trial run from naming convention
                        temp = pd.read_csv(iter, sep='\t')              # Read in events as DataFrame object
                        temp['run'] = temp_run                          # Assign run to column
                        temp['onset'] = pd.to_numeric(temp['onset'])    # Convert 'onset' 
                        
                        temp.sort_values(by='onset',
                                         ascending=True,
                                         inplace=True)

                        keepers.append(temp)                            # Add to container

                  output = pd.concat(keepers).reset_index(drop=True)    # Concatenate run-wise DF objects

                  # Move run variable to front of DataFrame column list
                  current = list(output.columns)
                  current.remove('run')
                  clean = ['run'] + current

                  if run == 'ALL':
                        return output[clean]
                  else:
                        temp = output[clean]
                        return temp[temp['run'] == run].reset_index(drop=True)


      def plot_anat(self, dim=-1.65, threshold=5.):
            """
            Uses Nilearn to plot anatomical scans
            """

            k = nip.plot_anat(self.anat[0], title=f"{self.subID}_T1W-scan",
                              threshold=threshold, dim=dim, draw_cross=False)

            plt.show()


      # -------- PREPROCESSED BRAIN DATA

      def set_dummyScans(self, LIST):
            """
            Updates # of dummy scans (default=2)
            """

            self.dummy_scans = LIST


      def _all_derivatives(self):
            """
            List of all preprocessed functional files
            """

            deriv = f'derivatives/fmriprep/sub-{self.subID}/func/'
            return os.listdir(os.path.join(root, deriv))


      def _pathToConfounds(self):
            """
            List of confounds derived from fmriprep preprocessing
            """

            deriv = self._all_derivatives()
            path = os.path.join(
                  root, f'derivatives/fmriprep/sub-{self.subID}/func/')
            return [os.path.join(path, file) for file in deriv if self.task in file if "confounds_timeseries.tsv" in file]


      def _pathToPreprocessed(self):
            """
            List of preprocessed NifTi files derived from fmriprep
            """

            deriv = self._all_derivatives()
            path = os.path.join(
                  root, f'derivatives/fmriprep/sub-{self.subID}/func/')
            return [os.path.join(path, file) for file in deriv if self.task in file if ".nii.gz" in file]


      def _BOLDOnly(self):
            """
            List of preprocessed BOLD runs only (i.e., no masks)
            """

            return [x for x in self.preprocessed if 'desc-preproc_bold' in x]


      def _getBrainMask(self):
            """
            List of preprocess brain masks derived from FMRIPREP
            """

            deriv = self._all_derivatives()
            path = os.path.join(
                  root, f'derivatives/fmriprep/sub-{self.subID}/func/')
            return [os.path.join(path, x) for x in deriv if self.task in x if 'brain_mask' in x if '.nii.gz' in x]


      def _setROI(self):
            filename = f'sub-{self.subID}_{self.task}_roi-file.nii.gz'
            return os.path.join(self.first_level_output, filename)


      def load_confounds(self, run='ALL'):
            """
            run => Functional run to isolate

            Reads in confound timeseries information derived from fmriprep

            Returns Pandas DataFrame object
            """

            # Case - One run per task
            if len(self.confounds) == 1:
                 temp = pd.read_csv(self.confounds[0], sep='\t')
                 temp['run'] = [1] * len(temp)
                 return temp

            # Case - Multiple runs per task
            else:
                  keepers = []                                          # Container for run-wise DF objects
                  for iter in self.confounds:
                        temp_run = self._isolateRun(iter)               # Isolate trial run
                        temp = pd.read_csv(iter, sep='\t')              # Read in confounds as DataFrame object
                        temp['run'] = temp_run                          # Assign run to column
                        keepers.append(temp)                            # Add to container

                  # Concatenates run-wise DF objects into one
                  output = pd.concat(keepers).sort_values(by='run').reset_index(drop=True)

                  # Move run variable to front of DataFrame column list
                  current = list(output.columns)
                  current.remove('run')
                  clean = ['run'] + current

                  if run == 'ALL':
                        return output[clean]
                  else:
                        temp = output[clean]
                        return temp[temp['run'] == run].reset_index(drop=True)
