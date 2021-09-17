#!/bin/python3

"""
SCP_Sub is a wrapper for first-level analysis in our project
This class allows us to run an analysis with a single function call

Ian Richard Ferguson | Stanford University
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import pathlib
from tqdm import tqdm

from nicursor import Subject
from nilearn.glm import first_level
from nilearn.reporting import make_glm_report
import nilearn.plotting as nip


class SCP_Sub(Subject):
      
      def __init__(self, subID, task):
            """
            At initialization, the following operations are performed:

            * Output directories are created under ./bids/derivatives
            * task_information JSON fill is read in and values assigned to attributes 
            """

            Subject.__init__(self, subID, task)
            task_info = self._taskfile_validator()                      # Ensures that neccesary JSON files exist
            output = self._nipype_output_directories()                  # Creates

            self.conditions = task_info['conditions']                   # In-scanner conditions
            self.tr = task_info['tr']                                   # Repetition time
            self.confound_regressors = task_info['confound_regressors'] # Regressors to include from fmriprep
            self.network_regressors = task_info['network_regressors']   # Regressors to include from SCP surveys
            self.block_regressors = task_info['block_regressors']       # Within-block regressors (for block-design)
            self.trial_type = task_info['trial_type']                   # Column in onsets file to split conditions on

            # Output Directories
            self.nilearn_first_level_condition = output[2]
            self.nilearn_first_level_contrasts = output[3]
            self.nilearn_plotting_condition = output[4]
            self.nilearn_plotting_contrasts = output[5]


      # -------- FIRST-LEVEL GLM

      def _taskfile_validator(self):
            """
            Confirms the existence of the following files at the same directory level:
                  * scp_subject_information.json
                  * scp_task_information.json
            """

            import json
            target_files = ['./scp_subject_information.json',
                            './scp_task_information.json']

            for target in target_files:
                  if not os.path.exists(target):
                        raise OSError(f"{target} not found in current directory")

            with open('./scp_task_information.json') as incoming:
                  info = json.load(incoming)                            # Read JSON as dictionary
                  reduced = info[self.task]                             # Reduce to task-specific information

                  return reduced


      def _nipype_output_directories(self):
            """
            Runs @ __init__

            Dedicated output directories for Nipype and Nilearn output
            This creates new dirs AND returns list of relative paths
            """

            base_dir = self.first_level_output                          # Base dir in derivatives directory
            runs = len(self.preprocessed_bold_only)+1                   # Number of functional runs

            subdirs = [
                  # For Z-Maps (.nii.gz)        
                  'nilearn/FirstLevelModel/condition',
                  'nilearn/FirstLevelModel/contrasts', 

                  # For brain maps (visualizations)
                  'nilearn/plotting/condition',
                  'nilearn/plotting/contrasts']

            keepers = []                                                # Container to return

            for subdir in subdirs:
                  k = os.path.join(base_dir, subdir)                    # Relative path to new dir
                  keepers.append(k)                                     # Add to container
                  pathlib.Path(k).mkdir(parents=True, exist_ok=True)    # Make directory if it doesn't exist

            for run in range(1, runs):
                  # Create sub-directories per run for given task
                  for subdir in ['nipype/FEATModel', 'nipype/Level1Design']:
                        temp = os.path.join(base_dir, subdir, f"run-{str(run)}")
                        pathlib.Path(temp).mkdir(parents=True, exist_ok=True)

            return keepers


      def _detonate(self):
            """
            Press reset ... for development only
            """

            import shutil
            shutil.rmtree(self.first_level_output)


      def _regen(self):
            """
            Press reset ... for development only
            """

            self._detonate()
            self._output_1L()
            self._nipype_output_directories()


      def set_contrasts(self, CONTRASTS):
            """
            CONSTRASTS => List of lists

            E..g, ['dorm_member', 'T', ['dorm_member'], [1]]
            """

            self.contrasts = CONTRASTS


      def derive_dummy_scans(self):
            """
            Returns list of binary 1,0 to denote number of dummy scans
            """

            events = self.load_events()                                 # Load event onset TSVs
            dummies = self.dummy_scans                                  # Hard-coded to 2

            if 'run' not in events.columns:                             # Single-run event onsets
                  size = len(events) - dummies                          # Number of non dummy scans
                  scans = ([1] * dummies) + ([0] * size)                # List of binary values

                  return scans

            else:                                                       # Multi-run event onsets
                  container = []                                        # To hold list of lists / run

                  for run in events['run'].unique():

                       # Isolate run + create dummy regressors per run
                        temp = events[events['run'] == run].reset_index(drop=True)
                        size = len(temp) - dummies
                        container.append(([1]*dummies) + ([0] * size))

                  # Concatenate lists into one
                  scans = []
                  for comp in container:
                        scans = scans + comp

                  return scans


      def longform_events(self, target_column, run):
            """
            target_column => Denotes in-scanner network stimulus (e.g., 'target')

            Perform the following operations:
                  * Match a DataFrame to the length of confound regressors
                  * Reduce events down to key columns and merge with full range of TRs
                  * Merge network regressors from external JSON file
                  * Binarize based on conditon

            Returns Pandas DataFrame object
            """

            # Base files (stored in BIDS directory)
            base_events = self.load_events()
            base_confounds = self.load_confounds()

            # Isolate DFs by current run
            base_events = base_events[base_events['run'] == run].reset_index(drop=True)
            base_confounds = base_confounds[base_confounds['run'] == run].reset_index(drop=True)

            # Number of TRs in current run
            tr_total = len(base_confounds)

            # One row / TR in DataFrame format
            empty_events = {'onset':list(range(0, tr_total))}
            empty_events = pd.DataFrame(empty_events)

            # Reduce to variables of interest
            voi = ['onset', 'duration'] + [self.trial_type] + [target_column]
            base_events = base_events.loc[:, voi]

            # Merge events with number of TRs
            events = empty_events.merge(base_events, on='onset', how='left')

            for idx, val in enumerate(events['duration']):
                  """
                  * This loop fixes issues with scan durations > 1. seconds
                  * If a task run has duration > 1s, 'target' and 'condition' rows are filled with the same values
                  """

                  if float(val) == 1.:
                        continue
                        
                  else:
                        end = val - 1
                        
                        while end > 0:
                              events[self.trial_type][idx+end] = events[self.trial_type][idx]
                              events['target'][idx+end] = events['target'][idx]
                              
                              end -= 1


            with open('./scp_subject_information.json') as incoming:
                  networks = json.load(incoming)
                  
                  def iso_value(x, var):
                        try:
                              return float(networks[x][var])
                        except:
                              return None
                        
                  for var in self.network_regressors:
                        # Derive network values from external JSON file
                        events[var] = events['target'].apply(lambda x: iso_value(x, var))


            def binarize_condition(x, condition):
                  if x == condition:
                        return 1
                  else:
                        return 0
                  
            for condition in list(self.conditions):
                  # Creates a binary-value column for every condition in task
                  events[condition] = events['condition'].apply(lambda x: binarize_condition(x, condition))

            for var in self.network_regressors:
                  # Fill network regressor columns with 0
                  events[var] = events[var].fillna(0)

            return events


      # ------ DEFINE DESIGN MATRICES AND RUN GLM WITH NILEARN


      def firstLevel_nilearn_design(self):
            """
            Creates design matrices per run using Nilearn

            Returns list of matrix objects
            """

            with open('./scp_task_information.json') as incoming:
                  networks = json.load(incoming)[self.task]             # Read in task information as a dictionary


            def generate_matrices(run):
                  """
                  Helper function to loop through runs and generate 1:1 matrix:run

                  Returns single design matrix
                  """

                  # Load in events file for the current run
                  events = self.load_events(run=run).loc[:,['onset','duration', self.trial_type]].reset_index(drop=True)
                  events.rename(columns={self.trial_type:'trial_type'}, inplace=True)

                  # Load in confound regressors for the current run
                  confound_regressor_names = list(networks['confound_regressors'])
                  confounds = self.load_confounds(run=run).loc[:, confound_regressor_names].reset_index(drop=False)

                  """
                  If you have network regressors defined they'll be added to the DM here
                  """

                  if len(self.network_regressors) > 0:
                        long_network = self.longform_events(target_column='target', run=run).loc[:, self.network_regressors].reset_index()
                        confounds = confounds.merge(long_network, on='index')
                        confound_regressor_names += list(self.network_regressors)

                  if len(self.block_regressors) > 0:
                        pass
                  
                  # Attributes for the DM
                  n_scans = len(confounds)
                  tr = self.tr
                  frame_times = np.arange(n_scans) * tr
                  hrf_model = 'spm'

                  """
                  Nilearn doesn't accept NA's in confound regressors
                  We'll mean impute any missing datat here
                  """

                  mean_impute = {}                                      # Empty dictionary to hold key:value pairs

                  for var in ['framewise_displacement', 'dvars']:
                        mean_impute[var] = np.mean(confounds[var])      # We'll accept FD and DVARS missing vals only

                  """
                  We want a list of *n* values per row (1 value per additional regressor)
                  Now we'll loop through confounds DataFrame and adding all values per row to a list
                  Then that list will be appended to the master list
                  """

                  motion = []                                           # Parent list

                  for ix, index in enumerate(confounds['index']):
                        package = []                                    # Child list (1 list per row)

                        for var in confound_regressor_names:
                              temp = confounds[var][ix]

                              if np.isnan(temp):
                                    # Impute null values with mean / var
                                    package.append(mean_impute[var])
                              else:
                                    package.append(temp)

                        motion.append(package)

                  return first_level.make_first_level_design_matrix(frame_times,
                                                                    events,
                                                                    drift_model='polynomial',
                                                                    drift_order=3,
                                                                    add_regs=motion,
                                                                    add_reg_names=confound_regressor_names,
                                                                    hrf_model=hrf_model)


            functional_runs = len(self.preprocessed_bold_only) + 1      # For an exclusive range   
            design_matrices = []                                        # Empty list to append matrix objects into
            
            for run in range(1, functional_runs):
                  # Generate a design matrix per run
                  matrix = generate_matrices(run=run)
                  
                  # Filename for DM output
                  file_name = f"sub-{self.subID}_task-{self.task}_run-{run}_design-matrix.png"
                  
                  # Relative path for DM output
                  output_path = os.path.join(self.first_level_output, 'nilearn/plotting', file_name)
                  
                  # Create and save design matrix
                  nip.plot_design_matrix(matrix, output_file=output_path)

                  # Add DM to list
                  design_matrices.append(matrix)

            return design_matrices


      def firstLevel_nilearn_contrasts(self):
            """
            Runs FirstLevelModel via Nilearn GLM package
            Automatically computes contrats for your conditions and 
            """

            from nilearn.glm import first_level
            from nilearn.reporting import make_glm_report
            import nilearn.plotting as nip

            # List of design matrices derived from helper function
            design_matrices = self.firstLevel_nilearn_design()

            # Scaffolding for model (HRF model and smoothing kernel)
            glm = first_level.FirstLevelModel(t_r=self.tr, smoothing_fwhm=4., hrf_model='spm')

            print("\n--------- Fitting model, please hold...")
            model = glm.fit(self.preprocessed_bold_only, design_matrices=design_matrices)
            contrasts_of_interest = list(self.conditions) + list(self.network_regressors)

            print("\n--------- Mapping condition z-maps\n")
            for contrast in tqdm(contrasts_of_interest):

                  # Relative paths for brain map visualizations
                  glass_output = f"{self.nilearn_plotting_condition}/sub-{self.subID}_condition-{contrast}_plot-glass-brain.png"
                  stat_output = f"{self.nilearn_plotting_condition}/sub-{self.subID}_condition-{contrast}_plot-stat-map.png"
                  report_output = f"{self.nilearn_plotting_condition}/sub-{self.subID}_condition-{contrast}_summary.html"
                  nifti_output = f"{self.nilearn_first_level_condition}/sub-{self.subID}_condition-{contrast}_z-map.nii.gz"

                  # Compute a condition-specific contrast relative to baseline
                  z_map = model.compute_contrast(contrast)

                  # Plot and save brain map visualizations
                  nip.plot_glass_brain(z_map, colorbar=False, threshold=3,
                                     plot_abs=False, display_mode='lyrz',
                                     title=contrast, output_file=glass_output)

                  nip.plot_stat_map(z_map, threshold=3, colorbar=False,
                                  draw_cross=False, display_mode='ortho',
                                  title=contrast, output_file=stat_output)

                  make_glm_report(model=model,
                                contrasts=contrast,
                                plot_type='glass').save_as_html(report_output)

                  # Save .nii.gz 
                  z_map.to_filename(os.path.join(nifti_output))

            print("\n--------- Mapping contrast z-maps\n")

            for outer in tqdm(self.conditions):
                  for inner in self.conditions:
                        if (inner == "attention_check" or outer == "attention_check"):
                              # No need to compare conditions to attention checks
                              continue

                        if outer != inner:
                              # E.g., nondorm - dorm
                              z_contrast = f"{outer} - {inner}"

                              # E.g., nondorm-dorm
                              contrast = z_contrast.replace(' ', '')

                              # Relative paths for brain map visualizations
                              glass_output = f"{self.nilearn_plotting_contrasts}/sub-{self.subID}_contrast-{contrast}_plot-glass-brain.png"
                              stat_output = f"{self.nilearn_plotting_contrasts}/sub-{self.subID}_contrast-{contrast}_plot-stat-map.png"
                              report_output = f"{self.nilearn_plotting_contrasts}/sub-{self.subID}_contrast-{contrast}_summary.html"
                              nifti_output = f"{self.nilearn_first_level_contrasts}/sub-{self.subID}_contrast-{contrast}_z-map.nii.gz"

                              # Compute two condition contrast
                              z_map = model.compute_contrast(z_contrast)

                              # Plot and save brain map visualizations
                              nip.plot_glass_brain(z_map, colorbar=False, threshold=3,
                                                   plot_abs=False, display_mode='lyrz',
                                                   title=z_contrast, output_file=glass_output)

                              nip.plot_stat_map(z_map, threshold=3, colorbar=False,
                                                draw_cross=False, display_mode='ortho',
                                                title=z_contrast, output_file=stat_output)

                              make_glm_report(model=model,
                                          contrasts=z_contrast,
                                          plot_type='glass').save_as_html(report_output)

                              # Save .nii.gz
                              z_map.to_filename(os.path.join(nifti_output))


      # ------ RUN GLM OF YOUR CHOOSING


      def run_first_level_glm(self):
            """
            Wraps everything defined above...

            Let's call it neurochristmas
            """

            self.firstLevel_nilearn_contrasts()          

            print(f"\n\n{self.task.upper()} contrasts computed! subject-{self.subID} has been mapped")
