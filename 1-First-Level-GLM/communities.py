#!/bin/python3

"""
SCP_Sub is a wrapper for first-level analysis in our project
This class allows us to run an analysis with a single function call

Ian Richard Ferguson | Stanford University
"""

"""
RUNNING TO-DO LIST

* For long network regressors ... mean inpute or 0's
* Adding network regressors to socialeval and stressbuffering task
"""

import warnings
warnings.filterwarnings('ignore')

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
      
      def __init__(self, subID, task, suppress=True, alternative_model=False):
            """
            At initialization, the following operations are performed:

            * Output directories are created under ./bids/derivatives/{sub}/{task}
            * task_information JSON fill is read in and values assigned to attributes 
            """

            Subject.__init__(self, subID, task, suppress=suppress)
            self.alt_model = alternative_model
            task_info = self._taskfile_validator()                      # Ensures that neccesary JSON files exist
            output = self._nipype_output_directories()                  # Creates output directories

            self.conditions = task_info['conditions']                   # In-scanner conditions
            self.tr = task_info['tr']                                   # Repetition time
            self.confound_regressors = task_info['confound_regressors'] # Regressors to include from fmriprep
            self.network_regressors = task_info['network_regressors']   # Regressors to include from SCP surveys
            self.block_regressors = task_info['block_regressors']       # Within-block regressors (for block-design)
            self.trial_type = task_info['trial_type']                   # Column in onsets file to split conditions on
            self.contrasts = task_info['design-contrasts']              # Weights to overwrite default 1 / -1

            # Output Directories
            self.nilearn_first_level_condition = output[0]
            self.nilearn_first_level_contrasts = output[1]
            self.nilearn_plotting_condition = output[2]
            self.nilearn_plotting_contrasts = output[3]

            if self.alt_model:
                  self.first_level_output = os.path.join(self.first_level_output, 'alt')


      # -------- FIRST-LEVEL GLM

      def _taskfile_validator(self):
            """
            Confirms the existence of the following files at the same directory level:
                  * subject_information.json
                  * task_information.json
            """

            import json
            target_files = ['./subject_information.json',
                            './task_information.json']

            for target in target_files:
                  if not os.path.exists(target):
                        raise OSError(f"{target} not found in current directory")

            with open('./task_information.json') as incoming:
                  info = json.load(incoming)                            # Read JSON as dictionary
            
            if self.alt_model:
                  return info[f"{self.task}_alt"]
            else:
                  return info[self.task]


      def _nipype_output_directories(self):
            """
            Runs @ __init__

            Dedicated output directories for Nipype and Nilearn output
            This creates new dirs AND returns list of relative paths
            """

            base_dir = self.first_level_output                          # Base dir in derivatives directory

            subdirs = [
                  # For Z-Maps (.nii.gz)        
                  'first-level-model/condition-maps',
                  'first-level-model/contrast-maps', 

                  # For brain maps (visualizations)
                  'plotting/condition-maps',
                  'plotting/contrast-maps']

            if self.alt_model:
                  subdirs = [f"alt/{x}" for x in subdirs]

            keepers = []                                                # Container to return

            for subdir in subdirs:
                  k = os.path.join(base_dir, subdir)                    # Relative path to new dir
                  keepers.append(k)                                     # Add to container
                  pathlib.Path(k).mkdir(parents=True, exist_ok=True)    # Make directory if it doesn't exist

            for subdir in ['glass', 'stat', 'summary']:
                  for dir in keepers[2:]:
                        k = os.path.join(dir, subdir)
                        pathlib.Path(k).mkdir(parents=True, exist_ok=True)

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


      def set_conditions(self, CONDITIONS):
            """
            CONDITIONS => List of condition regressors
            """

            self.conditions = CONDITIONS


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
            base_events = self.load_events(run=run)
            base_confounds = self.load_confounds(run=run)

            # Number of TRs in current run
            tr_total = len(base_confounds)

            # One row / TR in DataFrame format
            empty_events = pd.DataFrame({'onset':list(range(0, tr_total))})

            # Reduce to variables of interest
            voi = ['onset', 'duration'] + [self.trial_type] + [target_column]

            if "trial_type" in list(base_events.columns):
                  voi = voi + ['trial_type']

            base_events = base_events.loc[:, set(voi)]

            # Merge events with number of TRs
            events = empty_events.merge(base_events, on='onset', how='left')
            modulators = list(events.columns)[2:]

            for index, value in enumerate(events['duration']):
                  """
                  * This loop fixes issues with scan durations > 1. seconds
                  * If a task run has duration > 1s, 'target' and 'condition' rows are filled with the same values
                  """
                  if not np.isnan(value):
                        if float(value) > 1.:
                              end = value - 1.
                              
                              while end > 0:
                                    for var in modulators:
                                          events[var][index+end] = events[var][index]
                                    end -= 1

            if len(self.network_regressors) > 0:
                  with open('./subject_information.json') as incoming:
                        networks = json.load(incoming)
                        
                        def iso_value(x, var):
                              try:
                                    return float(networks[x][var])
                              except:
                                    return None
                              
                        for var in self.network_regressors:
                              # Derive network values from external JSON file
                              events[var] = events[target_column].apply(lambda x: iso_value(x, var))
                              events[var].fillna(0, inplace=True)


            def binarize_condition(x, condition):
                  if x == condition:
                        return 1
                  else:
                        return 0
                  
            for condition in list(self.conditions):
                  # Creates a binary-value column for every condition in task
                  events[condition] = events[self.trial_type].apply(lambda x: binarize_condition(x, condition))

            if len(self.block_regressors) > 0:
                  for condition in list(self.block_regressors):
                        events[condition] = events['trial_type'].apply(lambda x: binarize_condition(x, condition))

            return events


      # ------ DEFINE DESIGN MATRICES AND RUN GLM WITH NILEARN


      def firstLevel_event_design(self):
            """
            NOTE: This function is intended for event-based designs
            Creates design matrices per run using Nilearn

            Returns list of matrix objects
            """

            with open('./task_information.json') as incoming:
                  networks = json.load(incoming)[self.task]             # Read in task information as a dictionary


            def generate_matrices(run):
                  """
                  Helper function to loop through runs and generate 1:1 matrix:run

                  Returns single design matrix
                  """

                  if self.task != 'faces':

                        # Load in events file for the current run
                        events = self.load_events(run=run).loc[:,['onset','duration', self.trial_type]].reset_index(drop=True)
                        events.rename(columns={self.trial_type:'trial_type'}, inplace=True)

                        # Load in confound regressors for the current run
                        confound_regressor_names = list(networks['confound_regressors'])
                        confounds = self.load_confounds(run=run).loc[:, confound_regressor_names].reset_index(drop=False)

                        """
                        If you have network regressors defined they'll be added to the DM here
                        """

                        if (len(self.network_regressors) > 0):
                              long_network = self.longform_events(target_column='target', run=run).loc[:, self.network_regressors].reset_index()
                              confounds = confounds.merge(long_network, on='index')
                              confound_regressor_names += list(self.network_regressors)


                  elif self.task == 'faces':

                        events = self.load_events(run=run).loc[:, ['onset', 'duration', self.trial_type]].reset_index(drop=True)

                        def faces_v_ac(x):
                              if x in ['dorm', 'nondorm']:
                                    return 'face'
                              else:
                                    return 'attention'

                        events['trial_type'] = events[self.trial_type].apply(lambda x: faces_v_ac(x))
                        events.drop(columns=[self.trial_type], inplace=True)
                        
                        self.conditions = list(events['trial_type'].unique())

                        confound_regressor_names = list(networks['confound_regressors'])
                        confounds = self.load_confounds(run=run).loc[:, confound_regressor_names].reset_index(drop=False)

                        if len(self.network_regressors) > 0:
                              long_network = self.longform_events(target_column='target', run=run).reset_index()
                              long_network['face'] = long_network['condition'].apply(lambda x: faces_v_ac(x))

                              def dorm_nondorm(df):
                                    check = df[self.trial_type]

                                    if check == 'dorm':
                                          return 1
                                    return 0

                              long_network['dorm-membership'] = long_network.apply(dorm_nondorm, axis=1)

                              keepers = ['index', 'dorm-membership'] + self.network_regressors

                              long_network = long_network.loc[:, keepers]
                              confounds = confounds.merge(long_network, on='index')
                              confound_regressor_names += ['dorm-membership'] + self.network_regressors

                  
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
                                    try:
                                          # Impute null values with mean / var
                                          package.append(mean_impute[var])
                                    except Exception as e:
                                          print(f"{self.subID}: {e} @ line 341\n")
                                          package.append(temp)
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
                  output_path = os.path.join(self.first_level_output, 'plotting', file_name)
                  
                  # Create and save design matrix
                  nip.plot_design_matrix(matrix, output_file=output_path)

                  # Add DM to list
                  design_matrices.append(matrix)

            return design_matrices


      def firstLevel_block_design(self):
            """
            NOTE: This function is intended for block-designs
            Creates a design matrix per functional run

            Returns list of design matrices
            """

            with open('./task_information.json') as incoming:
                  networks = json.load(incoming)[self.task]

            def generate_matrices(run):
                  events = self.load_events(run=run)

                  events = events[events['trial_type'] != "fixation"].reset_index(drop=True)

                  def derive_block(DF):
                        return f"{DF['block_type']}_{DF['trial_type']}"

                  events['long'] = events.apply(derive_block, axis=1)
                  events = events.loc[:, ['onset','duration','long']].rename(columns={'long':'trial_type'})

                  self.set_conditions(list(events['trial_type']))

                  confound_regressor_names = list(networks['confound_regressors'])
                  confounds = self.load_confounds(run=run).loc[:, confound_regressor_names].reset_index(drop=False)

                  if len(self.network_regressors) > 0:
                        long_network = self.longform_events(target_column='target', run=run).loc[:, self.network_regressors].reset_index()
                        confounds = confounds.merge(long_network, on='index')
                        confound_regressor_names += list(self.network_regressors)

                  # Attributes for the DM
                  n_scans = len(confounds)
                  tr = self.tr
                  frame_times = np.arange(n_scans) * tr
                  hrf_model = 'spm'

                  """
                  Nilearn doesn't accept NA's in confound regressors
                  We'll mean impute any missing data here
                  """

                  # Empty dictionary to hold key:value pairs
                  mean_impute = {}

                  for var in ['framewise_displacement', 'dvars']:
                        # We'll accept FD and DVARS missing vals only
                        mean_impute[var] = np.mean(confounds[var])

                  """
                  We want a list of *n* values per row (1 value per additional regressor)
                  Now we'll loop through confounds DataFrame and adding all values per row to a list
                  Then that list will be appended to the master list
                  """

                  motion = []                                           # Parent list

                  for ix, index in enumerate(confounds['index']):
                        # Child list (1 list per row)
                        package = []

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

            # For an exclusive range
            functional_runs = len(self.preprocessed_bold_only) + 1
            # Empty list to append matrix objects into
            design_matrices = []

            for run in range(1, functional_runs):
                # Generate a design matrix per run
                matrix = generate_matrices(run=run)

                # Filename for DM output
                file_name = f"sub-{self.subID}_task-{self.task}_run-{run}_design-matrix.png"

                # Relative path for DM output
                output_path = os.path.join(self.first_level_output, 'plotting', file_name)

                # Create and save design matrix
                nip.plot_design_matrix(matrix, output_file=output_path)

                # Append DM to list
                design_matrices.append(matrix)

            return design_matrices


      def _default_contrasts(self):
            """
            If no contrasts are pre-defined for a task we'll compare all of them implicitly

            Returns dictionary of contrasts
            """

            contrasts = {}

            for outer in self.conditions:
                  for inner in self.conditions:
                        if outer != 'attention_check':
                              if inner != 'attention_check':
                                    if outer != inner:
                                          k = f"{outer}-{inner}"
                                          contrasts[k] = k

            return contrasts


      def _run_contrast(self, glm, contrast, title, output_type):
            """
            glm => FirstLevelModel object
            contrast => specific condition or contrast equation (e.g., high_trust - low_trust)
            title => key from contrast dictionary
            output_type => "condition" or "contrast"

            Performs the following actions:
                  * Computes contrast on GLM
                  * Defines relative output file paths
                  * Plots glass brain, stat map, and summary HTML
                  * Saves NifTi
            """
            
            contrast = str(contrast).replace(' ', '').strip()


            # Relative paths for brain map visualizations
            # Different sub-dirs for condition / contrast maps
            if output_type == "condition":
                  v_base = self.nilearn_plotting_condition
                  n_base = self.nilearn_first_level_condition
            
            elif output_type == "contrast":
                  v_base = self.nilearn_plotting_contrasts
                  n_base = self.nilearn_first_level_contrasts

            glass_output = f"{v_base}/glass/sub-{self.subID}_{output_type}-{title}_plot-glass-brain.png"
            stat_output = f"{v_base}/stat/sub-{self.subID}_{output_type}-{title}_plot-stat-map.png"
            report_output = f"{v_base}/summary/sub-{self.subID}_{output_type}-{title}_summary.html"

            nifti_output = f"{n_base}/sub-{self.subID}_{output_type}-{title}_z-map.nii.gz"

            # Compute the contrast itself
            z_map = glm.compute_contrast(contrast)

            # Plot and save brain map visualizations
            nip.plot_glass_brain(z_map, colorbar=False, threshold=2.3,
                                 plot_abs=False, display_mode='lyrz',
                                 title=title, output_file=glass_output)

            nip.plot_stat_map(z_map, threshold=2.3, colorbar=False,
                              draw_cross=False, display_mode='ortho',
                              title=title, output_file=stat_output)

            make_glm_report(model=glm,
                            contrasts=contrast,
                            plot_type='glass').save_as_html(report_output)

            # Save .nii.gz
            z_map.to_filename(os.path.join(nifti_output))
                  

      def firstLevel_contrasts(self, conditions=True):
            """
            Runs FirstLevelModel via Nilearn GLM package
            If conditions=False then only pairwise trial_type contrasts are computed
            """

            # Read in task-specific parameters from external JSON file
            with open('./task_information.json') as incoming:
                  if self.alt_model:
                        networks = json.load(incoming)[f"{self.task}_alt"]
                  else:
                        networks = json.load(incoming)[self.task]

            # If user defined contrasts don't exist, derive them
            if networks['design-contrasts'] == 'default':
                  contrasts = self._default_contrasts()
            else:
                  contrasts = networks['design-contrasts']

            # Event and Block designs have different helper functions
            if networks['design-type'] == 'event':
                  design_matrices = self.firstLevel_event_design()
            elif networks['design-type'] == 'block':
                  design_matrices = self.firstLevel_block_design()
                  

            # Scaffolding for model (HRF model and smoothing kernel)
            glm = first_level.FirstLevelModel(t_r=self.tr, smoothing_fwhm=4., hrf_model='spm')

            print("\n--------- Fitting model, please hold...")
            # Fit data to model
            model = glm.fit(self.preprocessed_bold_only, design_matrices=design_matrices)

            # Map baseline trial types if user desires
            if conditions:
                  print("\n--------- Mapping condition z-scores\n")
                  for contrast in tqdm(self.conditions):
                        self._run_contrast(glm=model, contrast=contrast, title=contrast, output_type="condition")

            # Contrasts will always be mapped
            print("\n--------- Mapping contrast z-scores\n")
            for k in tqdm(list(contrasts.keys())):
                  self._run_contrast(glm=model, contrast=contrasts[k], title=k, output_type="contrast")


      def run_first_level_glm(self, conditions=True):
            """
            If conditions if False, baseline z-maps are not calculated

            This helper is technically extraneous but it wraps everything nicely, so why not
            """

            self.firstLevel_contrasts(conditions=conditions)          
            print(f"\n\n{self.task.upper()} contrasts computed! subject-{self.subID} has been mapped")
