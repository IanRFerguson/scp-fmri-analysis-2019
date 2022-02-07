#!/bin/python3

"""
SCP_Sub is a wrapper for first-level analysis in our project
This class allows us to run an analysis with a single function call

Ian Richard Ferguson | Stanford University
"""

"""
TODO:

      * Collapse high and low (e.g., self vs. other)
      * Try intensity instead of valence (rating AND delta)
      * Contrasting with active baseline
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, json, pathlib
from tqdm import tqdm

from nicursor import Subject
from nilearn.glm import first_level
from nilearn.reporting import make_glm_report
import nilearn.plotting as nip


class SCP_Sub(Subject):
      
      def __init__(self, subID, task, user='ian', suppress=True, alternative_model=False, input_space="MNI152NLin6", trim=False):
            """
            At initialization, the following operations are performed:

            * Output directories are created under ./bids/derivatives/{sub}/{task}
            * task_information JSON fill is read in and values assigned to attributes 
            """

            Subject.__init__(self, subID, task, user, suppress=suppress, input_space=input_space)
            self.alt_model = alternative_model
            task_info = self._taskfile_validator()                      # Ensures that neccesary JSON files exist
            output = self._nipype_output_directories()                  # Creates output directories

            self.user = user
            self.container = self._first_level_container()

            self._quality_updating(trim=trim)

            self.conditions = task_info['conditions']                   # In-scanner conditions
            self.contrasts = task_info['design-contrasts']              # Weights to overwrite default 1 / -1
            self.tr = task_info['tr']                                   # Repetition time
            self.trial_type = task_info['trial_type']                   # Column in onsets file to split conditions on
            self.design_type = task_info['design-type']                 # Event or Block

            # Regressors for first-level design matrix
            self.confound_regressors = task_info['confound_regressors']
            self.network_regressors = task_info['network_regressors']   
            self.block_regressors = task_info['block_regressors']       
            
            # Output Directories
            self.nilearn_first_level_condition = output[0]
            self.nilearn_first_level_contrasts = output[1]
            self.nilearn_plotting_condition = output[2]
            self.nilearn_plotting_contrasts = output[3]

            if self.alt_model:
                  self.first_level_output = os.path.join(self.first_level_output, 'alt')

            if len(task_info['modulators']) > 0:
                  self._mean_center_modulators(task_info)


      # -------- Utility Functions

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
            Dedicated output directories for Nipype and Nilearn output
            This creates new dirs AND returns list of relative paths
            """

            # Base dir in derivatives directory
            base_dir = self.first_level_output                          

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


      def _first_level_container(self):
            """
            This is a useful dictionary to store run-wise and object-wise lists of aboslute paths
            """

            container = {'ordered-runs':[], 'ordered-events':[], 'ordered-confounds':[]}

            for index, brain in enumerate(self.preprocessed_bold_only):
                  
                  # E.g., index 0 == run-1, index 1 == run-2, etc.
                  run_check = f"run-{index+1}"           

                  # Standardize run-wise containers
                  container[run_check] = {'bold':'', 'event':'', 'confound':''}

                  """
                  
                  """

                  try:
                        current_bold = [x for x in self.preprocessed_bold_only if run_check in x][0]
                        container[run_check]['bold'] = current_bold
                        container['ordered-runs'].append(current_bold)
                  except:
                        print(f"No bold signal for {run_check}")

                  try:
                        current_event = [x for x in self.events if run_check in x][0]
                        container[run_check]['event'] = current_event
                        container['ordered-events'].append(current_event)
                  except:
                        print(f"No event file for {run_check}")

                  try:
                        current_confound = [x for x in self.confounds if run_check in x][0]
                        container[run_check]['confound'] = current_confound
                        container['ordered-confounds'].append(current_confound)
                  except:
                        print(f"No confound derivative for {run_check}")

            return container


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
            self._output_1L(self.user)
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

      
      def _isolate_run(self, X):
          return super()._isolate_run(X)


      def _quality_updating(self, trim):
            """
            trim => Boolean, determines if we'll drop runs or not

            This function performs the following operations

            * Read in QA JSON file
            * Drop any runs from excludes
            * Set proper # of dummy TR's
            """

            subject_key = f"sub-{self.subID}"

            with open('./quality-assurance.json') as incoming:
                  qa = json.load(incoming)

            # ---- Drop runs

            if trim:

                  try:
                        excludes = qa['excludes'][subject_key]

                        for run_to_exclude in excludes:
                              for key in list(self.container.keys()):
                                    if key == run_to_exclude:
                                          self.container.pop(key, None)
                                          continue

                                    for item in self.container[key]:
                                          if run_to_exclude in item:
                                                self.container[key].remove(item)

                  except KeyError as e:
                        print(f"{subject_key} not in excludes ... {e}")


            # ---- Update dummy scans

            try:
                  includes = qa['includes'][subject_key]
                  self.dummy_scans = includes['dummy']

                  for run, dummy in zip(includes['run'], includes['dummy']):
                        self.container[f"run-{run}"]['dummy'] = dummy

            except Exception as e:
                  print(f"sub-{self.subID} ... {e}")


      def _mean_center_modulators(self, task_file):
            """
            
            """

            for novel_modulator in task_file['modulators']:
                  if novel_modulator == 'intensity_delta':
                        events = self.load_events(run='ALL')
                        events['intensity_delta'] = events['intensity_rating'] - events['initial_intensity']
                        self.container[f'{novel_modulator}_mean'] = np.mean(events['intensity_delta'])

                  else:
                        iso_column = self.load_events(run='ALL').loc[:, novel_modulator]
                        self.container[f'{novel_modulator}_mean'] = np.mean(iso_column)


      # -------- Modeling Functions


      def generate_matrices(self, run_value):
            """
            index => Value to pull data from self.container() dictionary

            Helper function to loop through runs and generate 1:1 matrix:run

            Returns single design matrix
            """

            # ----- Helper functions

            def non_steady_state_aggregate(length, dummy_value):
                  """

                  """

                  dummy_value = int(dummy_value)
                  real_length = length - dummy_value

                  return [1] * dummy_value + [0] * real_length


            def block_regressors(DF):
                  """
                  Derives block-trial pair ... e.g., high_trust_perspective
                  """
                  
                  block = DF['block_type']
                  trial = DF['trial_type']

                  if trial == 'memory':
                        return trial
                  elif trial == "spatial":
                        return trial

                  """if block in ['high_trust', 'low_trust']:
                        return f"other_{DF['trial_type']}"""

                  return f"{DF['block_type']}_{DF['trial_type']}"


            def create_temp_dm(events, mod_name, frame_times):
                  """
                  events => 
                  mod_name => Modulator value, should be column in your events file
                  """

                  if mod_name == 'intensity_delta':
                        events['intensity_delta'] = events['intensity_rating'] - events['initial_intensity']

                  temp = events.rename(columns={mod_name: 'modulation'})
                  temp['modulation'] = temp['modulation'] - self.container[f'{mod_name}_mean']
                  temp['trial_type'] = temp['trial_type'].apply(lambda x: f'{x}_x_{mod_name}')

                  if mod_name == 'intensity_rating':

                      output_cols = ['memory_x_intensity_rating']

                  elif mod_name == 'intensity_delta':

                        """output_cols = ['self_perspective_x_intensity_delta',
                                      'other_perspective_x_intensity_delta']"""

                        output_cols = ['self_perspective_x_intensity_delta',
                                       'high_trust_perspective_x_intensity_delta',
                                       'low_trust_perspective_x_intensity_delta']

                  else:
                        output_cols = list(temp['trial_type'].unique())

                  dm = first_level.make_first_level_design_matrix(frame_times, temp, hrf_model='spm')
                  
                  return dm.loc[:, output_cols].reset_index()


            def reorder_dm_columns(dm):
                  dm.drop(columns=['index'], inplace=True)

                  tail = [x for x in dm.columns if 'drift' in x] + ['constant']
                  head = [x for x in dm.columns if x not in tail]

                  clean = head + tail

                  return dm.loc[:, clean] 


            with open('./task_information.json') as incoming:
                  networks = json.load(incoming)[self.task]             # Read in task information as a dictionary

            iso_container = self.container[f"run-{run_value}"]

            voi = ['onset', 'duration', 'target', 'trial_type']
            is_block_design = False
            has_modulator = False
            
            if self.design_type == 'block':
                  is_block_design = True
                  voi += ['block_type']

            if len(networks['modulators']) > 0:
                  has_modulator = True

                  if 'intensity_delta' in networks['modulators']:
                        mod_names = networks['modulators'].copy()
                        mod_names.remove('intensity_delta')
                        mod_names += ['initial_intensity', 'intensity_rating']
                  else:
                        mod_names = networks['modulators']
                  
                  voi += set(mod_names)

            
            events = pd.read_csv(iso_container['event'], sep='\t').loc[:, voi]
            confounds = pd.read_csv(iso_container['confound'], sep='\t')

            n_scans = len(confounds)
            tr = self.tr
            frame_times = np.arange(n_scans) * tr

            # ----- Events
            events = events[events['trial_type'] != 'fixation'].reset_index(drop=True)

            if is_block_design:
                  events['trial_type'] = events.apply(block_regressors, axis=1)
                  events.drop(columns=['block_type'], inplace=True)    

            if has_modulator:

                  mod_dms = []

                  for novel_modulator in networks['modulators']:
                        mod_dms.append(create_temp_dm(events, novel_modulator, frame_times))


            # NOTE: Conditions are reset here for first-level model
            self.set_conditions(set(events['trial_type'].unique()))

            events = first_level.make_first_level_design_matrix(frame_times, events, hrf_model='spm').reset_index()

            if has_modulator:

                  for dm in mod_dms:
                        events = events.merge(dm, on='index', how='left')

            # ----- Confounds
            try:
                  dummy_value = iso_container['dummy']
            except:
                  dummy_value = 2      
            
            #
            confounds['non_steady_state'] = non_steady_state_aggregate(length=len(confounds), dummy_value=dummy_value) 

            motion_outliers = [x for x in list(confounds.columns) if 'motion_outlier' in x]
            keepers = self.confound_regressors + motion_outliers

            #
            confounds = confounds.loc[:, keepers].reset_index(drop=False)

            # ----- Aggregate

            events = events.merge(confounds, on='index', how='left')

            """
            Nilearn doesn't accept NA's in confound regressors
            We'll mean impute any missing datat here
            """

            for var in ['framewise_displacement', 'dvars']:
                  events[var].fillna(np.mean(events[var]), inplace=True)

            
            """
            We want a list of *n* values per row (1 value per additional regressor)
            Now we'll loop through confounds DataFrame and adding all values per row to a list
            Then that list will be appended to the master list
            """

            motion = []                                                                   #

            for ix, val in enumerate(events['index']):
                  package = []                                                            #

                  for var in keepers:
                        package.append(events[var][ix])                                 #

                  motion.append(package)                                                  #

            events = reorder_dm_columns(events)

            return events

            """
            return first_level.make_first_level_design_matrix(frame_times,
                                                              events,
                                                              drift_model='polynomial',
                                                              drift_order=3,
                                                              add_regs=motion,
                                                              add_reg_names=keepers,
                                                              hrf_model='spm')            
            """



      def firstLevel_event_design(self):
            """
            NOTE: This function is intended for event-based designs
            Creates design matrices per run using Nilearn

            Returns list of matrix objects
            """

            design_matrices = []
            
            for scan in self.container['ordered-runs']:

                  run_value = self._isolate_run(scan)

                  # Generate a design matrix per run
                  matrix = self.generate_matrices(run_value)
                  
                  # Filename for DM output
                  file_name = f"sub-{self.subID}_task-{self.task}_run-{run_value}_design-matrix.png"
                  
                  # Relative path for DM output
                  output_path = os.path.join(self.first_level_output, 'plotting', file_name)
                  
                  # Create and save design matrix
                  nip.plot_design_matrix(matrix, output_file=output_path)

                  # Add DM to list
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


      def _run_contrast(self, glm, contrast, title, output_type, smoothing, plot_brains=False):
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
            kernel = str(int(smoothing))


            # Relative paths for brain map visualizations
            # Different sub-dirs for condition / contrast maps
            if output_type == "condition":
                  v_base = self.nilearn_plotting_condition
                  n_base = self.nilearn_first_level_condition
            
            elif output_type == "contrast":
                  v_base = self.nilearn_plotting_contrasts
                  n_base = self.nilearn_first_level_contrasts

            # Compute the contrast itself
            nifti_output = f"{n_base}/sub-{self.subID}_{output_type}-{title}_smoothing-{kernel}mm_z-map.nii.gz"
            z_map = glm.compute_contrast(contrast)

            # Save .nii.gz
            z_map.to_filename(os.path.join(nifti_output))

            if plot_brains:
                  glass_output = f"{v_base}/glass/sub-{self.subID}_{output_type}-{title}_smoothing-{kernel}mm_plot-glass-brain.png"
                  stat_output = f"{v_base}/stat/sub-{self.subID}_{output_type}-{title}_smoothing-{kernel}mm_plot-stat-map.png"
                  report_output = f"{v_base}/summary/sub-{self.subID}_{output_type}-{title}_smoothing-{kernel}mm_summary.html"

                  nifti_output = f"{n_base}/sub-{self.subID}_{output_type}-{title}_smoothing-{kernel}mm_z-map.nii.gz"

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


      def evaluate_model(self, model, contrast="default"):
            """
            model => FirstLevelModel object
            contrast => 

            This function performs the following operations
                  * Extract clusters
                  * Calculate and plot residuals
            """

            from nilearn.reporting import get_clusters_table
            from nilearn import input_data, image, masking
            import matplotlib.pyplot as plt

            test_img = image.concat_imgs(self.container['ordered-runs'])
            mean_img = image.mean_img(test_img)
            mask = masking.compute_epi_mask(mean_img)

            fmri_img = image.clean_img(test_img, standardize=False)
            fmri_img = image.smooth_img(fmri_img, fwhm=8.)

            if contrast == "default":
                  contrast = "+".join(self.conditions)

            z_map = model.compute_contrast(contrast)

            table = get_clusters_table(z_map, stat_threshold=3., cluster_threshold=20).set_index('Cluster ID', drop=True)
            coords = table.loc[range(1,6), ['X','Y','Z']].values

            masker = input_data.NiftiSpheresMasker(coords)
            real_timeseries = masker.fit_transform(fmri_img)
            predicted_timeseries = masker.fit_transform(fmri_img.predicted[0])

                  
      def firstLevel_contrasts(self, conditions=True, contrasts=True, smoothing=4., default_design=True, user_design=None, plot_brains=False):
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

            if default_design:
                  # Event and Block designs have different helper functions
                  design_matrices = self.firstLevel_event_design()

            else:
                  design_matrices = user_design

                  if isinstance(design_matrices, list):
                        if not isinstance(design_matrices[0], pd.DataFrame):
                              raise TypeError("User defined design matrices should be DataFrame object, or list of DataFrames")

                  elif not isinstance(design_matrices, pd.DataFrame):
                        raise TypeError("User defined design matrices should be DataFrame object, or list of DataFrames")
                  

            # Scaffolding for model (HRF model and smoothing kernel)
            glm = first_level.FirstLevelModel(t_r=self.tr, smoothing_fwhm=smoothing, hrf_model='spm', minimize_memory=False)

            print("\n--------- Fitting model")
            # Fit data to model
            model = glm.fit(self.container['ordered-runs'], design_matrices=design_matrices)


            if conditions:
                  print("\n--------- Mapping condition z-scores\n")
                  for contrast in tqdm(self.conditions):
                        self._run_contrast(glm=model, contrast=contrast, title=contrast, output_type="condition", smoothing=smoothing, plot_brains=plot_brains)


            if contrasts:
                  print("\n--------- Mapping contrast z-scores\n")
                  for k in tqdm(list(contrasts.keys())):
                        self._run_contrast(glm=model, contrast=contrasts[k], title=k, output_type="contrast", smoothing=smoothing, plot_brains=plot_brains)


      def run_first_level_glm(self, conditions=True, smoothing=4., plot_brains=False):
            """
            If conditions if False, baseline z-maps are not calculated

            This helper is technically extraneous but it wraps everything nicely, so why not
            """

            self.firstLevel_contrasts(conditions=conditions, smoothing=smoothing, plot_brains=plot_brains)          
            print(f"\n\n{self.task.upper()} contrasts computed! subject-{self.subID} has been mapped")
