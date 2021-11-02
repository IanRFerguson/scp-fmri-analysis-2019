#!/bin/python3

"""
GroupLevel class facilitates group-level analyses with first-level brain data
Uncorrected and corrected brain maps are available to run with a single function call

Ian Richard Ferguson | Stanford University
"""

# ----- Imports
import warnings, os, json, pathlib
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.glm import second_level, threshold_stats_img
import nilearn.plotting as nip
from nilearn import image
from bids.layout import BIDSLayout

# ----- BIDS Directory Specifications
root = './bids'

try:
      layout = BIDSLayout(root)
      all_subjects = layout.get_subjects()
      all_tasks = layout.get_tasks()            
except:
      raise OSError("Check your BIDS root definition")


class GroupLevel:

      def __init__(self, task, complete_BIDS=False):
            """
            task => Corresponds to valid task in BIDS dataset
            complete_BIDS => Boolean, True if all raw data is available (on HPC)

            The following operations are performed:
                  * Output directories are created if they don't exist
                  * A list of all NifTi maps are assigned to object
            """

            self.task = task                                            # Valid BIDS task
            self._output_directory()                                    # Create relevant output dirs

            if complete_BIDS:
                self.subjects = self._iso_BIDS_subjects()
            else:
                  self.subjects = self._iso_subjects()

            self.all_brain_data = self._brain_data()                    # List of ALL NifTi's
            self.task_file = self._taskfile_validator()                 # Confirms presence of JSON info files
            
            self.group_regressors = self.task_file['group-level-regressors']

            # Paths to model and brain map output directories
            self.plotting_output = os.path.join(root, f"derivatives/second-level/task-{self.task}/plotting")
            self.nifti_output = os.path.join(root, f"derivatives/second-level/task-{self.task}/second-level-model")


      def _taskfile_validator(self):
            """
            Confirms the existence of the following files at the same directory level:
                  * scp_subject_information.json
                  * scp_task_information.json
            """

            target_files = ['./scp_subject_information.json',
                            './scp_task_information.json']

            for target in target_files:
                  if not os.path.exists(target):
                        raise OSError(f"{target} not found in current directory")

            with open('./scp_task_information.json') as incoming:
                  info = json.load(incoming)                            # Read JSON as dictionary
                  reduced = info[self.task]                             # Reduce to task-specific information

            return reduced


      def _output_directory(self):
            """
            Runs at __init__
            Creates second-level subdirectories in derivatives if they don't exist
            """

            target = os.path.join(root, f"derivatives/second-level/task-{self.task}")

            if not os.path.exists(target):
                  pathlib.Path(target).mkdir(exist_ok=True, parents=True)

            # Create subdirs for corrected and uncorrected models
            for subdir in ['second-level-model', 'plotting']:
                  for cor in ['corrected', 'uncorrected']:
                        temp = os.path.join(target, subdir, cor)

                        if not os.path.exists(temp):
                              pathlib.Path(temp).mkdir(exist_ok=True, parents=True)


      def _iso_BIDS_subjects(self):
            """
            Deployment function - Only works with complete BIDS dataset
            Returns a list of subjects with first-level maps
            """

            # Empty list to append into
            task_subjects = []                                          
            
            # Relative path to first-level maps
            deriv_level = os.path.join(root, "derivatives/first-level") 

            # Leverages BIDSLayout to isolate subject ID numbers
            for sub in all_subjects:
                  temp = os.path.join(deriv_level, f"sub-{sub}", f"task-{self.task}")
                  if os.path.isdir(temp):
                        task_subjects.append(f"sub-{sub}")

            return task_subjects


      def _iso_subjects(self):
            """
            Development function - Works with isolated derivatives
            Returns a list of subjects with first-level maps
            """

            # Empty list to append into
            task_subjects = []                                          
            
            # Relative path to first-level maps
            deriv_level = os.path.join(root, "derivatives/first-level") 
            
            # Complete list of subjects
            subject_level = [x for x in os.listdir(deriv_level) if "sub-" in x]

            for sub in subject_level:
                  # Arbitrarily uses condition-maps subdir to confirm presence of first-level models
                  temp = os.path.join(deriv_level, sub, f"task-{self.task}/first-level-model/condition-maps")

                  if os.path.isdir(temp):
                        if len(os.listdir(temp)) > 0:
                              task_subjects.append(sub)                 # If brain maps are present, add sub to list

            return task_subjects


      def _brain_data(self):
            """
            Returns a dictionary of subject:run key-value pairs
            """

            data = {k:[] for k in self.subjects}                        # Dictionary with empty lists for each subject
            base = os.path.join(root, "derivatives/first-level")        # Relative path to first-level subdirectory

            for sub in self.subjects:
                  # Define relative paths to first-level-models per subject
                  sub_directory = os.path.join(base, sub, f"task-{self.task}", "first-level-model")
                  conditions = os.path.join(sub_directory, "condition-maps")
                  contrasts = os.path.join(sub_directory, "contrast-maps")

                  output = []

                  for dir in [conditions, contrasts]:
                        temp = os.listdir(dir)
                        temp = [os.path.join(dir, x) for x in temp]
                        output += temp

                  data[sub] = output

            return data


      def get_brain_data(self, contrast):
            """
            contrast => Used to isolate relevant NifTi files for group-level analysis

            Returns a list of absolute paths to relevant NifTi files
            """

            output = []                                                 # Empty list to append into

            try:
                  for sub in self.subjects:                             # Checks list of NifTi's for each subject
                        for scan in self.all_brain_data[sub]:
                              if contrast in scan:
                                    output.append(scan)                 # Add NifTi to list if contrast matches

            except:
                  raise ValueError(f"Your specfied contrast {contrast} not found in functional runs")

            if len(output) == 0:
                  raise ValueError(f"Your specified contrast {contrast} not found in functional runs")
            
            return output


      # ------- Model Utilities


      def build_design_matrix(self):
            """
            Creates a design matrix with subject-wise regressors

            Work to do
                  * Should accomodate subject demographics
                  * Should accomodate indegree/outdegree/other network regressors

            Returns design matrix as Pandas DF
            """
            from nilearn.glm.second_level import make_second_level_design_matrix

            # Read in subject data from Networks survey
            with open('./scp_subject_information.json') as incoming:
                  scp_subjects = json.load(incoming)

            # Task specific information
            networks = self.task_file

            def change_key(key):
                  iso = str(key)[1:]
                  return f"sub-{iso}"

            # Changes key formatting from sXXXXX to sub-XXXXX
            subject_data = {change_key(k):v for k,v in scp_subjects.items()}

            # Helper function to kick out value for a specific variable / subject
            def network_value(x, var):
                  try:
                        return subject_data[x][var]
                  except:
                        return 'NA'

            subjects_label = self.subjects
            design_matrix = pd.DataFrame({'subject_label': subjects_label})

            for var in networks['group-level-regressors']:
                  # Bring in network variables
                  design_matrix[var] = design_matrix['subject_label'].apply(lambda x: network_value(x, var=var))
                  design_matrix[var] = pd.to_numeric(design_matrix[var])

            return make_second_level_design_matrix(subjects_label, design_matrix)


      def uncorrected_group_model(self, contrast, columns=[], smoothing=4., all_contrasts=True, model_output=False):
            """
            contrast => baseline condition or contrast from first-level-model
            columns => regressors of interest to include in design matrix (defaults to ALL)
            smoothing => kernel to smooth brain regions during model fitting
            all_contrasts => If True, models each design matrix column explicitly
            model_output => Determines if a SecondLevelModel object will be returned

            The following operations are performed:
                  * SecondLevelModel object is instantiated and fit with contrast brain data
                  * Uncorrected z_map is computed, visualized, and saved locally
                  * Unocrrected model is returned
            """

            brain_data = self.get_brain_data(contrast=contrast)         # List of relevant NifTi maps
            design_matrix = self.build_design_matrix()                  # Define design matrix with helper function

            if len(columns) > 0:
                  if "intercept" not in columns:
                        columns = columns + ['intercept']

                  design_matrix = design_matrix.loc[:, columns]         # Reduce design matrix if desired

            for path in [self.nifti_output, self.plotting_output]:
                  out = os.path.join(path, f"uncorrected/{contrast}")

                  if not os.path.isdir(out):
                        pathlib.Path(out).mkdir(exist_ok=True, parents=True)

            # Instantiate and fit a second-level model
            model = second_level.SecondLevelModel(smoothing_fwhm=smoothing).fit(brain_data, design_matrix=design_matrix)

            if all_contrasts:
                  for var in (design_matrix.columns):

                        # Compute basic contrast with intercept
                        z_map = model.compute_contrast(var, output_type="z_score")

                        var = var.lower()

                        # Define output file names
                        nifti_filename = os.path.join(self.nifti_output, f"uncorrected/{contrast}/second-level_uncorrected_contrast-{contrast}_regressor-{var}.nii.gz")
                        map_filename = os.path.join(self.plotting_output, f"uncorrected/{contrast}/second-level_uncorrected_contrast-{contrast}_regressor-{var}.png")

                        TITLE = f"uncorrected-{contrast}-{var}"

                        # Save NifTi and brain map
                        z_map.to_filename(nifti_filename)
                        nip.plot_glass_brain(z_map, threshold=2.3, plot_abs=False, 
                                          display_mode='lyrz', title=TITLE, output_file=map_filename)

            everything = "+".join(design_matrix.columns)
            z_map = model.compute_contrast(everything, output_type="z_score")

            nifti_filename = os.path.join(self.nifti_output, f"uncorrected/{contrast}/second-level_uncorrected_contrast-{contrast}_all-regressors.nii.gz")
            map_filename = os.path.join(self.plotting_output, f"uncorrected/{contrast}/second-level_uncorrected_contrast-{contrast}_all-regressors.png")

            z_map.to_filename(nifti_filename)
            nip.plot_glass_brain(z_map, threshold=2.3, plot_abs=False, 
                                 display_mode='lyrz', title=f"{contrast}_all-regressors", 
                                 output_file=map_filename)

            if model_output:
                  return model


      def _batch_uncorrected_model(self, contrasts, columns=[], all_contrasts=False):
            """
            contrasts => List of valid contrasts to model
            columns => Design matrix columns of interest, defaults to All
            all_contrasts => Determines if all design matrix columns are modeled or not

            Loops through user input contrasts and models contrast of interest
            """

            for test in contrasts:
                  self.uncorrected_group_model(contrast=test, columns=columns, all_contrasts=all_contrasts)


      def _basic_model(self, contrast):
            """
            Mostly a developmental function...
            Runs a simple, intercept-only model on the specified contrast
            """

            # List of absolute paths to NifTi files for the specified contrast
            brain_data = self.get_brain_data(contrast=contrast)

            # Intercept-only design matrix
            dm = pd.DataFrame([1] * len(brain_data))

            # Instantiate + Fit second level mode
            model = second_level.SecondLevelModel(smoothing_fwhm=4.).fit(brain_data, design_matrix=dm)
            
            # Compute contrast on intercept column
            z_map = model.compute_contrast(output_type="z_score")
            return z_map


      def contrast_QA(self, contrast, verbose=False):
            """
            contrast => Valid first-level contrast

            Returns DataFrame object specifying quantity of contrasts and NifTi shapes for each contrast
            """

            data = self.get_brain_data(contrast=contrast)               # List of relative paths to contrast maps

            output = {'sub_id':[],                                      # Emtpy dictionary to append into
                      'path_to_nifti':[],
                      'shape':[]}

            # For all valid contrasts, add to dictionary object
            for file in tqdm(data):                           
                  output['sub_id'].append(file.split('/')[-1].split('_')[0])
                  output['path_to_nifti'].append(file)
                  output['shape'].append(image.load_img(file).shape)

            # Print message for end user + return Pandas DataFrame object
            print(f"\nYou have {len(output['sub_id'])} valid contrasts eligible for second-level modeling...\n")

            if verbose:
                  return pd.DataFrame(output).sort_values(by="sub_id").reset_index(drop=True)



      # ------- Multiple Comparisons and Analysis Tools


      def one_sample_test(self, nifti, contrast, height_control="fpr", alpha=0.001, cluster_threshold=0, return_map=False):
            """
            nifti => z_map contrast computed from a SecondLevelModel object
            contrast => String used for title on plot
            height_control => False positive rate (FPR / FDR / Bonferroni)
            alpha => Significance level
            cluster_threshold => Groups of connected voxels
            return_map => Kicks out significant voxels if you want to assign to a variable
            """

            # Threshold map at the specified significance level
            temp_map, temp_thresh = threshold_stats_img(nifti, alpha=alpha, 
                                                        height_control=height_control,
                                                        cluster_threshold=cluster_threshold)

            # Plot output
            title = f"{contrast} @ {alpha}"
            nip.plot_glass_brain(temp_map, threshold=temp_thresh, display_mode='lyrz',
                                plot_abs=False, colorbar=False, title=title)

            if return_map:
                  return temp_map


      def load_PINES_signature(self):
            """
            Helper function to instantiate NifTi object with negative emotion signature
            This should be saved under derivatives/masks/ ... can be found on Tor Wager's GitHub page if need be

            Returns NifTi image of PINES signature for multivariate analysis and masking
            """

            path = os.path.join(root, "derivatives/masks/Rating_Weights_LOSO_2.nii")
            
            try:
                  return image.load_img(path)
            except:
                  raise OSError("No PINES map detected! Make sure it's located under ./$bids_root/derivatives/masks/")


      def resample_to_MNI152(self, target_img):
            """
            target_img => NifTi image that we want to resample

            Helper function to convert target image dimensionality to MNI152 template
            """

            from nilearn.datasets import load_mni152_template  
            template = load_mni152_template()                           

            return image.resample_to_img(target_img, template)


      def estimate_PINES_signature(self, contrast):
            """
            contrast => Contrast of interest that we want to compare to PINES signature

            Vectorizes brain data from a given task and calculates dot product with PINES signature vector
            """

            brain_data = self.get_brain_data(contrast=contrast)         # List of relative paths to NifTi files
            sub_data = []                                               # Empty list to append into, will convert to array

            print("Vectorizing brain maps...\n")

            for sub in tqdm(brain_data):
                  test_img = image.load_img(sub)                        # Convert Path to NifTi image

                  # Resample to MNI152 and vectorize voxels into array
                  test_resample = self.resample_to_MNI152(test_img).get_fdata().flatten()
                  
                  # Add voxels to master array
                  sub_data.append(test_resample)

            brain_vector = np.array(sub_data)                           # Convert list to Numpy array

            # Read in PINES map, resample, and vectorize
            pines = self.load_PINES_signature()
            pines_flat = self.resample_to_MNI152(pines).get_fdata().flatten()

            # Return array of dot products - represents 
            return np.dot(brain_vector, pines_flat)
