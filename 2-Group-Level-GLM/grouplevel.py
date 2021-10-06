#!/bin/python3

# ----- Imports
import warnings
warnings.filterwarnings('ignore')

import os, json, pathlib
import pandas as pd

import nilearn.plotting as nip
from bids.layout import BIDSLayout

# PROJECT ROOT
root = './bids'

try:
      layout = BIDSLayout(root)
except:
      raise OSError("Check your BIDS root definition @ line 14")

all_subjects = layout.get_subjects()
all_tasks = layout.get_tasks()

class GroupLevel:

      def __init__(self, task):
            """

            """

            self.task = task
            self.subjects = self._iso_subjects()
            self.all_brain_data = self._brain_data()
            task_file = self._taskfile_validator()


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


      def _iso_subjects(self):
            """
            Returns a list of subjects with first-level maps
            """

            task_subjects = []
            deriv_level = os.path.join(root, "derivatives/first-level")

            for sub in all_subjects:
                  temp = os.path.join(deriv_level, f"sub-{sub}", f"task-{self.task}")
                  if os.path.isdir(temp):
                        task_subjects.append(f"sub-{sub}")

            return task_subjects


      def _brain_data(self):
            """
            Returns a dictionary of subject:run key-value pairs
            """

            data = {k:[] for k in self.subjects}
            base = os.path.join(root, "derivatives/first-level")

            for sub in self.subjects:
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
            """

            output = []

            try:
                  for sub in self.subjects:
                        for scan in self.all_brain_data[sub]:
                              if contrast in scan:
                                    output.append(scan)

            except:
                  raise ValueError(f"Your specfied contrast {contrast} not found in functional runs")

            if len(output) == 0:
                  raise ValueError(f"Your specified contrast {contrast} not found in functional runs")
            
            return output


      def build_design_matrix(self):
            """
            Creates a design matrix with subject-wise regressors

            Work to do
                  * Should accomodate subject demographics
                  * Should accomodate indegree/outdegree/other network regressors

            Returns design matrix as Pandas DF
            """
            from nilearn.glm.second_level import make_second_level_design_matrix

            with open('./scp_subject_information.json') as incoming:
                  scp_subjects = json.load(incoming)

            with open('./scp_task_information.json') as incoming:
                  networks = json.load(incoming)[self.task]

            def change_key(key):
                  iso = str(key)[1:]
                  return f"sub-{iso}"

            subject_data = {change_key(k):v for k,v in scp_subjects.items()}

            def network_value(x, var):
                  try:
                        return subject_data[x][var]
                  except:
                        return 'NA'

            subjects_label = self.subjects
            design_matrix = pd.DataFrame({'subject_label': subjects_label})

            for var in networks['group-level-regressors']:
                  design_matrix[var] = design_matrix['subject_label'].apply(lambda x: network_value(x, var=var))
                  design_matrix[var] = pd.to_numeric(design_matrix[var])

            return make_second_level_design_matrix(subjects_label, design_matrix)
