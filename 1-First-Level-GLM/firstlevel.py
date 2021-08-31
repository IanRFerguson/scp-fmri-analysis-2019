#!/bin/python3

"""
This script leverages py-bids to run a first-level GLM on all complete subjects
Two command line arguments required (in this order):
      * Path to your BIDS project
      * Task name

Ian Richard Ferguson | Stanford University
"""

# ---- Imports
import warnings
warnings.filterwarnings('ignore')

from communities import SCP_Sub
import os
import json
import sys
from bids.layout import BIDSLayout

arg = sys.argv[1]
layout = BIDSLayout(arg)
task = sys.argv[2]


# ---- Functions
def confirmSetup():
      """
      Determines:
            * Presence of proper command line arguments
            * Presence of JSON files 
      """

      arguments = sys.argv

      if len(arguments) != 3:
            raise RuntimeError(f"""
            Missing command line arguments (need (1) BIDS root and (2) functional task ... 
            Your input: {arguments}
            """)

      check1 = os.path.join('./scp_subject_information.json')
      check2 = os.path.join('./scp_task_information.json')
      
      for check in [check1, check2]:
            if not os.path.exists(check):
                  raise OSError(f"Missing {check} @ root directory...")


def dumpSubs(TASK):
      """
      Returns list of subjects to skip
      NOTE: These subjects were identified via MRIQC quality checks per run
      """

      with open("./scp_task_information.json") as incoming:
            data = json.load(incoming)
            return list(data[TASK]['excludes'])


def run1stLevel(SUB):
      """
      * Instantiates SCP_Sub object
      * Sets dummy scans from MRIQC output
      """

      sub = SCP_Sub(SUB, task)                  # Instantiates SCP_Sub object wrapper

      print(f"\n--------- Running sub-{sub.subID} {sub.task} first level GLM\n")
      
      sub.run_first_level_glm()                 # Runs first level GLM


def main():
      """
      Runs when program is called
      """

      confirmSetup()
            
      bad_subs = dumpSubs(task)                 # Subs to skip
      all_subs = layout.get_subjects()          # List of all subjects

      for subject in all_subs:
            if subject not in list(bad_subs):
                  with open('./firstlevel_errorLog.txt', 'w') as log:
                        try:
                              run1stLevel(subject)
                        except Exception as e:
                              log.write(f"\n------Error @ {subject}")
                              log.write(f"\n{e}\n\n")

            print("\n\n--------- All subjects mapped")


# ---- Run 
if __name__ == "__main__":
      main()
