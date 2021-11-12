#!/bin/python3

"""
About this Script

Our fmriprep job script takes a list of subjects from your BIDS project and deploys a preprocessing script
for each subject. This script saves a text file that you can copy and paste into Job-Script.sh to facilitate
preprocessing for each subject. A little hacky but it gets the job done

Ian Richard Ferguson | Stanford University
"""


# ----- Imports
from bids import BIDSLayout
import sys


# ----- Functions
def validate_bids():
      """
      Confirms that your project is BIDS-valid

      Returns BIDSLayout object
      """

      try:
            layout = BIDSLayout(sys.argv[1])
      except:
            raise OSError("Missing command line argument: please supply relative path to BIDS root")

      return layout


def derive_subs(layout):
      """
      Isolates subject ID's in your BIDS project

      Returns bash-formatted list of subject ID's
      """

      subs = [x for x in layout.get_subjects()]

      return str(subs)[1:-1].replace(",", "")


def write_subs(subs):
      """
      Writes list of subject ID's to a local text file
      This will be saved at the directory level you ran this script from
      """

      with open('./bids-project-subjects.txt', 'w') as outgoing:
            outgoing.write(subs)


def main():
      layout = validate_bids()
      subs = derive_subs(layout)
      write_subs(subs)

      print("\nSee bids-project-subjects.txt - plug these into fmriprep Job Script directly")


if __name__ == "__main__":
      main()