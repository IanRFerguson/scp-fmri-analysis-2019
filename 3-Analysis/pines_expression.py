#!/bin/python3

"""
About this Script

We're going to loop through a subset of first-level contrast maps
and assess the relative PINES expression for each map. We'll aggregate
these in one local CSV file. 

Assumptions:
      * You've run at least one first-level contrast
      * This script is deployed from one directory above your BIDS project

Ian Richard Ferguson
"""

# --- Imports
import os, glob, sys
from nilearn.image import resample_to_img, image
from bids import BIDSLayout
from tqdm import tqdm
import nibabel as nib
import pandas as pd
import numpy as np


# --- Helpers
def paths_to_maps(rel_path, contrast_of_interest=None):
      """
      This function returns a list of relative paths to contrasts
      of interest. If no contrast is explicitly supplied, we'll
      default to ALL contrasts

      Parameters
            rel_path: str | Relative path to your first-level output
            contrast_of_interest: str | Name of contrast to isolate

      Returns
            List of relative paths to contrast maps
      """

      all_maps = [x for x in glob.glob(f'{rel_path}/**/*.nii.gz', recursive=True)]

      if contrast_of_interest is not None:
            return [x for x in all_maps if contrast_of_interest in x]
      
      return all_maps


def estimate_PINES_expression(new_map, PINES_map):
      """
      After checking for matrix similarity, we're going to
      compute the dot product of a given contrast map and our
      pre-trained PINES weights

      Parameters
            new_map: str | Relative path to a contrast map
            PINES_map: str | Relative path to our PINES map

      Returns
            Float
      """

      nifti = nib.load(new_map)
      pines = nib.load(PINES_map)

      # Congruence checks
      if nifti.shape != pines.shape:
            need_to_resample = True
      elif not (nifti.affine.flatten() == pines.affine.flatten()).all():
            need_to_resample = True
      else:
            need_to_resample = False

      if need_to_resample:
            pines = resample_to_img(pines, nifti)

      flat_nifti = nifti.get_fdata().flatten()
      flat_pines = pines.get_fdata().flatten()

      return np.dot(flat_nifti, flat_pines)


def run_script(contrast_path, BIDS_path, contrast_of_interest=None):
      """
      We'll loop through maps of interest, calculate each
      map's PINES expression, and aggregate to a Pandas DataFrame

      Parameters
            rel_path: str | Relative path to your first-level output
            contrast_of_interest: str | Name of contrast to isolate

      Returns
            Pandas DataFrame object
      """

      list_of_maps = paths_to_maps(contrast_path, contrast_of_interest)
      pines_path = os.path.join(BIDS_path, 'derivatives/masks/Rating_Weights_LOSO_2.nii')
      output = pd.DataFrame()

      for map_ in tqdm(list_of_maps):
            subject_id = map_.split('/')[-1].split('_')[0]
            contrast_name = map_.split('/')[-1]

            pines_expression = estimate_PINES_expression(map_, pines_path)

            temp = pd.DataFrame({'subID': subject_id, 'contrastName':contrast_name, 'pinesValue':pines_expression}, index=[0])

            output = output.append(temp, ignore_index=True)

      if contrast_of_interest is not None:
            filename = f'pines_pattern_expression_contrast-{contrast_of_interest}.csv'
      else:
            filename = 'pines_pattern_expression_ALL-CONTRASTS.csv'

      output.to_csv(os.path.join('.', filename), index=False)


def main():

      try:
            user_contrast_path = sys.argv[1]
      except:
            user_contrast_path = input('\nRelative path to first-level output:\t')

      try:
            user_bids_path = sys.argv[2]
      except:
            user_bids_path = input('Relative path to BIDS project:\t\t')

      try:
            user_contrast_interest = sys.argv[3]

            if user_contrast_interest.upper() == 'ALL':
                  user_contrast_interest = None
      except:
            user_contrast_interest = input('Contrast of interest (or type ALL):\t')

            if user_contrast_interest.upper() == 'ALL':
                  user_contrast_interest = None

      # --- Let's rock
      run_script(contrast_path=user_contrast_path, 
                 BIDS_path=user_bids_path, 
                 contrast_of_interest=user_contrast_interest)

      print('\n** Local CSV saved! **\n')


if __name__ == "__main__":
      main()