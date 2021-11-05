#!/bin/python3

"""
About this Script

* Creates subject-wise job scripts to deploy on HPC
* Accepts one command line argument - must correspond to to functional task

Ian Richard Ferguson | Stanford University
"""

# ----- Imports
import os, json, sys, warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


# ----- Functions
def write_job(SUB, SHORT, LONG):
      """
      SUB => Five digit subject identifier
      SHORT => Abbreviated task name (e.g., eval)
      LONG => Full task name (e.g., socialeval)

      Returns string that will be converted to job script
      """

      file_name = f"""#!/bin/bash
#SBATCH --job-name={SHORT.lower()}-{SUB}.job
#SBATCH --time=8-00:00
#SBATCH --mem=12000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irf823@stanford.edu
#SBATCH -c 8
#SBATCH -N 1

ml python/3.6.1
source ./level1/bin/activate

python3 wrapper.py {LONG} {SUB} 
      """

      return file_name


def derive_task_name(TASK):
      """
      TASK => ['faces', 'socialeval', 'stressbuffer']

      Returns shortened version of functional task name
      """

      TASK = TASK.lower()

      if TASK not in ['faces', 'socialeval', 'stressbuffer']:
            raise ValueError(f"{TASK} is not a valid functional task")

      if TASK == "faces":
            return "faces"
      elif TASK == "socialeval":
            return "eval"
      elif TASK == "stressbuffer":
            return "stress"


def main():
      bids_root = sys.argv[1]
      task_name = sys.argv[2]

      with open('./task_information.json') as incoming:
            task_data = json.load(incoming)[task_name]

      excludes = task_data['excludes']
      short_name = derive_task_name(task_name)

      for dir in tqdm(os.listdir(os.path.join('.', bids_root))):
            if "sub-" in dir:
                  sub_id = dir.split('-')[1]

                  if sub_id not in excludes:
                        filename = f"{short_name}_{sub_id}.sh"

                        with open(filename, "w") as output:
                              output.write(write_job(sub_id, short_name, task_name))


if __name__ == "__main__":
      main()