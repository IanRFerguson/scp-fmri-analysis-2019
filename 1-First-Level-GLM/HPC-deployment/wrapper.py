#!/bin/python3

"""
About this Script

* Instantiates subject object
* Runs first-level GLM

Ian Richard Ferguson | Stanford University
"""

# ----- Imports
from communities import SCP_Sub
import sys

task_name = sys.argv[1]
sub_id = sys.argv[2]

for smooth in [4., 8.]:
      sub = SCP_Sub(sub_id, task_name)
      sub.run_first_level_glm(smoothing=smooth)