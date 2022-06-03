# Analysis Scripts

This repo has several analysis scripts we used to conduct different modeling approaches

* `abbreviate_perspective.py`: Used to assess HRF drift in our perspective task, which is 30s by default

* `easy_RSA.py`: This script is optimized to compare neural patterns

* `pool_active_baseline.py`: This script merges the `spatial` trial type with our implicit baseline, which allows us to better estimate the targeted social trial types in this task


The `task_information.json` file is a derivative of <a href="https://github.com/IanRFerguson/glm-express" target="_blank">`GLM Express`</a>, the primary analysis engine for this project.