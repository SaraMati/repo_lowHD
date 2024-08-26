"""
This module contains the functions required to create the
cell metrics file to your local computer. It will use the DANDI
dataset that you have downloaded and run analyses and will save them to a
.csv file titled "cellme
Creating this file is a prerequisite for the downstream analysis.

@Author: Selen Calgin
@Date: 26/08/2024
"""

def create_cell_metrics():

    # set up configuration
    cell_metrics_path, data_dir, project_dir, results_dir = config()

