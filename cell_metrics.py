"""
This module contains the functions required to create the
cell metrics file to your local computer. It will use the DANDI
dataset that you have downloaded and run analyses and will save them to a
.csv file titled "cellmetricsA37.csv"
Creating this file is a prerequisite for downstream analysis.
@Author: Selen Calgin
@Date created: 26/08/2024
@Last updated: 14/09/2024
"""
import numpy as np

import pandas as pd
import os
import pynapple as nap
import scipy
import matplotlib.pyplot as plt
import configparser
from functions import *
import seaborn as sns
from angular_tuning_curves import *

# set up configuration
data_dir, results_dir, cell_metrics_dir, cell_metrics_path = config()

def create_cell_metrics():

    try:
        # Check if the file exists
        if not os.path.isfile(cell_metrics_path):

            cell_metrics_data = [] # will hold data
            global_cell_count = 1
            sessions = get_sessions()

            # Iterate through sessions, load data, and analyze data
            for session in sessions:
                print(session) # for debugging and tracking
                # Load current session
                data = load_data_DANDI_postsub(session, remove_noise=False, lazy_loading=False)
                units = data['units'] # Get units

                # Get cell types
                cell_types = get_cell_types_from_DANDI(data)

                # Compute tuning curves
                tc = compute_angular_tuning_curves(session)
                tc_control = compute_control_angular_tuning_curves(session)
                tc_half1, tc_half2 = compute_split_angular_tuning_curves(session)
                tc_half1_control, tc_half2_control = compute_control_split_angular_tuning_curves(session)

                # Compute HD info
                hd_info = compute_hd_info(data, tc, control=False)
                hd_info_control = compute_hd_info(data, tc_control, control=True)

                # Compute split tuning curve correlations
                tc_correlations = compute_tuning_curve_correlations(tc_half1, tc_half2)
                tc_correlations_control = compute_tuning_curve_correlations(tc_half1_control, tc_half2_control)

                # Get rates
                average_rate = units['rate']
                wake_rate = units.restrict(get_open_field_ep(data))['rate']

                # Get trough to peak
                trough_to_peak = units['trough_to_peak']

                for unit in units:
                    cellID = unit+1

                    # Add all information of one neuron to one row of information
                    cell_metrics_data.append({
                        'sessionName': session,
                        'cell': global_cell_count, # Global count of all cells
                        'cellID': cellID, # Local count within the session
                        'firingRate': average_rate[unit],
                        'firingRateExplore': wake_rate[unit],
                        'troughtoPeak': trough_to_peak[unit],
                        'ex': cell_types['ex'][unit],
                        'hd': cell_types['hd'][unit],
                        'fs': cell_types['fs'][unit],
                        'nhd': cell_types['nhd'][unit],
                        'other': cell_types['other'][unit],
                        'hdInfo': hd_info.values.flatten()[unit],
                        'hdInfo_rev': hd_info_control.values.flatten()[unit],
                        'pearsonR': tc_correlations['pearsonR'][unit],
                        'pearsonR_rev': tc_correlations_control['pearsonR'][unit]
                    })

                    # Increment global cell counter
                    global_cell_count +=1

            # Create DataFrame from collected cell metrics data
            cell_metrics_df = pd.DataFrame(cell_metrics_data)

            # Save the DataFrame to CSV
            cell_metrics_df.to_csv(cell_metrics_path)
            print(f"Cell metrics file created at {cell_metrics_path}")

        else:
            print(f"Cell metrics file already exists at {cell_metrics_path}")

    except Exception as e:
        # Handle any exceptions during file creation or writing
        print(f"An error occurred while creating the cell metrics file: {str(e)}")