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
        cell_metrics = load_cell_metrics(path=cell_metrics_path)
        if cell_metrics.empty:         

            cell_metrics_data = [] # will hold data
            global_cell_count = 1
            sessions = get_sessions()

            # Iterate through sessions, load data, and analyze data
            for session in sessions:
                print(session) # for debugging and tracking
                # Load current session
                data = load_data_DANDI_postsub(session, remove_noise=False, lazy_loading=False)

                ## This block cleans up the data to the epoch of interest
                ## change based on your needs, the rest of the analysis is based on this final epoch
                # select the desired epoch, data has wake_square and wake_triangle 
                desired_epoch = 'wake_square'
                epoch = data['epochs'][desired_epoch]
                # restrict angle (head direction) to the epoch
                angle = data['head-direction'].restrict(epoch)
                # we also have to restrict all the time series to the time of the angles (because the Motive software/Optptrack was turned on after the start of the electrophysiology recording)
                epoch2 = nap.IntervalSet(start=angle.index[0], end=angle.index[-1])
                #restrict units and behavioral data to the epoch
                units = data['units'].restrict(epoch2)
                angle = angle.restrict(epoch2)
                position = data['position'].restrict(epoch2)
                speed = calculate_speed(position)
                # Further restrict epoch by high speed
                desired_speed_threshold = 3 
                high_speed_ep = speed.threshold(desired_speed_threshold, 'above').time_support
                epoch3 = epoch2.intersect(high_speed_ep)
                units = units.restrict(epoch3)
                angle = angle.restrict(epoch3)
                position = position.restrict(epoch3)
                speed = speed.restrict(epoch3)
                ## End of block
                
                # Get cell types
                cell_type_labels = get_cell_types_from_DANDI(units)

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
                #TODO: make open field and wake epoch consistent 
                average_rate = units['rate']
                wake_explore = units.restrict(get_open_field_ep(data))['rate']

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
                        'firingRateExplore': wake_explore[unit],
                        'troughtoPeak': trough_to_peak[unit],
                        'ex': cell_type_labels['ex'][unit],
                        'hd': cell_type_labels['hd'][unit],
                        'fs': cell_type_labels['fs'][unit],
                        'nhd': cell_type_labels['nhd'][unit],
                        'other': cell_type_labels['other'][unit],
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