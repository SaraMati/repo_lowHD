"""
This file contains functions to assist analysis.
To be further organized.

Date created: 22/07/2024
Author: Selen Calgin
"""
import numpy as np
import pandas as pd
import os
import pynapple as nap
from misc import *
import scipy
import matplotlib.pyplot as plt
import configparser


cell_metrics_path, data_dir, project_dir, results_dir = config()

def load_cell_metrics(path=cell_metrics_path):
    """
    Loads the cell metrics CSV file into a Pandas dataframe.
    :param path: (str) Path to cell metrics spreadsheet.
    :return: Returns Pandas dataframe of the cell metrics
    """
    return pd.read_csv(path)

def load_data(session, remove_noise=True, data_directory=data_dir,
              cell_metrics_path=cell_metrics_path, lazy_loading=True):
    """
    :param session: (str) Session name, formatted according to cell metrics spreadsheet
    :param remove_noise: (bool) Whether to remove noisy cells. Default is True.
    :param data_directory: (str) Directory where data is stored
    :return: Pynapple data with the cell metrics added to the units
    """

    # load data
    folder_name, file_name = generate_session_paths(session)
    data_path = os.path.join(data_directory, folder_name, file_name)
    data = nap.NWBFile(data_path, lazy_loading=lazy_loading)

    # load cell metrics
    cell_metrics = load_cell_metrics(path=cell_metrics_path)
    cell_metrics = cell_metrics[cell_metrics['sessionName'] == session]  # restrict cell metrics only to current session
    cell_metrics = cell_metrics.reset_index(drop=True)  # reset index to align with spikes indices

    # add cell metrics to spike metadata
    data['units'].set_info(cell_metrics)

    if remove_noise:
        # remove noisy cells
        cell_tags = data['units'].getby_category('gd')
        data['units'] = cell_tags[1]  # getting all cells where good = 1 (=True)

    return data

def smooth_angular_tuning_curves(tuning_curves, window=20, deviation=3.0):
    new_tuning_curves = {}
    for i in tuning_curves.columns:
        tcurves = tuning_curves[i]
        offset = np.mean(np.diff(tcurves.index.values))
        padded = pd.Series(index=np.hstack((tcurves.index.values - (2 * np.pi) - offset,
                                            tcurves.index.values,
                                            tcurves.index.values + (2 * np.pi) + offset)),
                           data=np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=window, win_type='gaussian', center=True, min_periods=1).mean(std=deviation)
        new_tuning_curves[i] = smoothed.loc[tcurves.index]

    new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

    return new_tuning_curves

def calculate_speed(position):
    """
    Calculate animal's speed from animal's 2D position data
    :param position: (TsdFrame) Animal's 2D position data
    :return: (Tsd) Animal's speed data over time
    """
    speed = np.hstack([0, np.linalg.norm(np.diff(position.d, axis=0), axis=1) / (position.t[1] - position.t[0])])
    speed = nap.Tsd(t=position.t, d=speed)
    return speed


def calculate_speed_adrian(position):
    """
    TODO: Convert to Tsd
    Speed calculation from Adrian
    :param position: (TsdFrame) Animal's 2D position data
    :return: (Tsd) Animal's speed data over time
    """
    dx = position['x'].diff()
    dy = position['y'].diff()
    dt = np.diff(position.t)

    # Calculate velocity components
    vx = dx / dt
    vy = dy / dt

    # calculate the magnitude of velocity
    v = np.sqrt(vx ** 2 + vy ** 2)

    return v


def generate_session_paths(session):
    """
    The data in folder 000939 is not named consistently with the data tables.
    This helper function takes the session name and converts it to the folder and file name
    Example:
    Input: 'A3707-200317'
    Output: ('sub-A3707', 'sub-A3707_behavior+ecephys.nwb')

    :param session: (str) session name
    :return: Tuple of folder and file name
    """
    # Extract subject ID from session name
    subject_id = session.split('-')[0]

    # Generate folder and file names
    folder_name = f'sub-{subject_id}'
    file_name = f'sub-{subject_id}_behavior+ecephys.nwb'

    return (folder_name, file_name)


def get_wake_square_high_speed_ep(data, thresh=3):
    """
    Get the wake square epoch restricted to the animal's high velocity.
    This is the epoch needed for tuning curves in square terrain

    TODO: If we incorportate speed into the data object itself, adjust accordingly
    :param data:
    :param thresh:
    :return:
    """

    # Get wake square epoch
    wake_square_ep = data['epochs']['wake_square']

    # Restrict head direction to square wake epoch
    wake_square_hd = data['head-direction'].restrict(wake_square_ep)

    # Redefine wake square epoch based on first and last timestamp of restricted head direction
    # (to remove recording artifact)
    wake_square_ep = nap.IntervalSet(start=wake_square_hd.index[0], end=wake_square_hd.index[-1])

    # Further restrict epoch by high speed
    speed = calculate_speed(data['position'])
    high_speed_ep = speed.threshold(3, 'above').time_support
    wake_square_high_velocity_ep = wake_square_ep.intersect(high_speed_ep)

    return wake_square_high_velocity_ep


def split_epoch(epoch):
    """
    Given an interval set, returns the interval split in half.

    :param epoch: nap.IntervalSet object that contains epoch
    :return: tuple of IntervalSets
    """

    start = min(epoch['start'])
    end = max(epoch['end'])
    mid = (start + end) / 2

    epoch_1 = nap.IntervalSet(start=start, end=mid)  # first half
    epoch_2 = nap.IntervalSet(start=mid, end=end)  # second half

    return epoch_1, epoch_2

def get_sessions(cell_metrics_path=cell_metrics_path):
    """
    To get a list of all sessions
    :param cell_metrics_path: (str) Path to cell metrics file
    :return: List of stirngs of the session names
    """

    cm = load_cell_metrics(path=cell_metrics_path)
    return cm['sessionName'].unique().tolist()


def get_cell_type(session, cellID, cell_metrics_path=cell_metrics_path, noise=False):
    """
    GEt the cell type of a given cell given its sesison and cell ID.
    :param session: (str) session name
    :param cellID: (str) cell ID (note cell ID starts counting at 1, not 0, and restarts per session)
    :param cell_metrics_path: path to cell metrics file
    :param noise: If True, returns "noise" for noisy. If False, returns None.
    :return: (str) cell type  ('hd', 'nhd', 'fs', 'other', 'noise', or None)

    """
    # import cell metrics
    cm = load_cell_metrics(cell_metrics_path=cell_metrics_path)

    # get the specific row of the cell
    cell = cm[(cm['sessionName'] == session) & (cm['cellID'] == cellID)]

    if cell['hd'].values[0] == 1:
        return 'hd'

    elif cell['nhd'].values[0] == 1:
        return 'nhd'

    elif cell['fs'].values[0] == 1:
        return 'fs'

    elif cell['gd'].values[0] == 1:  # not hd, nhd, or fs, but good
        return 'other'

    else:
        if noise:
            return 'noise'
        else:
            return None

def get_cell_parameters(session, cellID, cell_metrics_path=cell_metrics_path):
    """
    This is useful when working with data without using Pynapple
    TODO: Check if cell ID is integer or string.
    :param session: (str) session name
    :param cellID: (int)
    :return: Pandas Datarame row of the cell
    """
    cm = load_cell_metrics(cell_metrics_path=cell_metrics_path)
    cell = cm[(cm['sessionName']==session) & (cm['cellID']==cellID)]
    return cell


def compute_log_bins(x, bins):
    """
    Computes number of bins for a histogram with a log scale
    :param x: data
    :param bins: number of bins
    :return: number of bins converted for a log scale
    """
    logbins = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), bins + 1)
    return logbins
