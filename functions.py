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
import scipy
import matplotlib.pyplot as plt

def load_cell_metrics(cell_metrics_path=r"D:\OneDrive - McGill University\Peyrache Lab\project_LowHD\data_tables\cellmetricsA37.csv"):
    """
    Loads the cell metrics CSV file into a Pandas dataframe.
    :param cell_metrics_path: (str) Path to cell metrics spreadsheet.
    :return: Returns Pandas dataframe of the cell metrics
    """
    return pd.read_csv(cell_metrics_path)


def load_data(session, remove_noise = True, base_path = r"D:\OneDrive - McGill University\Peyrache Lab\Data\000939", cell_metrics_path =r"D:\OneDrive - McGill University\Peyrache Lab\project_LowHD\data_tables\cellmetricsA37.csv" ):
    """
    TODO: Once session naming is consistent in datafiles, load the Pynapple file directly here.
    :param data: Pynapple object
    :param session: (str) Session name, formatted according to cell metrics spreadsheet
    :param remove_noise: (bool) Whether to remove noisy cells. Default is True.
    :param base_path
    :return: Pynapple data with the cell metrics added to the units
    """

    # load data
    folder_name, file_name = generate_session_paths(session)
    data_path = os.path.join(base_path, folder_name, file_name)
    data = nap.load_file(data_path)

    # load cell metrics
    cell_metrics = load_cell_metrics(cell_metrics_path=cell_metrics_path)
    cell_metrics = cell_metrics[cell_metrics['sessionName']==session] # restrict cell metrics only to current session
    cell_metrics = cell_metrics.reset_index(drop=True) # reset index to align with spikes indices

    # add cell metrics to spike metadata
    data['units'].set_info(cell_metrics)

    if remove_noise:
        # remove noisy cells
        cell_tags = data['units'].getby_category('gd')
        data['units'] = cell_tags[1] # getting all cells where good = 1 (=True)

    return data

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
    vx = dx/dt
    vy = dy/dt

    # calculate the magnitude of velocity
    v = np.sqrt(vx**2 + vy**2)

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

    # Restrct head direction to square wake epoch
    wake_square_hd = data['head-direction'].restrict(wake_square_ep)

    # Redefine wake square epoch based on first and last timestamp of restricted head direction
    # (to remove recording artifact)
    wake_square_ep = nap.IntervalSet(start=wake_square_ep.index[0], end=wake_square_ep.index[-1])

    # Further restrict epoch by high speed
    speed = calculate_speed(data['position'])
    high_speed_ep = speed.threshold(3, 'above')
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

    return (epoch_1, epoch_2)