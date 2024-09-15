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
import configparser

def config():
    """
    Gets paths from config.ini
    :return: paths
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    data_dir = config['Directories']['data_dir']
    results_dir = config['Directories']['results_dir']
    cell_metrics_dir = config['Directories']['cell_metrics_dir']
    cell_metrics_path = os.path.join(cell_metrics_dir, 'cellmetricsA37.csv')

    return data_dir, results_dir, cell_metrics_dir, cell_metrics_path

data_dir, results_dir, cell_metrics_dir, cell_metrics_path = config()

def load_cell_metrics(path=cell_metrics_path):
    """
    Loads the cell metrics CSV file into a Pandas dataframe.
    :param path: (str) Path to cell metrics spreadsheet.
    :return: Returns Pandas dataframe of the cell metrics
    """
    if not os.path.exists(path):
        # Create an empty DataFrame
        empty_df = pd.DataFrame()
        
        # Save the empty DataFrame to a CSV file
        empty_df.to_csv(path)

    df = pd.read_csv(path, index_col=0) 
    return df


def merge_cell_metrics(data, on, new_col, replace_existing_data, save, save_path=None, path=cell_metrics_path):
    """
    Merges a new data column into the cell metrics DataFrame.

    This function merges a new data column into an existing cell metrics DataFrame
    based on specified columns. If the column already exists and `replace_existing_data`
    is True, the existing column will be replaced.

    :param: data : pandas.DataFrame
         containing the data to be merged.
    :param: on : list
        List of columns to merge on.
    :param: new_col : str
        Name of the new data column.
    :param: replace_existing_data : bool
        If True, replaces the existing data column if it already exists.
    :param: save : bool
        If True, saves the new merged cell metrics as a CSV file.
    :para: save_path : str or None, optional
        Path to where the file will be saved if save=True. If None, will replace the
        existing cell metrics file (default is None).
    :param: cell_metrics_path : str, optional
        Path to the cell metrics file (default is None).

    :return: pandas.DataFrame
        The merged cell metrics DataFrame.
    """

    cell_metrics = load_cell_metrics()

    # drop index of new dataframe
    data = data.reset_index(drop=True)

    # if replacing the existing data
    if replace_existing_data and (new_col in cell_metrics.columns):
        # remove the existing column
        cell_metrics = cell_metrics.drop(new_col, axis=1)

    # merge data
    cell_metrics = pd.merge(cell_metrics, data, on=on, how='left')

    if save and (save_path is not None):
        cell_metrics.to_csv(save_path)  # save to new location
    elif save:
        cell_metrics.to_csv(cell_metrics_path)  # replace old cell metrics file

    return cell_metrics

def load_data_DANDI_postsub(session, remove_noise=True, data_directory=data_dir,
              cell_metrics_path=cell_metrics_path, lazy_loading=False):
    """
    :param session: (str) Session name, formatted according to cell metrics spreadsheet
    :param remove_noise: (bool) Whether to remove noisy cells. Default is True.
    :param data_directory: (str) Directory where data is stored
    :param: cell_metrics_path: (str) Directory to cell metrics file
    :param: lazy_loading: (bool) highly recommended this stays as False for further analysis
    :return: Pynapple data with the cell metrics added to the units
    """

    # load data
    folder_name, file_name = generate_session_paths(session)
    data_path = os.path.join(data_directory, folder_name, file_name)
    data = nap.NWBFile(data_path, lazy_loading=lazy_loading)

    # load cell metrics
    cell_metrics = load_cell_metrics(path=cell_metrics_path)
    if not cell_metrics.empty:
        cell_metrics = cell_metrics[cell_metrics['sessionName'] == session]  # restrict cell metrics only to current session
        cell_metrics = cell_metrics.reset_index(drop=True)  # reset index to align with spikes indices
        # add cell metrics to spike metadata
        data['units'].set_info(cell_metrics)

    #TODO: why was this commented out?
    # if remove_noise:
    #     # remove noisy cells
    #    cell_tags = data['units'].getby_category('gd')
    #    data['units'] = cell_tags[1]  # getting all cells where good = 1 (=True)

    # add waveforms to units
    # temp_variable = data.nwb.units.to_dataframe()
    # temp_Var = temp_variable['waveform_mean']
    # N = len(temp_variable['waveform_mean'][0])
    # sampling_rate = 20000 # we should read this from the nwb file 
    # time_vector = np.arange(0,N)/sampling_rate
    # waveforms = {}
    # for index, row in temp_variable.iterrows():
    #     waveforms[index] = nap.TsdFrame(t = time_vector, d = np.array(row['waveform_mean']))
    # not finished. will add to the main script for now
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
    vx = dx / dt
    vy = dy / dt

    # calculate the magnitude of velocity
    v = np.sqrt(vx ** 2 + vy ** 2)

    return v


def generate_session_paths(session):
    """
    The data in folder 000939 is named as "sub-[session name]".
    This helper function takes the session name and converts it to the folder and file name
    Example:
    Input: 'A3707'
    Output: ('sub-A3707', 'sub-A3707_behavior+ecephys.nwb') OR ('sub-A3707', 'sub-A3707_behavior+ecephys+ogen.nwb')

    :param session: (str) session name
    :return: Tuple of folder and file name
    """

    # Generate folder and file names
    folder_name = f'sub-{session}'
    folder_path = os.path.join(data_dir, folder_name)

    file_name = files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))][0]

    return (folder_name, file_name)

def get_open_field_ep(data, thresh=3):
    """
    Get open field exploration (both square and triangle). Exploration
    during wake is defined as when the animal is has high velocity.
    :param data:
    :param thresh:
    :return: Interval set of the epoch
    """

    epochs = data['epochs']
    epoch_names = epochs.keys()

    if 'wake_triangle' in epoch_names:
        wake_epoch = epochs['wake_triangle'].union(epochs['wake_square'])
    else:
        wake_epoch = epochs['wake_square'] # all sessions have square open field but not all have triangle

    # Restrict further to high velocity
    speed = calculate_speed(data['position'])
    high_speed_ep = speed.threshold(3, 'above').time_support
    exploration_ep = wake_epoch.intersect(high_speed_ep)

    return exploration_ep

def get_wake_square_high_speed_ep(data, thresh=3):
    """
    Get the wake square epoch restricted to the animal's high velocity.
    This is the epoch needed for tuning curves in square terrain.
    This function also defines the epoch based on the start and end time
    of head direction time support, to remove recording artifact
    (head direction tracking is started after the wake epoch starts).
    Thus this is the epoch that should be used for angular tuning curves.

    :param data: Pynapple data object
    :param thresh: Threshold for high speed
    :return: Interval set of the epoch
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

def remove_na(epoch, feature):
    """
    Given an epoch and feature, this function returns the epoch where the feature has value,
    i.e. restricts the epoch to exclude when the feature has nan

    Note this menas the epoch will be a subset of the feature's time support.
    If the epoch originally exceeded the bounds of the fetaure's time support, then it will
    be restricted.
    :param epoch:
    :param feature:
    :return:
    """

    feature_no_nans_epoch = feature.dropna().time_support

    return epoch.intersect(feature_no_nans_epoch)

def split_epoch(epoch):
    """
    Given an interval set, returns the interval split in half.

    :param epoch: nap.IntervalSet object that contains epoch
    :return: tuple of IntervalSets
    """

    start = min(epoch['start'])
    end = max(epoch['end'])
    mid = (start + end) / 2

    epoch_1 = epoch.intersect(nap.IntervalSet(start=start, end=mid))
    epoch_2 = epoch.intersect(nap.IntervalSet(start=mid, end=end))

    return epoch_1, epoch_2

def time_reverse_feature(feature, epoch=None):
    """
    Feature is time reversed
    :param feature:
    :param epoch: If none uses epoch of feature
    :return:
    """
    if epoch is None:
        epoch = feature.time_support

    # restrict feature to epoch
    feature_ep = feature.restrict(epoch)

    # extract feature values and time stamps
    feature_t = feature_ep.t
    feature_values = feature_ep.values

    # reverse values
    feature_values_rev = feature_values[::-1]

    # create new Tsd of time reversed feature
    feature_rev = nap.Tsd(t=feature_t, d=feature_values_rev)

    return feature_rev


def get_sessions():
    """
    To get a list of all sessions from the data directory
    :return: List of stirngs of the session names
    """

    # Get all folder names in the directory
    folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    # Extract the part after 'sub-' (like 'A303') from the folder names
    sessions = [folder.split('sub-')[-1] for folder in folders if folder.startswith('sub-')]

    return sessions

def get_cell_types_from_DANDI(data):
    """
    TODO: Figure out how to detect noisy cells
    This function goes through each unit of a session and based on the default
    classifications in the DANDI dataset (i.e. is_excitatory, is_fast_spiking, is_head_direction)
    will create a dataframe that will one-hot-encode the cell type.
    :param data: (Pynapple object) containing the session data
    :return: (Pandas Dataframe) of one-hot-encoded cell types
    """

    units = data['units']

    ex = units._metadata['is_head_direction']
    hd = units._metadata['is_excitatory']
    fs = units._metadata['is_fast_spiking']
    nhd = ((ex == 1) & (hd== 0)).astype(int)
    other = ((ex == 0) & (hd == 0) & (nhd == 0) & (fs == 0)).astype(int)

    one_hot_encoded_cell_types = pd.DataFrame({
        'ex': ex,
        'hd': hd,
        'fs': fs,
        'nhd': nhd,
        'other': other
    })

    return one_hot_encoded_cell_types


def get_viable_cells(data):
    """
    Identifies viable cells from PoSub recordings based on criteria from Duszkiewicz et al. (2024).

    Viable units are determined by two criteria:
    1. Units with an average firing rate of at least 0.5 Hz during open field exploration.
    2. Units with a waveform that exhibits a negative deflection, to exclude spikes from fibers of passage.

    Cells that do not meet these criteria are considered 'noisy'.

    :param data: Pynapple data object containing unit recordings and metadata.
    :return: DataFrame with one-hot encoded values indicating viable/good cells (1 = good, 0 = noise).
    """

    # Get epochs during open field exploration
    exploration_ep = get_open_field_ep(data)

    # Restrict units to open field exploration
    units_explore = data['units'].restrict(exploration_ep)

    # Get cells' rate during open field exploration
    rate_explore = units_explore['rate']

    # Get cells' trough to peaks
    t2p = units_explore['trough_to_peak'] # trough to peak stays same regardless of epoch

    # Classify cells as good or noise
    classification = ((rate_explore >= 0.5) & (t2p < 0)).astype(int)


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
    cm = load_cell_metrics(path=cell_metrics_path)

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
    cm = load_cell_metrics(path=cell_metrics_path)
    cell = cm[(cm['sessionName'] == session) & (cm['cellID'] == cellID)]
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


def compute_good_waveform(mean_w):
    """
    For a single neuron, returns the good waveform (i.e. from the best channel), and the channel
    on which the best waveform is recorded.
    :param mean_w: waveforms of a single neuron (40x64) array
    :return:(array with good waveform, channel in which good waveform is recorded)
    """
    minimum = 0
    channel = -1
    waveform = None

    for chan, wave in enumerate(mean_w):
        temp_min = min(wave)
        if temp_min < minimum:
            minimum = temp_min
            channel = chan
            waveform = wave

    return (waveform, channel)

def ensure_dir_exists(directory):
    """
    Check if directory exist, if not create it.
    :param directory: (str)
    :return: bool: True if already exists, False if not
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return False
    else:
        return True