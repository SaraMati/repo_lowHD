"""
Functions for angular tuning curves
@Author: Selen Calgin
@Date: 29/07/2024
@Last edited: 14/09/2024
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

# Set up configuration
data_dir, results_dir, cell_metrics_dir, cell_metrics_path = config()


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




def compute_angular_tuning_curves(session):
    """
    This function calculates the smoothed angular tuning curves of a session, restricted to the high velocity
    square epoch.
    :param session: (str) session name
    :return: dataframe of angular tuning curves
    """

    data = load_data_DANDI_postsub(session, remove_noise=False, lazy_loading=False)

    # Get square wake + high velocity epoch 
    epoch = get_wake_square_high_speed_ep(data)

    # Get head direction data
    angle = data['head-direction']

    # Restrict epoch to where angle has no nas
    epoch = remove_na(epoch, angle)

    # Restrict angle to epoch
    angle_wake = angle.restrict(epoch)

    # Calculate tuning curves
    bins = np.linspace(0, 2 * np.pi, 180)
    nb_bins = len(bins)
    # epoch will be the time support of the feature
    tc = pd.DataFrame(smooth_angular_tuning_curves(
        nap.compute_1d_tuning_curves(group=data['units'], feature=angle_wake, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))

    return tc


def compute_control_angular_tuning_curves(session):
    """
    This function calculates the control smooth angular tuning curves with the head-direction time-reversed.
    The tuning curves is restricted to the high velocity square epoch.
    :param session: (str) session name
    :return: dataframe of angular tuning curves
    """

    data = load_data_DANDI_postsub(session, remove_noise=False, lazy_loading=False)

    # Get square wake + high velocity epoch 
    epoch = get_wake_square_high_speed_ep(data)

    # Get head direction data
    angle = data['head-direction']

    # Restrict epoch to where angle has no nas
    epoch = remove_na(epoch, angle)

    # Restrict angle to epoch
    angle_wake_reversed = time_reverse_feature(angle,epoch)

    # Calculate tuning curves
    bins = np.linspace(0, 2 * np.pi, 180)
    nb_bins = len(bins)
    # epoch will be the time support of the feature
    tc = pd.DataFrame(smooth_angular_tuning_curves(
        nap.compute_1d_tuning_curves(group=data['units'], feature=angle_wake_reversed, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))

    return tc


def compute_split_angular_tuning_curves(session):
    """
    This function calculates two tuning curves, one for each half of the epoch.
    The epoch is the high velocity square epoch. These split tuning curves
    is used to compare the stability of head direction tuning across the epoch.
    :param session: (str) session name
    :return: tupple of dataframe containing the two sets of angular tuning curves
    """
    data = load_data_DANDI_postsub(session, remove_noise=False, lazy_loading=False)

    # Get square wake + high velocity epoch and split it
    epoch = get_wake_square_high_speed_ep(data)
    epoch_half1, epoch_half2 = split_epoch(epoch)

    # Get head direction data
    angle = data['head-direction']

    # Restrict epoch to where angle has no nas
    epoch_half1 = remove_na(epoch_half1, angle)
    epoch_half2 = remove_na(epoch_half2, angle)

    # Restrict angle to epochs
    angle_half1 = angle.restrict(epoch_half1)
    angle_half2 = angle.restrict(epoch_half2)

    # Calculate tuning curves
    bins = np.linspace(0, 2 * np.pi, 180)
    nb_bins = len(bins)

    # epoch will be the time support of the feature
    tc_half1 = pd.DataFrame(smooth_angular_tuning_curves(
        nap.compute_1d_tuning_curves(group=data['units'], feature=angle_half1, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))
    tc_half2 = pd.DataFrame(smooth_angular_tuning_curves(
        nap.compute_1d_tuning_curves(group=data['units'], feature=angle_half2, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))

    return tc_half1, tc_half2

def compute_control_split_angular_tuning_curves(session):
    """
    This function calculates two control tuning curves, one for each half of the epoch,
    using time-reversed head-direction.
    The epoch is the high velocity square epoch. These split tuning curves
    is used to compare the stability of head direction tuning across the epoch.
    :param session: (str) session name
    :return: tupple of dataframe containing the two sets of angular tuning curves
    """
    data = load_data_DANDI_postsub(session, remove_noise=False, lazy_loading=False)

    # Get square wake + high velocity epoch and split it
    epoch = get_wake_square_high_speed_ep(data)
    epoch_half1, epoch_half2 = split_epoch(epoch)

    # Get head direction data
    angle = data['head-direction']

    # Restrict epoch to where angle has no nas
    epoch_half1 = remove_na(epoch_half1, angle)
    epoch_half2 = remove_na(epoch_half2, angle)

    # Restrict angle to epochs
    angle_half1 = time_reverse_feature(angle, epoch_half1)
    angle_half2 = time_reverse_feature(angle, epoch_half2)

    # Calculate tuning curves
    bins = np.linspace(0, 2 * np.pi, 180)
    nb_bins = len(bins)

    # epoch will be the time support of the feature
    tc_half1 = pd.DataFrame(smooth_angular_tuning_curves(
        nap.compute_1d_tuning_curves(group=data['units'], feature=angle_half1, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))
    tc_half2 = pd.DataFrame(smooth_angular_tuning_curves(
        nap.compute_1d_tuning_curves(group=data['units'], feature=angle_half2, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))

    return tc_half1, tc_half2

def compute_hd_info(data, tuning_curves, control):
    """
    Computes mutual info of the angular tuning curves and the head direction
    feature given the tuning curve and the session data
    :param data: Pynapple object containing the data for the session
    :param tuning_curves: The angular tuning curve of the session
    :param control: (bool) Whether tuning curves are time-reversed control or not
    :return: Pandas Dataframe containing the mutual info
    """

    # Extract needed data
    wake_epoch = remove_na(get_wake_square_high_speed_ep(data), data['head-direction'])
    angle_wake = data['head-direction'].restrict(wake_epoch)

    if control:
        angle_wake = time_reverse_feature(angle_wake)

    # Calculate hd info
    hd_info = nap.compute_1d_mutual_info(tuning_curves, angle_wake, minmax=(0, 2 * np.pi))

    return hd_info

def compute_tuning_curve_correlations(tuning_curve_1, tuning_curve_2):
    """
    Computes Pearson correlation efficient between two tuning curves.
    Should be used on two tuning curves from the session (i.e. same
    number of neurons)
    Can be used on control or normal tuning curves
    :param tuning_curve_1: (Pandas.Dataframe) First tuning curve set
    :param tuning_curve_2: (Pandas.Dataframe) second tuning curve set
    :return:
    """

    data = {"cellID": [],
            "pearsonR": []}

    for cell in tuning_curve_1:

        # Calculate Pearson correlation coefficient between the two tuning curves
        pear_corr = scipy.stats.pearsonr(tuning_curve_1[cell], tuning_curve_2[cell])

        # Append info to arrays
        data['cellID'].append(np.int64(cell)+1)
        data['pearsonR'].append(pear_corr[0])


    return pd.DataFrame(data)


