"""
Functions for angular tuning curves
To be further organized
@Author: Selen Calgin
@Date: 29/07/2024
"""

import numpy as np

import pandas as pd
import os
import pynapple as nap
import scipy
import matplotlib.pyplot as plt
import configparser
from functions import *
from misc import *
import seaborn as sns

# Set up configuration
cell_metrics_path, data_dir, project_dir, results_dir = config()

def compute_all_angular_tuning_curves(time_reverse, print_progress=False):
    """
    Computing tuning curves for all cells in each session, including split tuning curves, and saving.

    The tuning curve is calculated for the entire square wake + high velocity epoch,
    and split the tuning curves within this epoch.

    Note that tuning curves are even computed for noisy cells. This is for easy integration
    into upstream analysis.
    :param: time_reverse : bool
        Set to True to calculate time reversed control
    :param: print_progress : bool
        To print progress session by session
    :return: None
    """

    # get all sessions
    sessions = get_sessions()

    for session in sessions:

        # folder to save tuning curves
        save_dir = os.path.join(results_dir, session, "Analysis")
        ensure_dir_exists(save_dir)

        if print_progress:
            print(session) # To know the progress of the script when running

        data = load_data(session, remove_noise=False, lazy_loading=False)

        # get square wake + high velocity epoch and split it
        epoch = get_wake_square_high_speed_ep(data)
        epoch_half1, epoch_half2 = split_epoch(epoch)

        # getting head direction data
        angle = data['head-direction']

        # restrict epoch to where angle has no nans
        epoch = remove_na(epoch, angle)
        epoch_half1 = remove_na(epoch_half1, angle)
        epoch_half2 = remove_na(epoch_half2, angle)

        if not time_reverse:
            angle_wake = angle.restrict(epoch)
            angle_half1 = angle.restrict(epoch_half1)
            angle_half2 = angle.restrict(epoch_half2)

        else:
            angle_wake = time_reverse_feature(angle, epoch)
            angle_half1 = time_reverse_feature(angle, epoch_half1)
            angle_half2 = time_reverse_feature(angle, epoch_half2)

        # calculate tuning curves
        bins = np.linspace(0, 2 * np.pi, 180)
        nb_bins = len(bins)
        # epoch will be the time support of the feature

        tc = pd.DataFrame(smooth_angular_tuning_curves(
            nap.compute_1d_tuning_curves(group=data['units'], feature=angle_wake, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))
        tc_half1 = pd.DataFrame(smooth_angular_tuning_curves(
            nap.compute_1d_tuning_curves(group=data['units'], feature=angle_half1, nb_bins=nb_bins, minmax=(0, 2 * np.pi) )))
        tc_half2 = pd.DataFrame(smooth_angular_tuning_curves(
            nap.compute_1d_tuning_curves(group=data['units'], feature=angle_half2, nb_bins=nb_bins, minmax=(0, 2 * np.pi))))

        # save tuning curves
        if time_reverse:
            full_title = "tuning_curves_smooth_reverse.csv"
            half1_title = "tuning_curves_smooth_half1_reverse.csv"
            half2_title = "tuning_curves_smooth_half2_reverse.csv"
        else:
            full_title = "tuning_curves_smooth.csv"
            half1_title = "tuning_curves_smooth_half1.csv"
            half2_title = "tuning_curves_smooth_half2.csv"

        tc.to_csv(os.path.join(save_dir, full_title))
        tc_half1.to_csv(os.path.join(save_dir, half1_title))
        tc_half2.to_csv(os.path.join(save_dir, half2_title))

def compute_tuning_curve_correlations(session):
    """
    TODO: Add catch if tuning curve files don't exist

    Computes Pearson correlation coefficient between the angular tuning curves split into the first and
    second half of the epoch.
    :param session: (str)
    :return:
    """
    tuning_curve_dir = os.path.join(results_dir, session, "Analysis")

    data = {"sessionName":[],
            "cellID": [],
            "pearsonR": []}

    # path to tuning curves
    path_half1 = os.path.join(tuning_curve_dir, "tuning_curves_smooth_half1.csv")
    path_half2 = os.path.join(tuning_curve_dir, "tuning_curves_smooth_half2.csv")

    # load tuning curves
    tc_half1 = pd.read_csv(path_half1, index_col=0)
    tc_half2 = pd.read_csv(path_half2, index_col=0)

    # iterate through cells
    for cell in tc_half1:

        # calculate Pearson correlation coefficient between the two tuning curves
        pear_corr = scipy.stats.pearsonr(tc_half1[cell], tc_half2[cell])

        # append info to arrays
        data['sessionName'].append(session)
        data['cellID'].append(np.int64(cell)+1)
        data['pearsonR'].append(pear_corr[0])

    return pd.DataFrame(data) # convert data into dataframe

def compute_control_tuning_curve_correlations(session):
    """
    TODO: Add catch if tuning curve files don't exist
    Computes Pearson correlation coefficient between the time-reversed angular tuning curves split into the first and
    second half of the epoch.
    :param session: (str)
    :return:
    """
    tuning_curve_dir = os.path.join(results_dir, session, "Analysis")

    data = {"sessionName":[],
            "cellID": [],
            "pearsonR_rev": []}

    # path to tuning curves
    path_half1 = os.path.join(tuning_curve_dir, "tuning_curves_smooth_half1_reverse.csv")
    path_half2 = os.path.join(tuning_curve_dir, "tuning_curves_smooth_half2_reverse.csv")

    # load tuning curves
    tc_half1 = pd.read_csv(path_half1, index_col=0)
    tc_half2 = pd.read_csv(path_half2, index_col=0)

    # iterate through cells
    for cell in tc_half1:

        # calculate Pearson correlation coefficient between the two tuning curves
        pear_corr = scipy.stats.pearsonr(tc_half1[cell], tc_half2[cell])

        # append info to arrays
        data['sessionName'].append(session)
        data['cellID'].append(np.int64(cell)+1)
        data['pearsonR_rev'].append(pear_corr[0])

    return pd.DataFrame(data) # convert data into dataframe
def compute_all_tuning_curve_correlations(control, save, replace_data=True, print_progress=False, save_path=None):
    """

    :param control:
    :param save:
    :param replace_data:
    :param save_path:
    :return:
    """
    sessions = get_sessions()
    cell_metrics = load_cell_metrics()

    # Collect correlation data for all sessions
    all_corrs = pd.DataFrame()

    for session in sessions:
        if print_progress:
            print(session)

        if control:
            corrs = compute_control_tuning_curve_correlations(session)
        else:
            corrs = compute_tuning_curve_correlations(session)

        all_corrs = pd.concat([all_corrs, corrs], ignore_index=True)

     # Depending on control or not, column name is different
    if control:
        new_col_name = 'pearsonR_rev'
    else:
        new_col_name = 'pearsonR'

    # merge correlation dataframe into cell metrics
    merged_df = merge_cell_metrics(all_corrs, on=['sessionName', 'cellID'], new_col=new_col_name, replace_existing_data=replace_data, save=save, save_path=save_path)

    return merged_df

def compute_hd_info(data, session):
    """
    TODO: Change the name of the column "hdInfo5"
    Computes the mutual info of the angular tuning curves and the head direction
    for a given session
    :param data: Pynapple object holding the data
    :param session: (str) session name
    :return:
    """

    # load existing tuning curve file
    tuning_curve_path = os.path.join(results_dir, session, "Analysis", "tuning_curves_smooth.csv")
    tuning_curves = pd.read_csv(tuning_curve_path, index_col=0)

    # extract data
    wake_epoch = remove_na(get_wake_square_high_speed_ep(data), data['head-direction'])
    angle_wake = data['head-direction'].restrict(wake_epoch)
    num_cells = len(tuning_curves.columns)

    # calculate hd info
    hd_info = nap.compute_1d_mutual_info(tuning_curves, angle_wake, minmax=(0, 2 * np.pi))

    # fill dataframe
    # note the reason data is loaded this way is to be able to merge into cell metrics file
    # in upstream analysis
    hd_info_data = {"sessionName": [session for _ in range(num_cells)],
            "cellID": (tuning_curves.columns.to_numpy().astype(int)+1).tolist(),
            "hdInfo5": hd_info.values.flatten().tolist()}

    return pd.DataFrame(hd_info_data)


def compute_control_hd_info(data, session):

    """
    TODO: Change the name of the column "hdInfo5_rev"
    Computes the mutual info of the angular tuning curves and the head direction
    for a given session
    :param data: Pynapple object holding the data
    :param session: (str) session name
    :return:
    """

    # load existing tuning curve file
    tuning_curve_path = os.path.join(results_dir, session, "Analysis", "tuning_curves_smooth_reverse.csv")
    tuning_curves = pd.read_csv(tuning_curve_path, index_col=0)

    # extract data
    wake_epoch = remove_na(get_wake_square_high_speed_ep(data), data['head-direction'])
    angle_wake = data['head-direction'].restrict(wake_epoch)
    angle_wake_reverse = time_reverse_feature(angle_wake)
    num_cells = len(tuning_curves.columns)

    # calculate hd info
    hd_info_rev = nap.compute_1d_mutual_info(tuning_curves, angle_wake_reverse, minmax=(0, 2 * np.pi))

    # fill dataframe
    # note the reason data is loaded this way is to be able to merge into cell metrics file
    # in upstream analysis
    hd_info_data = {"sessionName": [session for _ in range(num_cells)],
            "cellID": (tuning_curves.columns.to_numpy().astype(int) + 1).tolist(),
            "hdInfo5_rev": hd_info_rev.values.flatten().tolist()}

    return pd.DataFrame(hd_info_data)

def compute_all_hd_info(control, save, replace_data=True, print_progress=False, save_path=None):
    """

    :param control:
    :param save:
    :param replace_data:
    :param print_progress:
    :param save_path:
    :return:
    """

    sessions = get_sessions()
    cell_metrics = load_cell_metrics()

    # collect hd info for all sessions
    all_hd_info = pd.DataFrame()

    for session in sessions:
        if print_progress:
            print(session)

        data = load_data(session, remove_noise=False, lazy_loading=False)

        if control:
            hd_info = compute_control_hd_info(data, session)
        else:
            hd_info = compute_hd_info(data, session)

        all_hd_info = pd.concat([all_hd_info, hd_info], ignore_index=True)

    # Depending on whether control or not, column name is different
    if control:
        new_col_name = 'hdInfo5_rev'

    else:
        new_col_name = 'hdInfo5'

    # merge hdInfo dataframe into cell metrics
    merged_df = merge_cell_metrics(all_hd_info, on=['sessionName', 'cellID'], new_col=new_col_name, replace_existing_data=replace_data, save=save, save_path=save_path)

    return merged_df

def compute_and_save_hd_info(data, session):
    """
    TODO: Bug here, returns array as 0
    Computes HD info for one session and saves the file
    :return:
    """

    # load existing tuning curve file
    tuning_curve_path = os.path.join(results_dir, session, "Analysis", "tuning_curves_smooth.csv")
    tuning_curves = pd.read_csv(tuning_curve_path, index_col=0)

    # extract data
    wake_epoch = get_wake_square_high_speed_ep(data)
    angle_wake = data['head-direction'].restrict(wake_epoch)

    # calculate hd info
    hd_info = nap.compute_1d_mutual_info(tuning_curves,angle_wake)

    # saving file
    save_path = os.path.join(results_dir, session, "Analysis")
    ensure_dir_exists(save_path)
    hd_info.to_csv(os.path.join(save_path, "hd_info.csv"))

    return hd_info

def add_hd_info_to_cell_metrics():
    """
    Reads HD Info files and merges info to cell metrics file
    :return:
    """

    sessions = get_sessions()

    for session in sessions:
        path_to_hd_info = os.path.join(results_dir, session, "Analysis", "hd_info.csv")
        hd_info = pd.read_csv(path_to_hd_info)

def plot_angular_tuning_curves(session, save, show=True, eps=False):
    """
    This function plots all the tuning curves of all neurons in one specified session.
    You can choose whether to save the plots, show the plots, and whether you want
    the figures to be saved as an .eps file.
    :param session: (str) name of session
    :param save: (bool) whether to save the figures
    :param show: (bool) whether to display the figures
    :param eps: (bool) saves figures as .eps if true. Otherwise saved as .png
    :return:
    """

    session_path = os.path.join(results_dir, session) # where session analysis data is stored
    tuning_curve_path = os.path.join(session_path, "Analysis", "tuning_curves_smooth.csv")

    # load csv
    tuning_curves = pd.read_csv(tuning_curve_path, index_col=0)

    # iterate through cells of session
    for cell in tuning_curves:
        # get cell info
        cellID = int(cell) + 1 # index starts at 0, cell number starts at 1. tuning curves use index as column names
        cell_params = get_cell_parameters(session, cellID)
        cell_type = get_cell_type(session, cellID)

        if cell_type is None:
            continue # noise, don't plot this cell

        # plotting
        palette = sns.color_palette()
        colors = {'hd': palette[0], 'fs': palette[1], 'nhd': palette[3], 'other': 'grey'}

        fig, axs = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))

        fig.suptitle(
            session + f' Neuron #: {str(cellID)}, hdInfo: {str(cell_params["hdInfo5"].values)}, Pcorr: {str(cell_params["pearsonR"].values)}')
        axs.plot(tuning_curves[cell], linewidth=5, color=colors[cell_type])
        axs.set_ylim(bottom=0)
        axs.grid(linestyle='dotted')
        axs.tick_params(axis='x', labelsize=13, pad=5)
        axs.set_yticks([])


        # save
        if save:
            # save figure to respective folder depending on cell type
            save_path = os.path.join(session_path, "Figures", "Tuning_curves", cell_type)
            ensure_dir_exists(save_path)
            if eps:
                plt.savefig(os.path.join(save_path,  str(cellID)), dpi=500, format='eps')

            else:
                plt.savefig(os.path.join(save_path,  str(cellID)), dpi=500)

        if show:
            plt.show()
        plt.close()


def plot_all_angular_tuning_curves(save, show=True, eps=False):
    """
        This function plots all the tuning curves of all neurons in all sessions.
        You can choose whether to save the plots, show the plots, and whether you want
        the figures to be saved as an .eps file.
        :param save: (bool) whether to save the figures
        :param show: (bool) whether to display the figures
        :param eps: (bool) saves figures as .eps if true. Otherwise, saved as .png
        :return:
    """

    sessions = get_sessions()

    for session in sessions:
        plot_angular_tuning_curves(session, save, show=show, eps=eps)

