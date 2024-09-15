"""
Contains functions to replicate figures.

@Author: Selen Calgin
@Date created: 15/09/2024
"""

from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir, results_dir, cell_metrics_dir, cell_metrics_path = config()

def plot_HD_info_distribution(cell_metrics, save, format=None, log_x=True, boundary=True, percentile=99, print_stats=True):
    """
    Plots the distribution of HD info of excitatory cells to show the definition of HD cells and Low HD cells.
    HD cells are cells that have higher HD Info than 99% of control distribution.
    Default is on a log x scale.
    :param cellmetrics: (Pandas DataFrame) Contains the cell metrics data for plotting
    :param save: (bool) Whether to save figure
    :param format: (str) Format of the image. If None, default is PNG
    :param log_x: (bool) Whether to plot on log x scale. Default is True.
    :param boundary: (bool) Whether to plot a boundary between HD and non HD cells
    :param percentile: (int) Indicates which percentile of control HD info will separate HD cells from low HD cells
    :param print_stats: (bool) Whether to print statistical analysis
    :return: None
    """

    # Extracting dataframes of the different cell types
    ex_cells = cell_metrics[cell_metrics['ex'] == 1]

    # Get HD info of excitatory cells
    hd_info = ex_cells['hdInfo']
    hd_info_control = ex_cells['hdInfo_rev']

    # Adjust bins based on whether x log scale
    if log_x:
        bins = compute_log_bins(hd_info, 50)
        control_bins = compute_log_bins(hd_info_control, 50)

    else:
        bins = 50
        control_bins = 50

    # Defining boundary between HD and Low HD cells.
    percentile_boundary = np.percentile(hd_info_control.dropna(), percentile)

    # Plotting
    plt.hist(hd_info_control, bins=control_bins, color='lightgrey', label='Control')
    plt.hist(hd_info, bins=bins, color='mediumorchid', label='Ex.', alpha=0.7)

    # Customizing axes
    if log_x:
        plt.xscale('log')
    plt.tick_params(axis='both', direction='in', pad=5, labelsize=20)

    # Labeling axes
    if log_x:
        plt.xlabel('log10(HD Info)', fontsize=25)
    else:
        plt.xlabel('HD Info (bits/spike)', fontsize=25)
    plt.ylabel('No. cells', fontsize=25)

    # Plotting percentile boundary
    if boundary:
        plt.axvline(x=percentile_boundary, color='black', linestyle='--', alpha=0.5, label='99th \nPerc.')

    plt.xticks([0.01, 0.1,1],[-2,-1,0])

    # Adding legend
    plt.legend(fontsize=15)

    plt.tight_layout()

    if save:
        save_path = os.path.join(results_dir, 'hdInfo_distribution')
        plt.savefig(save_path, dpi=500, format=format)

    plt.show()

    # Statistics
    if print_stats:
        print("Statistical analysis from HD Info Distribution:")
        print(f"The percentile boundary at HD Info: {percentile_boundary}.")

        mean_hd = np.mean(hd_info)
        median_hd = np.median(hd_info)

        print(f"Mean HD: {mean_hd:.2f}, Median HD: {median_hd:.2f}")

        # Wilcoxon analysis comparing HD info distribution with control distrbution
        print(scipy.stats.wilcoxon(hd_info, hd_info_control, alternative='greater'))
        print()
def plot_correlation_vs_hd_info(cell_metrics, save, format=None):
    """
    This plots a scatter plot of the tuning curve correlation vs. the HD Info
    of HD cells, low HD cells, and FS cells.
    :param cell_metrics: (Pandas DataFrame) Contains the cell metrics data for plotting
    :param save: (bool) Whether to save the figure
    :param format: (str) Specifies the image type when saving the figure. If None, will save as PNG.
    :return: None
    """
    # Extracting cell types as separate DataFrames
    hd_cells = cell_metrics[cell_metrics['hd']==1]
    lowhd_cells = cell_metrics[cell_metrics['nhd']==1]
    fs_cells = cell_metrics[cell_metrics['fs']==1]

    # Defining colours
    palette = sns.color_palette()
    colours = {'HD': palette[0], 'FS': palette[1], 'lowHD': palette[3]}

    # Plotting
    plt.scatter(x=hd_cells['hdInfo'], y=hd_cells['pearsonR'], color=colours['HD'], s=20, alpha=0.9, label='HD')
    plt.scatter(x=fs_cells['hdInfo'], y=fs_cells['pearsonR'], color=colours['FS'], s=20, alpha=0.9, label='FS')
    plt.scatter(x=lowhd_cells['hdInfo'], y=lowhd_cells['pearsonR'], color=colours['lowHD'], s=20, alpha=1, label='Low HD')

    # Customizing axes
    plt.xscale('log')
    plt.tick_params(axis='both', direction='in', pad=5, labelsize=20)
    plt.xlabel('log10(HD Info)', fontsize=25)
    plt.ylabel('Correlation', fontsize=25)
    plt.legend(fontsize=18, markerscale=3)
    plt.xticks([0.001, 0.01, 0.1, 1], [-3,-2,-1,0])
    plt.tight_layout()

    # Saving (optional)
    if save:
        plt.savefig(os.path.join(results_dir, "correlation_vs_hdInfo"), dpi=500, format=format)

    plt.show()

def plot_correlation_distribution(cell_metrics, cell_type, save, format=None, print_stats=True):
    """
    Plots the distribution of split tuning curve correlation metrics for a specified cell type.

    The function allows you to visualize the distribution for three different cell types:
    - "hd" (Head Direction Cells)
    - "nhd" (Low Head Direction Cells)
    - "fs" (Fast Spiking Cells)

    Additionally, it offers options to save the plot and perform statistical analysis on the data.

    :param cell_metrics: (pd.DataFrame) DataFrame containing the cell metrics for plotting.
    :param cell_type: (str) Specify the cell type: "hd" for HD Cells, "nhd" for Low HD Cells, or "fs" for fast spiking cells.
    :param save: (bool) If True, the plot will be saved to a file.
    :param format: (str) Format to save the figure in (e.g., "png", "eps"). If None, default is None.
    :param print_stats: (bool) If True, statistical analysis will be printed to the console.
    :return: None
    """
    # Extracting desired cell type
    cells = cell_metrics[cell_metrics[cell_type]==1]

    # Defining label based on cell type
    labels = {'hd': 'HD', 'fs': 'FS', 'nhd': 'Low HD'}

    # Defining colours
    palette = sns.color_palette()
    colours = {'hd': palette[0], 'fs': palette[1], 'nhd': palette[3]}

    # Getting data
    correlation = cells['pearsonR']
    control_correlation = cells['pearsonR_rev']

    # Plotting
    plt.hist(control_correlation, bins=50, color='lightgrey', label='Control')
    plt.hist(correlation, bins=50, color=colours[cell_type], label=labels[cell_type])

    # Customizing axes
    plt.tick_params(axis='both', direction='in', pad=5, labelsize=20)
    plt.xlabel('Correlation', fontsize=25)
    plt.ylabel('No. cells', fontsize=25)
    plt.legend(fontsize=20)
    plt.tight_layout()

    # Saving (optional)
    if save:
        plt.savefig(os.path.join(results_dir,f"correlation_distribution_{cell_type}"), dpi=500, format=format)

    plt.show()

    # Statistical analysis
    if print_stats:
        print(f"Statistical analysis from Correlation Distribution of {labels[cell_type]} Cells:")
        print(f"Wilcoxon test result: {scipy.stats.wilcoxon(correlation, control_correlation, alternative='greater')}")
        print(
            f"Mean correlation: {np.mean(correlation):.2f}, Mean control correlation: {np.mean(control_correlation):.2f}")
        print(
            f"Median correlation: {np.median(correlation):.2f}, Median control correlation: {np.median(control_correlation):.2f}")
        print(
            f"Standard deviation of correlation: {np.std(correlation):.2f}, Standard deviation of control correlation: {np.std(control_correlation):.2f}")
        print()