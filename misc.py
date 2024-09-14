import configparser
import os
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
    cell_metrics_path = os.path.join(cell_metrics_dir, 'cell_metricsA37.csv')

    return data_dir, results_dir, cell_metrics_dir, cell_metrics_path
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