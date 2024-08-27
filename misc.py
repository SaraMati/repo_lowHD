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
    project_dir = config['Directories']['project_dir']
    cell_metrics_path = config['Paths']['cell_metrics_path']
    results_dir = config['Directories']['results_dir']

    return cell_metrics_path, data_dir, project_dir, results_dir

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