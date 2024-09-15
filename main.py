import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import warnings
from functions import *
from figures import *
import configparser
#import workshop_utils
#import nemos as nmo
from sklearn.model_selection import GridSearchCV
import cell_metrics
warnings.filterwarnings("ignore")

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# Create cell metrics file
cell_metrics.create_cell_metrics()

# Load cell metrics file
cell_metrics = load_cell_metrics()

# Figures
plot_HD_info_distribution(cell_metrics=cell_metrics, save=False)
plot_correlation_vs_hd_info(cell_metrics=cell_metrics, save=False)
plot_correlation_distribution(cell_metrics=cell_metrics, cell_type='hd', save=False) # HD
plot_correlation_distribution(cell_metrics=cell_metrics, cell_type='nhd', save=False) # Low HD
plot_correlation_distribution(cell_metrics=cell_metrics, cell_type='fs', save=False) # FS
