import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import warnings
from functions import *
from angular_tuning_curves_old import *
import configparser
#import workshop_utils
#import nemos as nmo
from sklearn.model_selection import GridSearchCV
from cell_metrics import create_cell_metrics
warnings.filterwarnings("ignore")

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# make cell metrics (data)
# figure

# Create cell metrics file
create_cell_metrics()

