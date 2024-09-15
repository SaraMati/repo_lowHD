import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import warnings
from functions import *
from angular_tuning_curves import *
import configparser
#import workshop_utils
#import nemos as nmo
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# make cell metrics (data)
# figure

# Loading data
session = "A3713-200909a" # session to load
data = load_data_DANDI_postsub(session)
print(data)
