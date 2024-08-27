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

# Loading data
session = "A3713-200909a" # session to load
data = load_data(session)
print(data)

spikes = data['units']
epochs = data['epochs']
print(epochs)

# get behavior data
wake_ep = data['epochs']['wake_square']
angle = data['head-direction']
position = data['position']
speed = calculate_speed(position)

# restrict to epoch
spikes = spikes.restrict(wake_ep)
angle = angle.restrict(wake_ep)
position = position.restrict(wake_ep)
speed = speed.restrict(wake_ep)

# compute tuning curves
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=angle, nb_bins=61, minmax=(0, 2 * np.pi)
)

# plot tunning curves

# choose neurons with higher than 1 Hz firing
threshold_hz = 1 
index_keep = spikes[spikes["is_head_direction"] == 1].index 
# filter neurons
tuning_curves = tuning_curves.loc[:, index_keep]
pref_ang = tuning_curves.idxmax().loc[index_keep]


# plot specs
cmap_label="hsv"
cmap = plt.get_cmap(cmap_label)
figsize=(12, 6)
fig = plt.figure(figsize=figsize)

unq_angles = pref_ang.values #np.unique(pref_ang.values)
sorted_angles = np.sort(pref_ang.values)
relative_color_levs = (sorted_angles - sorted_angles[0]) / (sorted_angles[-1] - sorted_angles[0])
n_subplots = len(unq_angles)

for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid(
            (4, n_subplots),
            loc=(3 + i // n_subplots, i % n_subplots),
            rowspan=1,
            colspan=1,
            fig=fig,
            projection="polar",
        )
        ax.fill_between(
            tuning_curves.iloc[:, neu_idx].index,
            np.zeros(len(tuning_curves)),
            tuning_curves.iloc[:, neu_idx].values,
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])

#plt.tight_layout()
plt.show()

# take only the lowHD neurons
spikes_lowHD = spikes[(spikes.is_excitatory & ~spikes.is_head_direction).astype(bool)]
