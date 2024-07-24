import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import pandas as pd
from plotting import plot_head_direction_tuning


#%%
# Now you can call the function
path = r"C:\Users\smahal1\OneDrive - McGill University\000939\sub-A3707\sub-A3707_behavior+ecephys.nwb"
data = nap.load_file(path)
print(data)

spikes = data['units']
epochs = data['epochs']
print(spikes)
print(epochs)

# get behavior data
wake_ep = data['epochs']['wake_square']
angle = data['head-direction']
position = data['position']
speed = np.hstack([0,np.linalg.norm(np.diff(position.d, axis=0),axis=1)/(position.t[1]-position.t[0])])
speed = nap.Tsd(t=position.t, d=speed)

# restrict to epoch
spikes = spikes.restrict(wake_ep)
angle = angle.restrict(wake_ep)
position = position.restrict(wake_ep)
speed = speed.restrict(wake_ep)

# we also have to restrict all the time series to the time of the angles (because the Motive software/Optptrack was turned on after the start of the electrophysiology recording)
wake_ep2 = nap.IntervalSet(start=angle.index[0], end=angle.index[-1])
spikes = spikes.restrict(wake_ep2)
angle = angle.restrict(wake_ep2)
position = position.restrict(wake_ep2)
speed = speed.restrict(wake_ep2)

# choose neurons that are HD and with higher than 1 Hz firing
threshold_hz = 1 
spikes = spikes.getby_threshold("rate", threshold_hz)
print(f"The number of cells with firing rates higher than 1 Hz are {len(spikes)}")
print(f"number  of excitatory neurons are {np.sum(spikes['is_excitatory'])}")
print(f"number  of HD neurons are {np.sum(spikes['is_head_direction'])}")
print(f"number  of FS neurons are {np.sum(spikes['is_fast_spiking'])}")

# compute tuning curves
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=angle, nb_bins=61, minmax=(0, 2 * np.pi)
)

#%%
index_HD = spikes[spikes["is_head_direction"] == 1].index 
# filter neurons for HD
spikes_HD = spikes[index_HD]
tuning_curves_HD = tuning_curves.loc[:, index_HD]

plot_window_start = 3430
plot_window_end = 3490
fig = plot_head_direction_tuning(tuning_curves_HD, spikes_HD, angle, threshold_hz=1, start=plot_window_start, end=plot_window_end)

