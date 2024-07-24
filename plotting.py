from typing import Optional

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

def plot_head_direction_tuning(
        tuning_curves: pd.DataFrame,
        spikes: nap.TsGroup,
        angle: nap.Tsd,
        threshold_hz: int = 1,
        start: float = 8910,
        end: float = 8960,
        cmap_label="hsv",
        figsize=(12, 6)
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:

    spikes:
        The spike times.
    angle:
        The heading angles.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------

    """
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.loc[:, index_keep]
    pref_ang = tuning_curves.idxmax().loc[index_keep]
    spike_tsd = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang)

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    unq_angles = np.unique(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (unq_angles[-1] - unq_angles[0])
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid((3, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig)
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")

    ax = plt.subplot2grid((3, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig)
    ax.set_title("Neural Activity")
    for i, ang in enumerate(unq_angles):
        sel = spike_tsd.d == ang
        ax.plot(spike_tsd[sel].t, np.ones(sel.sum()) * i, "|", color=cmap(relative_color_levs[i]), alpha=0.5)
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid((3, n_subplots), loc=(2 + i // n_subplots, i % n_subplots),
                              rowspan=1, colspan=1, fig=fig, projection="polar")
        ax.fill_between(tuning_curves.iloc[:, neu_idx].index, np.zeros(len(tuning_curves)),
                        tuning_curves.iloc[:, neu_idx].values, color=cmap(relative_color_levs[i]), alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
