import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from statannotations.Annotator import Annotator

import numpy as np



# Load the data from the CSV file
data = pd.read_csv('../data_tables/synaptic_connections_wake.csv')
allcells = pd.read_csv('../data_tables/all_CellTypes_fromMonoSyn_27datasets_Full_wake.csv')

# Replace 'Unknown' with 'lowHD' based on the provided conditions
IDs_ro_changetoHDlow =data.loc[(data['PreSynaptic Cell Type'] == 'Unknown') & (data['PreSynaptic classification_FR'] < 16) & (data['PreSynaptic T2P'] > 0.4) & (data['PreSynaptic HD info'] < 0.2), 'PreSynaptic Index'].values
IDs_ro_changetoHDlow2 = data.loc[(data['PostSynaptic Cell Type'] == 'Unknown') & (data['PostSynaptic classification_FR'] < 16) & (data['PostSynaptic T2P'] > 0.4) & (data['PostSynaptic HD info'] < 0.2),'PostSynaptic Index'].values


allcells.loc[allcells['Neuron Index'].isin(IDs_ro_changetoHDlow),'Cell type'] = 'lowHD'
allcells.loc[allcells['Neuron Index'].isin(IDs_ro_changetoHDlow2),'Cell type'] = 'lowHD'


data.loc[(data['PreSynaptic Cell Type'] == 'Unknown') & (data['PreSynaptic classification_FR'] < 16) & (data['PreSynaptic T2P'] > 0.4) & (data['PreSynaptic HD info'] < 0.2), 'PreSynaptic Cell Type'] = 'lowHD'
data.loc[(data['PostSynaptic Cell Type'] == 'Unknown') & (data['PostSynaptic classification_FR'] < 16) & (data['PostSynaptic T2P'] > 0.4) & (data['PostSynaptic HD info'] < 0.2), 'PostSynaptic Cell Type'] = 'lowHD'

# remove 'unknown' type and remove synaptic connections that are basically caused by common input
filtered_unknown_data = data[(data['PreSynaptic Cell Type'] != 'Unknown') & (data['PostSynaptic Cell Type'] != 'Unknown')]
filtered_unknown_latency_data = filtered_unknown_data[filtered_unknown_data['Latency'] > 0.0005]

# the final clean dataset
subset_data = filtered_unknown_latency_data[(filtered_unknown_latency_data['PreSynaptic Cell Type'].isin(['HD', 'lowHD'])) & (filtered_unknown_latency_data['PostSynaptic Cell Type'] == 'FS')]


session_names = subset_data['Session Name'].unique()
# Generate and save figures for all sessions
for session_name in session_names:
    output_filename = plot_neuron_network(session_name, allcells, subset_data)
    print(f"Saved figure: {output_filename}")




def plot_neuron_network(session_name,allcells,data): 
    subset_session_data = data[data['Session Name'] == session_name]
    all_cells_session = allcells[allcells['Session Name'] == session_name].reset_index(drop=True)
    all_cells_session.dropna(inplace=True)
    all_cells_session = all_cells_session[all_cells_session['Cell type'] != 'Unknown'].reset_index(drop=True)
    
    # Create a mapping from neuron index to jittered x positions
    central_x_position = 100
    x_jitter = np.random.uniform(-10, 10, len(all_cells_session))
    
    # Plotting all neurons of the session
    plt.figure(figsize=(5, 20))
    
    # Remove background grid
    plt.grid(False)
    
    # Colors for cell types
    palette = [sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[3]]
    color_dict = {'HD': palette[0], 'lowHD': palette[2], 'FS': palette[1]}
    
    for index, row in all_cells_session.iterrows():
        neuron_x_position = central_x_position + x_jitter[index]
        plt.scatter(neuron_x_position, row['Cell Depth'], color=color_dict[row['Cell type']], s=300, label=row['Cell type'])
    
    # Plot connections
    for index, row in subset_session_data.iterrows():
        pre_x = central_x_position + x_jitter[all_cells_session[all_cells_session['Neuron Index'] == row['PreSynaptic Index']].index[0]]
        pre_y = row['PreSynaptic Cell Depth']
        post_x = central_x_position + x_jitter[all_cells_session[all_cells_session['Neuron Index'] == row['PostSynaptic Index']].index[0]]
        post_y = row['PostSynaptic Cell Depth']
        
        plt.plot([pre_x, post_x], [pre_y, post_y], 'k-', alpha=0.7, linewidth=4)
    
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 1),fontsize=20)
    
    plt.xlabel('Jittered X Position', fontsize=20)
    plt.ylabel('Cell Depth', fontsize=20)
    plt.xticks([],fontsize=20)
    plt.yticks(fontsize=20)
    output_filename = f'Network{session_name}'
    plt.title(session_name, fontsize=20)
    plt.savefig(output_filename+'.png',format='png', bbox_inches='tight')
    plt.savefig(output_filename+'.eps', format='eps', bbox_inches='tight')
    
    #plt.show()
    return output_filename