#%%
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import os 
import glob
import sys
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
import time

# %%
def plot_train_logs(train_fpaths, valid_fpaths, network_names):
    train_dfs = [pd.read_csv(path) for path in train_fpaths]
    valid_dfs = [pd.read_csv(path) for path in valid_fpaths]

    train_losses = [df['Loss'].values for df in train_dfs]
    valid_metrics = [df['Metric'].values for df in valid_dfs]
    train_epochs = [np.arange(len(train_loss))+1 for train_loss in train_losses]
    valid_epochs = [2*(np.arange(len(valid_metric))+1) for valid_metric in valid_metrics]
    min_losses = [np.min(train_loss) for train_loss in train_losses]
    min_losses_epoch = [np.argmin(train_loss) + 1 for train_loss in train_losses]
    max_dscs = [np.max(valid_metric) for valid_metric in valid_metrics]
    max_dscs_epoch = [2*(np.argmax(valid_metric)+1) for valid_metric in valid_metrics]
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)

    for i in range(len(train_losses)):
        ax[0].plot(train_epochs[i], train_losses[i], linewidth=3)
        ax[1].plot(valid_epochs[i], valid_metrics[i], linewidth=3)
                
    legend_labels_trainloss = [f"{network_names[i]}; Min loss: {round(min_losses[i], 4)} ({min_losses_epoch[i]}|{len(train_epochs[i])})" for i in range(len(network_names))]
    legend_labels_validdice = [f"{network_names[i]}; Max DSC: {round(max_dscs[i], 4)} ({max_dscs_epoch[i]})" for i in range(len(network_names))]
    
    for i in range(len(train_losses)):
        ax[0].plot(min_losses_epoch[i], min_losses[i], '-o', color='red', label='')
        ax[1].plot(max_dscs_epoch[i], max_dscs[i], '-o', color='red', label='')

    ax[0].legend(legend_labels_trainloss, fontsize=16, loc='upper left', bbox_to_anchor=(-0.01, -0.1))
    ax[1].legend(legend_labels_validdice, fontsize=16, loc='upper left', bbox_to_anchor=(-0.01, -0.1))
    ax[0].set_title('Train loss', fontsize=25)
    ax[1].set_title('Valid DSC', fontsize=25)
    ax[0].set_ylabel('Dice loss', fontsize=20)
    ax[1].set_ylabel('Dice score', fontsize=20)
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()


network = ['unet1']*9
centers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
expcodes = [f'loco{c}_trn64_val128_WDL' for c in centers]#,'locoC', 'locoD', 'locoE']#, 'locoF', ] #['locoA','locoB','locoC','locoD','locoE', 'locoF']
# network = ['unet1']
# # centers = ['A', 'A']#, 'C', 'D', 'E', 'F', 'G', 'H', 'I']
# expcodes = ['onlyE_trn64_val128_WDL']
# expcodes = [f'locoA_randcrop64_WDL', 'locoA_trn64_val128_WDL', 'locoE_randcrop64_WDL', 'locoE_trn64_val128_WDL', 'locoE_trn64_val128_WDL_origspacing']
legend_lbls = [f'{n}_{e}' for n, e in zip(network, expcodes)]
experiment_code = [f"{network[i]}_{expcodes[i]}" for i in range(len(network))]
save_logs_dir = os.path.join(RESULTS_FOLDER, 'logs')
save_logs_folders = [os.path.join(save_logs_dir, experiment_code[i]) for i in range(len(experiment_code))]
train_fpaths = [os.path.join(save_logs_folders[i], 'trainlog_gpu0.csv') for i in range(len(save_logs_folders))]
valid_fpaths = [os.path.join(save_logs_folders[i], 'validlog_gpu0.csv') for i in range(len(save_logs_folders))]
plot_train_logs(train_fpaths, valid_fpaths, legend_lbls)
# %%
