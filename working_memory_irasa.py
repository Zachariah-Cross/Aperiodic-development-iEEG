
"""
Script for applying IRASA and peak oscillation detection functions
Working memory task
@author: Zachariah R. Cross, November 2022
"""

# %%
##############################################################################
# Load modules
##############################################################################

# modules required for IRASA
import os
import mne
import yasa
import glob
import numpy as np
import pandas as pd
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt

# modules required for peak oscillation detection
from scipy.signal import find_peaks

# set parameters for plotting figures
%matplotlib qt
sns.set(style='white', font_scale=1.2)

# %%
##############################################################################
# Prepare files and data frames
##############################################################################

# get list of epoched working memory data files
epoch_files = glob.glob('/Volumes/Zach_HD/ieeg_aperiodic/epoched_data/working_memory/*_new.mat-epo.fif.gz')

# loop through each epoched file
for epoch_name in epoch_files:
    
    # extract participant ID number
    head_tail = os.path.split(epoch_name)
    subj = head_tail[1]
    print("processing " + subj)
    
    # load the epoch file
    epochs = mne.read_epochs(epoch_name,preload=True)

    # extract information from epoched data
    data = epochs.get_data(tmin = 0, tmax = 3) # 0 to 3s after stim onset
    sf = epochs.info['sfreq'] # set sf as the sampling frequency
    chan = epochs.ch_names # define channel as ch_names
    
    # initialise data frames
    dfs = []
    df_osc = []
    psd_total = []
    df_aperiodic = []

# %%
##############################################################################
# Run IRASA
##############################################################################

    # loop through each epoch to estimate trial-level aperiodic activity
    for idx in range(data.shape[0]):
        freqs, psd_aperiodic, psd_osc, fit_params = yasa.irasa(data[idx, :, :], sf, 
                                                               ch_names = chan, 
                                                               band = (1, 60), 
                                                               win_sec = 1, 
                                                               return_fit = True)
        
        # generate data frames required for later analysis
        fit_params.insert(loc=0, column="epoch", value=idx) # add epoch column
        fit_params['subj'] = subj # add subject column
        fit_params['task'] = "wm" # add task column
        dfs.append(fit_params) # append each epoch data frame
        
        # save IRASA output file for each subject
        # fit_params.to_csv('irasa_data/' + subj + '_aperiodic_wm.csv', header = True)
        
        # # create larger IRASA data frame with each participant
        # df_export = pd.DataFrame(fit_params)
        # if op.isfile('aperiodic_wm.csv'):
        #     df_export.to_csv('aperiodic_wm.csv', sep=',', mode='a', header=False)
        # else:
        #     df_export.to_csv('aperiodic_wm.csv', sep=',', mode='a', header=True)
            
        # append the psd arrays
        psd_total.append(psd_osc)
            
        # average psd across epochs for peak detection
        test = np.average(psd_total, axis = 0)
        
        # generate data frame for residual oscillatory activity
        df_osc_epoch = pd.DataFrame(psd_osc)
        df_osc_epoch.insert(loc=0, column="epoch", value=idx)
        df_osc_epoch['subj'] = subj
        df_osc_epoch['task'] = "wm"
        df_osc.append(df_osc_epoch)
        
        # generate data frame for residual aperiodic psd
        df_aperiodic_epoch = pd.DataFrame(psd_aperiodic)
        df_aperiodic_epoch.insert(loc=0, column="epoch", value=idx)
        df_aperiodic_epoch['subj'] = subj
        df_aperiodic_epoch['task'] = "wm" # add task column
        df_aperiodic.append(df_aperiodic_epoch)
        
    # concatenate each epoch structure
    df = pd.concat(dfs)
    df_osc_df = pd.concat(df_osc)
    df_aperiodic_df = pd.concat(df_aperiodic)
    
    # # save psd aperiodic and psd oscillatory output files for each subject
    # df_osc_df.to_csv('power_psd_data/' + subj + '_power_psd_wm.csv', header = True)
    # df_aperiodic_df.to_csv('aperiodic_psd_data/' + subj + '_aperiodic_psd_wm.csv', header = True)

    # # plot the aperiodic component on a linear-log scale
    # plt.plot(freqs, psd_aperiodic[2, :], 'k', lw=2.5)
    # plt.fill_between(freqs, psd_aperiodic[2, :], cmap='Spectral')
    # plt.xlim(1, 60)
    # plt.yscale('log')
    # sns.despine()
    # plt.title('Aperiodic component at ' + chan[2], fontsize = 15)
    # plt.xlabel('Frequency [Hz]',fontsize = 20)
    # plt.ylabel('PSD log($uV^2$/Hz)',fontsize = 20)
    # plt.savefig('irasa_figures/' + subj + '_aperiodic_wm.png',dpi = 300, 
    #             bbox_inches='tight');
    # plt.close();
    
    # # and oscillatory component on a linear-linear scale
    # plt.plot(freqs, psd_osc[2, :], 'k', lw=2.5)
    # plt.fill_between(freqs, psd_osc[2, :], cmap='Spectral')
    # plt.xlim(1, 60)
    # sns.despine()
    # plt.title('Oscillatory component at ' + chan[2],fontsize = 15)
    # plt.xlabel('Frequency [Hz]',fontsize = 20)
    # plt.ylabel('PSD log($uV^2$/Hz)',fontsize = 20)
    # plt.savefig('irasa_figures/' + subj + '_oscillatory_wm.png',dpi = 300, 
    #             bbox_inches='tight');
    # plt.close();

# %%
##############################################################################
# Peak oscillation detection
##############################################################################

    # define the frequency range (i.e., x axis)
    x = freqs
     
    # create empty arrays for maxima and minima peaks
    maxima_df = []
    minima_df = []
    
    # loop through each participant
    for s in range(test.shape[0]): # shape refers to channel
        peaks = find_peaks(test[s], height = 0.5, prominence = 3, distance = 2)
        maxima_pos = x[peaks[0]] # list of maxima positions
        maxima_height = peaks[1]['peak_heights'] # list of height of maximas
    
        # find the minima peaks
        psd_2 =  test[s, :] * -1
        minima = find_peaks(psd_2, height = 0.5, prominence = 3, distance = 2)
        min_pos = x[minima[0]] # list of minima positions
        min_height = psd_2[minima[0]] # list of mirrored heights
        
        # append each epoch to larger data frame
        maxima_df.append(maxima_pos)
        minima_df.append(min_pos)
            
        # convert to pandas data frame
        df_max = pd.DataFrame(maxima_df)
        df_min = pd.DataFrame(minima_df)
    
        # add subject and channel index/columns
        df_max['subj'] = subj
        df_min['subj'] = subj
        
        df_max['ch_index'] = df_max.index # channel index
        df_min['ch_index'] = df_min.index # channel index
        
        df_max['task'] = "wm" # task name
        df_min['task'] = "wm" # task name
        
        # create data frame with channel and subject names to append later
        chan_information = pd.DataFrame(chan)
        chan_information['subj'] = subj
        chan_information['ch_index'] = chan_information.index
        chan_information.rename(columns = {0:'ch_name'}, inplace = True)
            
        # plot the maxima and minima
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(x,test[s, :])
        ax.scatter(maxima_pos, maxima_height, color = 'r', s = 15, marker = 'D',
                    label = 'Maxima')
        ax.scatter(min_pos, min_height, color = 'gold', s = 15, marker = 'X',
                    label = 'Minima')
        ax.legend()
        ax.grid()
        plt.title('Oscillatory peaks at ' + str(s), fontsize = 15)
        plt.xlabel('Frequency [Hz]', fontsize = 20)
        plt.ylabel('PSD log($uV^2$/Hz)', fontsize = 20)
        plt.savefig('peak_oscillations/' + subj + '_' + str(s) + '_oscillatory_wm.png', 
                    dpi = 300, bbox_inches='tight');
        plt.close();
                
        # save output file for each subject
        df_max.to_csv('peak_data/' + subj + '_maxima_oscillations_wm.csv', header = True)
        df_min.to_csv('peak_data/' + subj + '_minima_oscillations_wm.csv', header = True)
        
    # save channel index information
    chan_information.to_csv('channel_information_wm.csv', sep=',', mode='a', header=False)
    