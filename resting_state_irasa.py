
"""
Script for applying IRASA and peak oscillation detection functions
Resting state recording
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
epoch_files = glob.glob('/Volumes/Zach_HD/ieeg_aperiodic/epoched_data/resting_state/converted/*-epo.fif.gz')

# loop through each epoched file
for epoch_name in epoch_files:
    
    # extract participant ID number
    head_tail = os.path.split(epoch_name)
    subj = head_tail[1]
    print("processing " + subj)
    
    # load the epoch file
    epochs = mne.read_epochs(epoch_name,preload=True)

    # extract information from epoched data
    # data = epochs.get_data(tmin = 0, tmax = 3) # 0 to 3s epochs
    data = epochs.get_data() # run this line for WashU cases
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
        fit_params['task'] = "resting" # add task column
        dfs.append(fit_params) # append each epoch data frame
        
        # save IRASA output file for each subject
        #fit_params.to_csv('irasa_data/' + subj + '_aperiodic_resting.csv', header = True)
        
        # # create larger IRASA data frame with each participant
        # df_export = pd.DataFrame(fit_params)
        # if op.isfile('aperiodic_resting.csv'):
        #     df_export.to_csv('aperiodic_resting.csv', sep=',', mode='a', header=False)
        # else:
        #     df_export.to_csv('aperiodic_resting.csv', sep=',', mode='a', header=True)
            
        # append the psd arrays
        psd_total.append(psd_osc)
            
        # average psd across epochs for peak detection
        test = np.average(psd_total, axis = 0)
        
        # generate data frame for residual oscillatory activity
        df_osc_epoch = pd.DataFrame(psd_osc)
        df_osc_epoch.insert(loc=0, column="epoch", value=idx)
        df_osc_epoch['subj'] = subj
        df_osc_epoch['task'] = "resting"
        df_osc.append(df_osc_epoch)
        
        # generate data frame for residual aperiodic psd
        df_aperiodic_epoch = pd.DataFrame(psd_aperiodic)
        df_aperiodic_epoch.insert(loc=0, column="epoch", value=idx)
        df_aperiodic_epoch['subj'] = subj
        df_aperiodic_epoch['task'] = "resting" # add task column
        df_aperiodic.append(df_aperiodic_epoch)
        
    # concatenate each epoch structure
    df = pd.concat(dfs)
    df_osc_df = pd.concat(df_osc)
    df_aperiodic_df = pd.concat(df_aperiodic)
    
    # save psd aperiodic and psd oscillatory output files for each subject
    # df_osc_df.to_csv('power_psd_data/' + subj + '_power_psd_resting.csv', header = True)
    # df_aperiodic_df.to_csv('aperiodic_psd_data/' + subj + '_aperiodic_psd_resting.csv', header = True)

    # # plot the aperiodic component on a linear-log scale
    # plt.plot(freqs, psd_aperiodic[2, :], 'k', lw=2.5)
    # plt.fill_between(freqs, psd_aperiodic[2, :], cmap='Spectral')
    # plt.xlim(1, 60)
    # plt.yscale('log')
    # sns.despine()
    # plt.title('Aperiodic component at ' + chan[2], fontsize = 15)
    # plt.xlabel('Frequency [Hz]',fontsize = 20)
    # plt.ylabel('PSD log($uV^2$/Hz)',fontsize = 20)
    # plt.savefig('irasa_figures/' + subj + '_aperiodic_resting.png',dpi = 300, 
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
    # plt.savefig('irasa_figures/' + subj + '_oscillatory_resting.png',dpi = 300, 
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
        
        df_max['task'] = "resting" # task name
        df_min['task'] = "resting" # task name
        
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
        plt.savefig('peak_oscillations/' + subj + '_' + str(s) + '_oscillatory_resting.png', 
                    dpi = 300, bbox_inches='tight');
        plt.close();
                
        # save output file for each subject
        df_max.to_csv('peak_data/' + subj + '_maxima_oscillations_resting.csv', header = True)
        df_min.to_csv('peak_data/' + subj + '_minima_oscillations_resting.csv', header = True)
        
    # save channel index information
    chan_information.to_csv('channel_information_rest.csv', sep=',', mode='a', header=False)
    
# %%
##############################################################################
# Generate PSD plot with rest vs task slope and oscillations
##############################################################################


import os
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Function to compute PSD and average across channels
def compute_average_psd(data, sfreq, fmin=1, fmax=60, n_fft=2048):
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft, average='mean'
    )
    # psds has shape (n_epochs, n_channels, n_freqs)
    # Average over epochs and channels
    psd_mean = np.mean(psds, axis=(0, 1))
    return psd_mean, freqs

# Function to resample data to a common sampling frequency
def resample_data(data, orig_sfreq, target_sfreq):
    return mne.filter.resample(data, up=target_sfreq, down=orig_sfreq, npad="auto")

# Function to estimate the 1/f slope
def estimate_1f_slope(freqs, psd):
    log_freqs = np.log10(freqs)
    log_psd = np.log10(psd)
    slope, intercept, _, _, _ = stats.linregress(log_freqs, log_psd)
    return slope, intercept

# Get working memory data set
working_memory = glob.glob('/Volumes/Zach_HD/ieeg_aperiodic/epoched_data/working_memory/CH16_new.mat-epo.fif.gz')

data_wm = []
sf_wm = None

# Loop through each epoched file
for wm_name in working_memory:
    head_tail = os.path.split(wm_name)
    subj = head_tail[1]
    print("processing " + subj)
    
    # Load the epoch file
    epochs_wm = mne.read_epochs(wm_name, preload=True)
    
    # Extract information from epoched data
    data_wm.append(epochs_wm.get_data(tmin=0, tmax=3)) # 0 to 3s after stim onset
    sf_wm = epochs_wm.info['sfreq']

# Concatenate all working memory data along the epoch dimension
data_wm = np.concatenate(data_wm, axis=0)

# Get resting state data set
resting = glob.glob('/Volumes/Zach_HD/ieeg_aperiodic/epoched_data/resting_state/converted/done/CH16_rest.mat-epo.fif.gz')

data_rest = []
sf_rest = None

# Loop through each epoched file
for rest_name in resting:
    head_tail = os.path.split(rest_name)
    subj = head_tail[1]
    print("processing " + subj)
    
    # Load the epoch file
    epochs_rest = mne.read_epochs(rest_name, preload=True)
    
    # Extract information from epoched data
    data_rest.append(epochs_rest.get_data(tmin=0, tmax=3)) # 0 to 3s after stim onset
    sf_rest = epochs_rest.info['sfreq']

# Concatenate all resting state data along the epoch dimension
data_rest = np.concatenate(data_rest, axis=0)

# Check if sampling frequencies are different and resample if necessary
if sf_wm != sf_rest:
    target_sfreq = min(sf_wm, sf_rest)  # Resample to the lower of the two sampling frequencies
    if sf_wm > target_sfreq:
        data_wm = resample_data(data_wm, sf_wm, target_sfreq)
        sf_wm = target_sfreq
    if sf_rest > target_sfreq:
        data_rest = resample_data(data_rest, sf_rest, target_sfreq)
        sf_rest = target_sfreq

# Compute average PSD for working memory and resting state data
psd_wm, freqs_wm = compute_average_psd(data_wm, sf_wm)
psd_rest, freqs_rest = compute_average_psd(data_rest, sf_rest)

# Ensure that the frequency arrays are identical
assert np.array_equal(freqs_wm, freqs_rest), "Frequency arrays do not match."

# Estimate the 1/f slope for working memory and resting state
slope_wm, intercept_wm = estimate_1f_slope(freqs_wm, psd_wm)
slope_rest, intercept_rest = estimate_1f_slope(freqs_rest, psd_rest)

# Compute the 1/f fit lines
fit_wm = 10**(intercept_wm + slope_wm * np.log10(freqs_wm))
fit_rest = 10**(intercept_rest + slope_rest * np.log10(freqs_rest))

# Plot the PSD with 1/f slopes
plt.figure(figsize=(7, 7))
plt.semilogy(freqs_wm, psd_wm, color='r', label='Working Memory PSD', linewidth=2)
plt.semilogy(freqs_wm, fit_wm, linestyle='--', color='r', label='Working Memory 1/f Slope', linewidth=2)
plt.semilogy(freqs_rest, psd_rest, color='gray', label='Resting State PSD', linewidth=2)
plt.semilogy(freqs_rest, fit_rest, linestyle='--', color='gray', label='Resting State 1/f Slope', linewidth=2)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (uV^2/Hz)')
plt.xlim([1, 60])
plt.legend(fontsize='small')
plt.show()

