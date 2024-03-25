# Seal-Sleep-Classifier
Original code: https://github.com/jmkendallbar/Seal-Sleep-Capstone
Code for seal sleep classifier for use in EcoViz+AI Workshop.

This GitHub repository is meant to serve as a demonstration for how Seal ECG, EEG, body movement, and pressure features can be used in a machine learning model to automate seal sleep state prediction.

# Directory structure

```
├── LICENSE
├── README.md
├── data
│   ├── 01_processed-data                                   (this directory is not committed to GitHub and must be downloaded separately)
│   │   ├── test12_Wednesday_05_ALL_PROCESSED.edf           All sensor data from Wednesday the seal
│   │   └── test12_Wednesday_05_DAY1_PROCESSED.edf          Day 1 sensor data from Wednesday the seal
│   ├── 02_annotated-data
│   │   └── test12_Wednesday_06_Hypnogram_JKB_1Hz.csv       1Hz sleep state hypnogram for Wednesday the seal
├── helpful-figs                                            Helpful figures used in the notebook to explain features
│   ├── light-sleep-spectral-power.png
│   ├── rem.png
│   ├── slow-wave-1.png
│   ├── slow-wave-2-spectral-power.png
│   └── slow-wave-2.png
├── notebooks
│   ├── 00_initial_feature_extraction.ipynb                 Notebook containing original feature extraction, with manually coded spectral powers
│   ├── 01_initial_models.ipynb                             Initial scikit-learn models using on the initial features 
│   ├── 01a_YASA_feature_extraction.ipynb                   Testing YASA's human sleep model on seals out-of-the-box
│   ├── 02_advanced_feature_generation_and_models.ipynb     More advanced feature generation and model building using adjusted features from YASA as well as movement and pressure data
│   └── 03_recursive_feature_elimination.ipynb              Recursive feature elimination on step 02 models
├── requirements.txt                                        python package requirements
└── src
    ├── feature_extraction.py                               Methods for features extraction
    └── feature_generation.py                               Wrapper for feature extraction that provides functionality for command-line feature extraction and a function to generate all the features from a specified .edf file
```

# Prerequisites
- Python 3.10 (tested with 3.10.9)
- [Jupyter notebooks](https://jupyter.org/install)
- Seal Sleep data files, downloaded from: [**FigShare Seal Sleep Project**](https://figshare.com/account/projects/199498/articles/25464379)

# Notebooks

#### 00_initial_feature_extraction.ipynb
Initial notebook used for feature extraction, not the prettiest of notebooks, but walks through heart rate extraction from ECG using peak detection. Also goes through each of the features and plots their distributions to get a first-pass glimpse at the feature space.

#### 01_initial_features_and_models.ipynb
Uses the features created in *00_initial_feature_extraction* to create a few rudimentary scikit-learn machine learning models, including a Support Vector Machine Classifier (SVC), K Nearest Neighbors Classifier (KNN), and Random Forest Classifier (RFC). Also performs grid searching on each of these estimators to find ideal or close-to-ideal parameterizations of these models.

#### 01a_YASA_feature_extraction.ipynb
Has a simple demonstration of applying the YASA sleep staging algorithm built for humans onto one seal. While the performance is not the best, many of the features implemented by YASA were used for this project, and in some cases their code was used as well (but adjusted for seals). Note that YASA requires a LOT of memory and the kernel often crashes for me if I have other notebooks running or too many Google Chrome tabs open.

#### 02_advanced_feature_generation_and_models.ipynb
Uses the (semi) finalized feature extraction code in ***feature_extraction.py*** and ***feature_generation.py*** to generate all the available features for sleep detection created so far. Also creates a RFC using the grid-searched parameters from ***01_initial_features_and_models***, and plots the predictions against the true labels and features for a few naps

#### 03_recursive_feature_elimination.ipynb
Performs recursive feature elimination using the model from ***02_advanced_feature_generation_and_models.ipynb*** to explore which features are the most informative for seal sleep prediction.

# Script and Command-line Usage
### feature_extraction.py
You can look at the function code in *src* to see optional arguments and their defaults

#### get_features_yasa
get_features_yasa(a, start_index, end_index, freq=500): Gets the EEG features from a 500 Hz (or any frequency, just make sure you change freq=500 to whatever your sample frequency is). Uses overlapping epochs with size of 30 seconds and 50% overlap (so epoch centers are every 15 seconds):
    - yasa_time                 datetime, included to make sure yasa features line up with other features
    - yasa_eeg_std              eeg epoch standard deviation
    - yasa_eeg_iqr              eeg interquartile range
    - yasa_eeg_skew             eeg skew
    - yasa_eeg_kurt             eeg kurtosis
    - yasa_eeg_nzc              eeg zero crossings
    - yasa_eeg_hmob             eeg hjorth mobility
    - yasa_eeg_hcomp            eeg hjorth complexity
    - yasa_eeg_sdelta           eeg slow delta power (0.4 - 1 Hz)
    - yasa_eeg_fdelta           eeg fast delta power (1 - 4 Hz)
    - yasa_eeg_theta            eeg theta power (4 - 8 Hz)
    - yasa_eeg_alpha            eeg alpha power (8 - 12 Hz)
    - yasa_eeg_sigma            eeg sigma power (12 - 16 Hz)
    - yasa_eeg_beta             eeg beta power (16 - 30 Hz)
    - yasa_eeg_dt               delta / theta power ratio
    - yasa_eeg_ds               delta / sigma power ratio
    - yasa_eeg_db               delta / beta power ratio
    - yasa_eeg_at               alpha / theta power ratio
    - yasa_eeg_abspow           eeg absolute power (0.4 - 30 Hz)
    - yasa_eeg_perm             eeg permutation entropy
    - yasa_eeg_higuchi          eeg higuchi fractal dimension
    - yasa_eeg_petrosian        eeg petrosian fractal dimension

#### get_rolling_band_power_welch
get_rolling_band_power_welch(a, start_index, end_index, freq_range=(0.5, 4), freq=500): Gets the power of the specified frequency range using Welch's method for calculating power spectral density and Simpson's method for approximating area under the curve of the power spectral density. freq_range specifies the frequency range, by default uses 0.5 to 4 Hz for delta power, but can be adjusted to anything above 0.1 Hz (Welch's method doesn't do the very low frequency ranges well).

#### get_rolling_band_power_fourier_sum
get_rolling_band_power_fourier_sum(a, start_index, end_index, freq_range=(0.001, 0.05)): Gets the power of the specified frequency range using a fourier transformation to calculate power spectral density, and a simple log of the sum over the given frequency range. This method is more effective than Welch's method for calculating very lower frequency ranges (and is used for the Heart Rate VLF Power feature).

#### get_rolling_mean_std
get_rolling_mean_std(a, start_index, end_index, freq=500, window_sec=30): Gets the mean values and standard deviation of a window specified by window_sec (window size in seconds), for a vector of a given frequency (default freq=500 Hz). By default calculates the 30-second mean and standard deviation at each data point (default every 1 second). 

#### get_heart_rate
get_heart_rate(ecg_data, fs=500, search_radius=200, filter_threshold=200): Gets the heart rate for a given ECG vector, with a certain search radius and filter_threshold. The search radius should be set so that you create an upper bound for heart rate depending on context of what a feasible heart rate is for the animal in question. By default 200 with 500 Hz data means the search radius is 2/5 of a second, creating an upper bound of 150 beats per minute. Filter threshold throws out any value returned by the heart rate calculation above a certain beats per minute (default of 200 means anything above 200 bpm gets its neighbors' average).

#### get_rolling_zero_crossings
get_rolling_zero_crossings(a, start_index, end_index, freq=500, window_sec=1): Gets the number of EEG zero crossings over a certain window (default 1 second) every second (with default step_size=1).

#### get_rolling_band_power_multitaper
get_rolling_band_power_multitaper(a, start_index, end_index, freq_range=(0.5, 4), ref_power=1e-13, freq=500, window_sec=2, step_size=1, in_dB=True): Gets the power of the specified frequency range using MNE's power spectral density multitaper method and Simpson's method for approximating area under the curve of the power spectral density (if in_dB=True, else just returns the PSD mean). While this is theoretically more accurate than Welch, it is much much slower and the increased granularity is not worth the time it takes when applied to large datasets.

### feature_generation.py
This python script is a wrapper of the feature_extraction.py script, and can be run from the terminal using `python feature_generation.py <Input_EDF_Filepath> <Output_CSV_Filepath> [Config_Filepath]`. The Config_Filepath should be a path to a json file that has key value pairs like the ones in the DEFAULT_CONFIG variable at the top of the feature_generation.py script. These parameters can be used to adjust window size, step size, and other parameters used by some of the feature functions (although some of the parameters have not been tested thoroughly so adjust them at your own risk). Any config keys not defined by the config file will use the default values defined at the top of feature_generation.py
