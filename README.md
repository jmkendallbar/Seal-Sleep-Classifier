# Seal-Sleep-Classifier
Original code: https://github.com/jmkendallbar/Seal-Sleep-Capstone
Code for seal sleep classifier for use in EcoViz+AI Workshop.

This GitHub repository is meant to serve as a demonstration for how Seal ECG, EEG, body movement, and pressure features can be used in a machine learning model to automate seal sleep state prediction.

## Directory structure

# **Insert tree here**
add descriptions to files

## Prerequisites
- Python 3.10 (tested with 3.10.9)
- [Jupyter notebooks](https://jupyter.org/install)
- Seal Sleep data files, downloaded from: [**FigShare Seal Sleep Project**](https://figshare.com/account/projects/199498/articles/25464379)

## Notebooks
- ***00_initial_feature_extraction.ipynb***: Initial notebook used for feature extraction, not the prettiest of notebooks, but walks through heart rate extraction from ECG using peak detection. Also goes through each of the features and plots their distributions to get a first-pass glimpse at the feature space.
- ***01_initial_features_and_models.ipynb***: Uses the features created in *00_initial_feature_extraction* to create a few rudimentary scikit-learn machine learning models, including a Support Vector Machine Classifier (SVC), K Nearest Neighbors Classifier (KNN), and Random Forest Classifier (RFC). Also performs grid searching on each of these estimators to find ideal or close-to-ideal parameterizations of these models.
- ***01a_YASA_feature_extraction.ipynb***: Has a simple demonstration of applying the YASA sleep staging algorithm built for humans onto one seal. While the performance is not the best, many of the features implemented by YASA were used for this project, and in some cases their code was used as well (but adjusted for seals). Note that YASA requires a LOT of memory and the kernel often crashes for me if I have other notebooks running or too many Google Chrome tabs open.
- ***02_advanced_feature_generation_and_models.ipynb***: Uses the (semi) finalized feature extraction code in ***feature_extraction.py*** and ***feature_generation.py*** to generate all the available features for sleep detection created so far. Also creates a RFC using the grid-searched parameters from ***01_initial_features_and_models***, and plots the predictions against the true labels and features for a few naps

## Usage
<ins>feature_extraction.py</ins>
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
get_rolling_band_power_welch(a, start_index, end_index, freq_range=(0.5, 4), freq=500): Gets the power of the specified frequency range using 
<ins>feature_generation.py</ins>
