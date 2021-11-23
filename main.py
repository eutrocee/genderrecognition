import numpy as np
import librosa
import os
from lpc import extract_features
from plotting import plot_lpc_coefficient_compare4, plot_dtw_path
from dtw import calculate_dtw_distance

frame = 256
sampling_rate = 8000
overlapping_rate = 0.5
order = 12
path_male = './audiodata/refdata/male/'
path_female = './audiodata/refdata/female/'
ref_frame = 70

# make reference data
file_list_male = os.listdir(path_male)
index_file = 0
data_co = np.array(0)

# ref_data_1 = librosa.load(path_male + file_list_male[0], sampling_rate)
# ref_data_co_1 = extract_features(ref_data_1, frame, overlapping_rate, order)
# ref_data_2 = librosa.load(path_male + file_list_male[1], sampling_rate)
# ref_data_co_2 = extract_features(ref_data_2, frame, overlapping_rate, order)
# ref_data_3 = librosa.load(path_male + file_list_male[2], sampling_rate)
# ref_data_co_3 = extract_features(ref_data_3, frame, overlapping_rate, order)
#
# max_len = max([ref_data_co_1.shape[0], ref_data_co_2.shape[0], ref_data_co_3.shape[0]])


# load data + data normalization
data_male_1, sr1 = librosa.load('audiodata/male_3.wav', sampling_rate)
data_male_2, sr2 = librosa.load('audiodata/male_4.wav', sampling_rate)
data_female_1, sr3 = librosa.load('audiodata/female_3.wav', sampling_rate)
data_female_2, sr4 = librosa.load('audiodata/female_4.wav', sampling_rate)

# feature extracting
male1_co = extract_features(data_male_1, frame, overlapping_rate, order)
male2_co = extract_features(data_male_2, frame, overlapping_rate, order)
female1_co = extract_features(data_female_1, frame, overlapping_rate, order)
female2_co = extract_features(data_female_2, frame, overlapping_rate, order)

# plotting
plot_lpc_coefficient_compare4(female1_co, female2_co, male2_co, 1)
# plot_dtw_path(male1_co, male2_co, 1)

# calculate dtw distance
output = calculate_dtw_distance(male1_co, female1_co)
output2 = calculate_dtw_distance(male2_co, female2_co)
output3 = calculate_dtw_distance(female1_co, female2_co)
output4 = calculate_dtw_distance(male1_co, male2_co)
print('male1 - female1', output.sum())
print('male2 - female2',output2.sum())
print('female1 - female2',output3.sum())
print('male1 - male2',output4.sum())