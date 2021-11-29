import librosa
import os
import sys
from lpc import extract_features
from dtw import calculate_dtw_distance

frame = 256
sampling_rate = 8000
overlapping_rate = 0.5
order = 12
threshold = 120

# load & setting reference data
path_male = './audiodata/refdata/male/'
file_list_male = os.listdir(path_male)
path_female = './audiodata/refdata/female/'
file_list_female = os.listdir(path_female)

ref_data_male_1, sr_ref = librosa.load(path_male + file_list_male[0], sampling_rate)
ref_data_male_2, sr_ref = librosa.load(path_male + file_list_male[1], sampling_rate)
ref_data_male_3, sr_ref = librosa.load(path_male + file_list_male[2], sampling_rate)
ref_data_female_1, sr_ref = librosa.load(path_female + file_list_female[0], sampling_rate)
ref_data_female_2, sr_ref = librosa.load(path_female + file_list_female[1], sampling_rate)
ref_data_female_3, sr_ref = librosa.load(path_female + file_list_female[2], sampling_rate)

ref_data_co_male_1 = extract_features(ref_data_male_1, frame, overlapping_rate, order)
ref_data_co_male_2 = extract_features(ref_data_male_2, frame, overlapping_rate, order)
ref_data_co_male_3 = extract_features(ref_data_male_3, frame, overlapping_rate, order)
ref_data_co_female_1 = extract_features(ref_data_female_1, frame, overlapping_rate, order)
ref_data_co_female_2 = extract_features(ref_data_female_2, frame, overlapping_rate, order)
ref_data_co_female_3 = extract_features(ref_data_female_3, frame, overlapping_rate, order)

# load data + data normalization
print('Input file:', sys.argv[1])
input_data, sr = librosa.load('./audiodata/' + sys.argv[1], sampling_rate)

# feature extracting
input_data_co = extract_features(input_data, frame, overlapping_rate, order)

# plotting
# plot_lpc_coefficient_compare4(input_data_co, ref_data_co_male_1, ref_data_co_male_2, ref_data_co_female_1, 1)
# plot_dtw_path(input_data_co, ref_data_co_male_1, 1)

# calculate dtw distance
min_distance_male = min(sum(calculate_dtw_distance(input_data_co, ref_data_co_male_1)),
                        sum(calculate_dtw_distance(input_data_co, ref_data_co_male_2)),
                        sum(calculate_dtw_distance(input_data_co, ref_data_co_male_3)))

min_distance_female = min(sum(calculate_dtw_distance(input_data_co, ref_data_co_female_1)),
                          sum(calculate_dtw_distance(input_data_co, ref_data_co_female_2)),
                          sum(calculate_dtw_distance(input_data_co, ref_data_co_female_3)))

# identify speaker
if (min_distance_male < min_distance_female) and (min_distance_male < threshold):
    print('Speaker is male.')
    print('Distance:', min_distance_male)
elif min_distance_female < threshold:
    print('Speaker is female.')
    print('Distance:', min_distance_female)
else:
    print('Cannot identify speaker!')
    print('Distance:', min(min_distance_male, min_distance_female))
