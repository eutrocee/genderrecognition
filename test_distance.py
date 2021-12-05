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

# to test distance, put every audio files to ref
# test 3 input files & 13 ref files each test

# load & setting reference data
path_male = './audiodata/refdata/male/'
file_list_male = os.listdir(path_male)
path_female = './audiodata/refdata/female/'
file_list_female = os.listdir(path_female)

ref_data_male_1, sr_ref = librosa.load(path_male + file_list_male[0], sampling_rate)
ref_data_male_2, sr_ref = librosa.load(path_male + file_list_male[1], sampling_rate)
ref_data_male_3, sr_ref = librosa.load(path_male + file_list_male[2], sampling_rate)
ref_data_male_4, sr_ref = librosa.load(path_male + file_list_male[3], sampling_rate)
ref_data_male_5, sr_ref = librosa.load(path_male + file_list_male[4], sampling_rate)
ref_data_male_6, sr_ref = librosa.load(path_male + file_list_male[5], sampling_rate)
ref_data_male_7, sr_ref = librosa.load(path_male + file_list_male[6], sampling_rate)
ref_data_male_8, sr_ref = librosa.load(path_male + file_list_male[7], sampling_rate)
ref_data_male_9, sr_ref = librosa.load(path_male + file_list_male[8], sampling_rate)
ref_data_male_10, sr_ref = librosa.load(path_male + file_list_male[9], sampling_rate)
ref_data_male_11, sr_ref = librosa.load(path_male + file_list_male[10], sampling_rate)
ref_data_male_12, sr_ref = librosa.load(path_male + file_list_male[11], sampling_rate)
ref_data_male_13, sr_ref = librosa.load(path_male + file_list_male[12], sampling_rate)

ref_data_female_1, sr_ref = librosa.load(path_female + file_list_female[0], sampling_rate)
ref_data_female_2, sr_ref = librosa.load(path_female + file_list_female[1], sampling_rate)
ref_data_female_3, sr_ref = librosa.load(path_female + file_list_female[2], sampling_rate)
ref_data_female_4, sr_ref = librosa.load(path_female + file_list_female[3], sampling_rate)
ref_data_female_5, sr_ref = librosa.load(path_female + file_list_female[4], sampling_rate)
ref_data_female_6, sr_ref = librosa.load(path_female + file_list_female[5], sampling_rate)
ref_data_female_7, sr_ref = librosa.load(path_female + file_list_female[6], sampling_rate)
ref_data_female_8, sr_ref = librosa.load(path_female + file_list_female[7], sampling_rate)
ref_data_female_9, sr_ref = librosa.load(path_female + file_list_female[8], sampling_rate)
ref_data_female_10, sr_ref = librosa.load(path_female + file_list_female[9], sampling_rate)
ref_data_female_11, sr_ref = librosa.load(path_female + file_list_female[10], sampling_rate)
ref_data_female_12, sr_ref = librosa.load(path_female + file_list_female[11], sampling_rate)
ref_data_female_13, sr_ref = librosa.load(path_female + file_list_female[12], sampling_rate)


ref_data_co_male_1 = extract_features(ref_data_male_1, frame, overlapping_rate, order)
ref_data_co_male_2 = extract_features(ref_data_male_2, frame, overlapping_rate, order)
ref_data_co_male_3 = extract_features(ref_data_male_3, frame, overlapping_rate, order)
ref_data_co_male_4 = extract_features(ref_data_male_4, frame, overlapping_rate, order)
ref_data_co_male_5 = extract_features(ref_data_male_5, frame, overlapping_rate, order)
ref_data_co_male_6 = extract_features(ref_data_male_6, frame, overlapping_rate, order)
ref_data_co_male_7 = extract_features(ref_data_male_7, frame, overlapping_rate, order)
ref_data_co_male_8 = extract_features(ref_data_male_8, frame, overlapping_rate, order)
ref_data_co_male_9 = extract_features(ref_data_male_9, frame, overlapping_rate, order)
ref_data_co_male_10 = extract_features(ref_data_male_10, frame, overlapping_rate, order)
ref_data_co_male_11 = extract_features(ref_data_male_11, frame, overlapping_rate, order)
ref_data_co_male_12 = extract_features(ref_data_male_12, frame, overlapping_rate, order)
ref_data_co_male_13 = extract_features(ref_data_male_13, frame, overlapping_rate, order)

ref_data_co_female_1 = extract_features(ref_data_female_1, frame, overlapping_rate, order)
ref_data_co_female_2 = extract_features(ref_data_female_2, frame, overlapping_rate, order)
ref_data_co_female_3 = extract_features(ref_data_female_3, frame, overlapping_rate, order)
ref_data_co_female_4 = extract_features(ref_data_female_4, frame, overlapping_rate, order)
ref_data_co_female_5 = extract_features(ref_data_female_5, frame, overlapping_rate, order)
ref_data_co_female_6 = extract_features(ref_data_female_6, frame, overlapping_rate, order)
ref_data_co_female_7 = extract_features(ref_data_female_7, frame, overlapping_rate, order)
ref_data_co_female_8 = extract_features(ref_data_female_8, frame, overlapping_rate, order)
ref_data_co_female_9 = extract_features(ref_data_female_9, frame, overlapping_rate, order)
ref_data_co_female_10 = extract_features(ref_data_female_10, frame, overlapping_rate, order)
ref_data_co_female_11 = extract_features(ref_data_female_11, frame, overlapping_rate, order)
ref_data_co_female_12 = extract_features(ref_data_female_12, frame, overlapping_rate, order)
ref_data_co_female_13 = extract_features(ref_data_female_13, frame, overlapping_rate, order)

# load data + data normalization
#print('Input file:', sys.argv[1])
input_data_1, sr = librosa.load('./audiodata/' + sys.argv[1], sampling_rate)
input_data_2, sr = librosa.load('./audiodata/' + sys.argv[2], sampling_rate)
input_data_3, sr = librosa.load('./audiodata/' + sys.argv[3], sampling_rate)

# feature extracting
input_data_co_1 = extract_features(input_data_1, frame, overlapping_rate, order)
input_data_co_2 = extract_features(input_data_2, frame, overlapping_rate, order)
input_data_co_3 = extract_features(input_data_3, frame, overlapping_rate, order)


# plotting
# plot_lpc_coefficient_compare4(input_data_co, ref_data_co_male_1, ref_data_co_male_2, ref_data_co_female_1, 1)
# plot_dtw_path(input_data_co, ref_data_co_male_1, 1)

# calculate dtw distance

lpc_list_male_1 = [sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_1)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_2)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_3)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_4)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_5)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_6)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_7)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_8)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_9)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_10)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_11)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_12)),
                   sum(calculate_dtw_distance(input_data_co_1, ref_data_co_male_13))]

lpc_list_female_1 = [sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_1)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_2)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_3)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_4)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_5)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_6)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_7)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_8)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_9)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_10)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_11)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_12)),
                     sum(calculate_dtw_distance(input_data_co_1, ref_data_co_female_13))]

lpc_list_male_2 = [sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_1)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_2)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_3)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_4)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_5)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_6)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_7)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_8)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_9)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_10)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_11)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_12)),
                   sum(calculate_dtw_distance(input_data_co_2, ref_data_co_male_13))]

lpc_list_female_2 = [sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_1)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_2)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_3)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_4)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_5)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_6)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_7)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_8)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_9)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_10)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_11)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_12)),
                     sum(calculate_dtw_distance(input_data_co_2, ref_data_co_female_13))]

lpc_list_male_3 = [sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_1)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_2)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_3)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_4)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_5)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_6)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_7)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_8)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_9)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_10)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_11)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_12)),
                   sum(calculate_dtw_distance(input_data_co_3, ref_data_co_male_13))]

lpc_list_female_3 = [sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_1)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_2)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_3)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_4)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_5)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_6)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_7)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_8)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_9)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_10)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_11)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_12)),
                     sum(calculate_dtw_distance(input_data_co_3, ref_data_co_female_13))]



# identify speaker

print('Closest speaker with ', sys.argv[1])
print('with male:', file_list_male[lpc_list_male_1.index(sorted(lpc_list_male_1)[1])])
print('Distance:', lpc_list_male_1[lpc_list_male_1.index(sorted(lpc_list_male_1)[1])])
print('with female:', file_list_female[lpc_list_female_1.index(sorted(lpc_list_female_1)[1])])
print('Distance:', lpc_list_female_1[lpc_list_female_1.index(sorted(lpc_list_female_1)[1])])

print('Closest speaker with ', sys.argv[2])
print('with male:', file_list_male[lpc_list_male_2.index(sorted(lpc_list_male_2)[1])])
print('Distance:', lpc_list_male_2[lpc_list_male_2.index(sorted(lpc_list_male_2)[1])])
print('with female:', file_list_female[lpc_list_female_2.index(sorted(lpc_list_female_2)[1])])
print('Distance:', lpc_list_female_2[lpc_list_female_2.index(sorted(lpc_list_female_2)[1])])

print('Closest speaker with ', sys.argv[3])
print('with male:', file_list_male[lpc_list_male_3.index(sorted(lpc_list_male_3)[1])])
print('Distance:', lpc_list_male_3[lpc_list_male_3.index(sorted(lpc_list_male_3)[1])])
print('with female:', file_list_female[lpc_list_female_3.index(sorted(lpc_list_female_3)[1])])
print('Distance:', lpc_list_female_3[lpc_list_female_3.index(sorted(lpc_list_female_3)[1])])