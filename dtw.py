import librosa.sequence
import numpy as np


# calculate multidimensional dtw distance(suppose independent)
# parameter ; np.array (output of function 'extract_features' in lpc.py)
# return ; dtw distance of each order, np.array
def calculate_dtw_distance(file1_co, file2_co):
    input_dtw_distance = np.zeros(file1_co.shape[1])
    for i in range(file1_co.shape[1]):
        # align feature sequence with dtw
        input_dtw, input_wp = librosa.sequence.dtw(file1_co[:, i], file2_co[:, i])

        # calculate distance
        input_path = np.zeros(len(input_wp[:, 0]))
        for j in range(len(input_wp[:, 0])):
            input_path[j] = input_dtw[input_wp[j, 0], input_wp[j, 1]]

        # store result of this order
        input_dtw_distance[i] = (input_path.sum() / len(input_path))

    return input_dtw_distance
