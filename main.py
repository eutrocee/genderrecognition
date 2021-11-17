import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.signal.windows import hamming
import math

frame = 256
target_sample_rate = 8000
overlapping_rate = 0.5
order = 12

def extract_features(data):
#make hamming window
    sym = False
    window = hamming(frame, sym)

#apply OLA + window + remove row with zeros
    count = 0
    step = math.floor(frame*overlapping_rate)
    block_number = math.floor((len(data) - len(window)) / step) + 1
    framing_result = np.zeros((block_number, len(window)))

    for i in range(block_number):
        offset = i * step
        if np.all(data[offset: len(window) + offset] == 0):  # skip frame with zeros
            continue
        framing_result[count, :] = window * data[offset: len(window) + offset]
        count = count + 1

    block_number = count

#extract lpc coefficient
    lpc_co = np.zeros((block_number, order + 1))
    for i in range(block_number):
        offset = i * step
        lpc_co[i, :] = librosa.lpc(framing_result[i, :], order)

    return lpc_co

#load data + data nomalization
data_male_1, sr1 = librosa.load('audiodata/male_1.wav', sampling_rate)
data_male_2, sr2 = librosa.load('audiodata/male_2.wav', sampling_rate)
data_female_1, sr3 = librosa.load('audiodata/female_1.wav', sampling_rate)
data_female_2, sr4 = librosa.load('audiodata/female_2.wav', sampling_rate)

male1_co = extract_features(data_male_1)
male2_co = extract_features(data_male_2)
female1_co = extract_features(data_female_1)
female2_co = extract_features(data_female_2)

#change number to change order of coefficient
plt.plot(male1_co[:, 2], 'b', label="male1")
plt.plot(male2_co[:, 2], 'c', label="male2")
plt.plot(female1_co[:, 2], 'r', label="female1")
plt.plot(female2_co[:, 2], 'm', label="female2")
plt.legend()
plt.xlabel("Order of frames")
plt.ylabel("Coefficient value")
plt.title("lpc coefficient")
#plt.xlim(0, 60)
plt.xlim(0, 300)
plt.show()