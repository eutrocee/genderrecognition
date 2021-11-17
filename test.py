import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.signal import resample
from scipy.signal.windows import hamming
import math

frame = 256
sampling_rate = 44100
target_sample_rate = 44100
file_path = 'audiodata/male_1.wav'
lapping_rate = 0.5
order = 12
count = 0

#load data
data, sr = librosa.load(file_path, sampling_rate)

#resample to 8khz
#target_size = int(len(data)*target_sample_rate/sr)
#data = resample(data, target_size)
#sr = target_sample_rate

#make hamming window
sym = False
window = hamming(frame, sym)

#apply OLA
step = math.floor(frame*lapping_rate)
block_number = math.floor((len(data) - len(window)) / step) + 1
framing_result = np.zeros((block_number, len(window)))

for i in range(block_number):
    offset = i * step
    if np.all(data[offset: len(window) + offset] == 0):  # skip store frame with zeros
        continue
    framing_result[count, :] = window * data[offset: len(window) + offset]
    count = count + 1

block_number = count

#extract lpc coefficient
lpc_co = np.zeros((block_number, order + 1))
for i in range(block_number):
    offset = i * step
    lpc_co[i, :] = librosa.lpc(framing_result[i,:], order)
