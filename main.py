import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.signal import resample
from scipy.signal.windows import hamming
import math

frame = 256
sampling_rate = 44100
target_sample_rate = 8000
file_path = 'voice1.wav'
lapping_rate = 0.5
order = 12

#load data
data, sr = librosa.load(file_path, sampling_rate)

#resample to 8khz
target_size = int(len(data)*target_sample_rate/sr)
data = resample(data, target_size)
sr = target_sample_rate

#make hamming window
sym = False
window = hamming(frame, sym)

#apply OLA
step = math.floor(frame*lapping_rate)
block_number = math.floor((len(data) - len(window)) / step) + 1
framming_result = np.zeros((block_number, len(window)))

for i in range(block_number):
    offset = i * step
    framming_result[i, :] = window * data[offset : len(window) + offset]

#extract lpc coefficient
lpc_co = np.zeros((block_number, order + 1))
for i in range(block_number):
    offset = i * step
    lpc_co[i, :] = librosa.lpc(framming_result[i,:], order)



#plt.figure()
#librosa.display.waveplot(data, sampling_rate, alpha=0.5)
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.title("Waveform")
#plt.show()