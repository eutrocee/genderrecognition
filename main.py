import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display

FIG_SIZE=(15, 10)


file = 'voice1.wav'

data, sampling_rate = librosa.load(file, sr=44100)

print(data, data.shape)

#plt.figure(figsize=FIG_SIZE)
#librosa.display.waveplot(data, sampling_rate, alpha=0.5)
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.title("Waveform")
