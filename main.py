import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.signal import resample
from scipy.signal.windows import hann
import math

frame = 256
sampling_rate = 44100
target_sample_rate = 8000
file_path = 'voice1.wav'

#load data
data, sr = librosa.load(file_path, sampling_rate)

#resample to 8khz
target_size = int(len(data)*target_sample_rate/sr)
data = resample(data, target_size)
sr = target_sample_rate

#apply OLA
def create_overlapping_blocks(x, w, R = 0.5):
    n = len(x)
    nw = len(w)
    step = math.floor(nw * (1 - R))
    nb = math.floor((n - nw) / step) + 1

    B = np.zeros((nb, nw))

    for i in range(nb):
        offset = i * step
        B[i, :] = w * x[offset : nw + offset]

        return B

def add_overlapping_blocks(B, w, R = 0.5):
    [count, nw] = X.shape
    step = math.floor(nw * R)

    n = (count-1) * step + nw
    x = np.zeros((n, ))

    for i in range(count):
        offset = i * step
        x[offset : nw + offset] += B[i, :]
    return x

def make_matrix_X(x, p):
    n = len(x)
    # [x_n, ..., x_1, 0, ..., 0]
    xz = np.concatenate([x[::-1], np.zeros(p)])

    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i
        X[i, :] = xz[offset : offset + p]
    return X

def solve_lpc(x, p, ii):
    b = x[1:]

    X = make_matrix_X(x, p)

    a = np.linalg.lstsq(X, b.T, rcond=-1)[0]

    e = b - np.dot(X, a)
    g = np.var(e)

    return [a, g]

def lpc_encode(x, p, w):
    B = create_overlapping_blocks(x, w)
    [nb, nw] = B.shape

    A = np.zeros((p, nb))
    G = np.zeros((1, nb))

    for i in range(nb):
        [a, g] = solve_lpc(B[i, :], p, i)

        A[:, i] = a
        G[:, i] = g

    return [A, G]


sym = False
window = hann((len(data)), sym)

p = 6
[A, G] = lpc_encode(data, p, window)

print(A)

A_new = librosa.lpc(data[0:frame], 5)

print(A_new)



#plt.figure()
#librosa.display.waveplot(data, sampling_rate, alpha=0.5)
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.title("Waveform")
#plt.show()