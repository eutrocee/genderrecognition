import math
import numpy as np
from scipy.signal.windows import hamming
import librosa


# extract lpc coefficient from data with OLA and hamming window
# parameter; data : np.array shape=(n,) - audio sequence(to be normalized) / frame : number of element of a frame
#        overlapping_rate : OLA overlapping ratio / order : order of lpc coefficient
# return; coefficient np.array(N*order)
def extract_features(data, frame, overlapping_rate, order):
    # make hamming window
    sym = False
    window = hamming(frame, sym)

    # apply OLA + window + remove row with zeros
    count = 0
    step = math.floor(frame * overlapping_rate)
    block_number = math.floor((len(data) - len(window)) / step) + 1
    framing_result = np.zeros((block_number, len(window)))

    for i in range(block_number):
        offset = i * step
        if np.all(data[offset: len(window) + offset] == 0):  # skip frame fill with zeros
            continue
        framing_result[count, :] = window * data[offset: len(window) + offset]
        count = count + 1

    block_number = count

    # extract lpc coefficient
    lpc_co = np.zeros((block_number, order + 1))
    for i in range(block_number):
        lpc_co[i, :] = librosa.lpc(framing_result[i, :], order)

    return lpc_co
