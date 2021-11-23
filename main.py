import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.signal.windows import hamming
import math

frame = 256
sampling_rate = 8000
overlapping_rate = 0.5
order = 12


def extract_features(data):
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
        if np.all(data[offset: len(window) + offset] == 0):  # skip frame with zeros
            continue
        framing_result[count, :] = window * data[offset: len(window) + offset]
        count = count + 1

    block_number = count

    # extract lpc coefficient
    lpc_co = np.zeros((block_number, order + 1))
    for i in range(block_number):
        lpc_co[i, :] = librosa.lpc(framing_result[i, :], order)

    return lpc_co


# load data + data normalization
data_male_1, sr1 = librosa.load('audiodata/male_1.wav', sampling_rate)
data_male_2, sr2 = librosa.load('audiodata/male_2.wav', sampling_rate)
data_female_1, sr3 = librosa.load('audiodata/female_1.wav', sampling_rate)
data_female_2, sr4 = librosa.load('audiodata/female_2.wav', sampling_rate)

male1_co = extract_features(data_male_1)
male2_co = extract_features(data_male_2)
female1_co = extract_features(data_female_1)
female2_co = extract_features(data_female_2)

# change number to change order of coefficient

# plt.plot(male1_co[:, 1], 'b', label="male1")
# plt.plot(male2_co[:, 1], 'c', label="male2")
# plt.plot(female1_co[:, 1], 'r', label="female1")
# plt.plot(female2_co[:, 1], 'm', label="female2")
# plt.legend()
# plt.xlabel("Order of frames")
# plt.ylabel("Coefficient value")
# plt.title("lpc coefficient")
# plt.xlim(0, 60)
# plt.xlim(0, 300)
# plt.show()

# align feature sequence with dtw

male_dtw, male_wp = librosa.sequence.dtw(male1_co[:, 1], male2_co[:, 1])
female_dtw, female_wp = librosa.sequence.dtw(female1_co[:, 1], female2_co[:, 1])
male_female_dtw, male_female_wp = librosa.sequence.dtw(male1_co[:, 1], female2_co[:, 1])

# calculate distance
male_path = np.zeros(len(male_wp[:, 0]))
for i in range(len(male_wp[:, 0])):
    male_path[i] = male_dtw[male_wp[i, 0], male_wp[i, 1]]

male_female_path = np.zeros(len(male_female_dtw[:, 0]))
for i in range(len(male_female_dtw[:, 0])):
    male_female_path[i] = male_female_dtw[male_female_wp[i, 0], male_female_wp[i, 1]]

male_distance = male_path.sum() / len(male_path)
male_female_distance = male_female_path.sum() / len(male_female_path)

# plan: make function, apply to another dimension and calculate multidimensional dtw distance(sum up all)

# male_wp_size = np.asarray(male_wp) * frame / sampling_rate
#
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# librosa.display.specshow(male_dtw, x_axis='time', y_axis='time',
#                          cmap='gray_r', hop_length=frame)
# imax = ax.imshow(male_dtw, cmap=plt.get_cmap('gray_r'),
#                   origin='lower', interpolation='nearest', aspect='auto')
# ax.plot(male_wp_size[:, 1], male_wp_size[:, 0], marker='o', color='r')
# plt.title('Warping Path')
# plt.colorbar()

fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(male_dtw, x_axis='frames', y_axis='frames', ax=ax[0])
ax[0].set(title='DTW cost', xlabel='male1', ylabel='male2')
ax[0].plot(male_wp[:, 1], male_wp[:, 0], label='Optimal path', color='y')
ax[0].legend()
# fig.colorbar(img, ax=ax[0])
ax[1].plot(male_dtw[-1, :] / male_wp.shape[0])
ax[1].set(xlim=[0, male_dtw.shape[1]], ylim=[0, 0.4], title='Matching cost function')

fig = plt.subplots()
img = librosa.display.specshow(male_dtw, x_axis='frames', y_axis='frames')
fig.set(title='DTW cost', xlabel='male1', ylabel='male2',xlim=[0, male_dtw.shape[1]], ylim=[0, 0.4])
fig.plot(male_wp[:, 1], male_wp[:, 0], label='Optimal path', color='y')
fig.legend()
fig.colorbar(img, ax=ax[0])


#
# fig_male = plt.figure(figsize=(16, 8))
#
# # plot male1
# plt.subplot(2, 1, 1)
# librosa.display.waveplot(data_male_1, sr=sampling_rate)
# plt.title('male1')
# ax1 = plt.gca()
#
# # plot male2
# plt.subplot(2, 1, 2)
# librosa.display.waveplot(data_male_2, sr=sampling_rate)
# plt.title('male2')
# ax2 = plt.gca()
#
# plt.tight_layout()
#
# trans_figure_male = fig_male.transFigure.inverted()
# lines = []
# arrows = 30
# points_idx_male = np.int16(np.round(np.linspace(0, male_wp.shape[0] - 1, arrows)))
#
# for tp1, tp2 in male_wp[points_idx_male] * frame / sampling_rate:
#     coord1 = trans_figure_male.transform(ax1.transData.transform([tp1, 0]))
#     coord2 = trans_figure_male.transform(ax2.transData.transform([tp2, 0]))
#
#     line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
#                                    (coord1[0], coord2[0]),
#                                    transform=fig_male.transFigure,
#                                    color='r')
#
#     lines.append(line)
#
# fig_male.lines = lines
# plt.tight_layout()
