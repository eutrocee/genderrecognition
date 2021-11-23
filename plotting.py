import librosa.display
from matplotlib import pyplot as plt


# change number to change order of coefficient
# parameter ; np.array (output of function 'extract_features' in lpc.py) * 4, order of lpc coefficient
def plot_lpc_coefficient_compare4(file1_1_co, file1_2_co, file2_1_co, index_order):
    plt.plot(file1_1_co[:, index_order], 'r', label="file1")
    plt.plot(file1_2_co[:, index_order], 'm', label="file2")
    plt.plot(file2_1_co[:, index_order], 'b', label="file3")
    # plt.plot(file2_2_co[:, index_order], 'c', label="file4")
    plt.legend()
    plt.xlabel("Order of frames")
    plt.ylabel("Coefficient value")
    plt.title("lpc coefficient")
    plt.xlim(0, max([len(file1_1_co[:, index_order]), len(file1_2_co[:, index_order]), len(file2_1_co[:, index_order])]))
                     # len(file2_1_co[:, index_order]), len(file2_2_co[:, index_order])]))
    plt.show()

    return 0


# dtw path calculate & plotting
# parameter; np.array (output of function 'extract_features' in lpc.py) * 2, order of lpc coefficient
def plot_dtw_path(input1_co, input2_co, index_order):
    input_dtw, input_wp = librosa.sequence.dtw(input1_co[:, index_order], input2_co[:, index_order])

    fig, ax = plt.subplots()
    img = librosa.display.specshow(input_dtw.T, x_axis='frames', y_axis='frames')
    ax.set(title='DTW cost', xlabel='first array', ylabel='second array')
    ax.plot(input_wp[:, 0], input_wp[:, 1], label='Optimal path', color='y')
    ax.legend()
    fig.colorbar(img)

    plt.show()

    return 0
