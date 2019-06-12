from scipy.signal import butter, filtfilt
from scipy import signal
import matplotlib.pyplot as plt
from tools.timeit import timeit

# @timeit
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# @timeit
def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# @timeit
def filter(data, params, only_inputs=False):
    cutoff, fs, order = params["cutoff"], params["fs"], params["order"]

    for split in data:
        for day in split:
            day["glucose"] = butter_lowpass_filtfilt(day["glucose"], cutoff, fs, order)
            if not only_inputs:
                day["y_ph-1"] = butter_lowpass_filtfilt(day["y_ph-1"], cutoff, fs, order)
                day["y_ph"] = butter_lowpass_filtfilt(day["y_ph"], cutoff, fs, order)

    return data


def plot_spectrogram(data):
    f, Pxx = signal.periodogram(data)
    plt.figure()
    plt.plot(f, Pxx)