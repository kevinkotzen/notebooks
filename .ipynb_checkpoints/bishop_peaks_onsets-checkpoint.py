"""Bishop PPG Peak and Onset Detector
# Python implementation: Lea Amar
# -----------------------------
# From Matlab function bishop_peak_detector by Peter Charlton
# from: https://link.springer.com/chapter/10.1007/978-3-319-65798-1_39
# ----------
# Physiology Feature Extraction Toolkit
# Dr Steven Bishop, 2015-16
# Division of Anaesthesia, University of Cambridge, UK
# Email: sbishop@doctors.org.uk
# ----------
# PEAK_TROUGH_FINDER
#
# Based upon the algorithm by (with updates and optimisations):
# Scholkmann F, Boss J, Wolk M. An Efficient Algorithm for Automatic Peak
# Detection in Noisy Periodic and Quasi-Periodic Signals. Algorithms 2012
# (5), p588-603; doi:10.3390/a5040588
# ----------
"""

import os
import math
import copy
import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt, resample
from numba import jit
import tqdm

def bishop_peaks_and_onsets(ppg_signal, fs, min_time_between_ms=20,
                            pre_filter=True, filter_lowcut=0.5, filter_highcut=8, filter_order=8,
                            resample=True, resample_fs=64,
                            window=100, hop=90, correct_points=True):
    
    raw_signal = copy.deepcopy(ppg_signal)
    
    if pre_filter:
        sos = butter(filter_order, [filter_lowcut / (fs / 2), filter_highcut / (fs / 2)], 'bandpass', output='sos')
        ppg_signal = sosfiltfilt(sos, ppg_signal, axis=0)  # , padlen=3 * (max(len(b), len(a)) - 1))

    if resample:
        assert filter_highcut * 4 < resample_fs
        assert fs % resample_fs == 0
        F = fs / resample_fs
        ppg_signal = scipy.signal.resample(ppg_signal, int(len(ppg_signal) / F))
        fs = resample_fs
    else:
        F = 1

    hop = hop*fs
    window = window*fs

    all_peaks = np.array([])
    all_onsets = np.array([])
    win_starts = np.arange(0, len(ppg_signal), hop)
    
    
    for i in range(len(win_starts)):
        try:
            ind = np.arange(win_starts[i], win_starts[i] + window)
            short_sig = ppg_signal[ind]
        except IndexError:
            ind = np.arange(win_starts[i], len(ppg_signal))
            short_sig = ppg_signal[ind]
        short_peaks, short_onsets = _bishop_peak_detector(short_sig)
        short_peaks = short_peaks + win_starts[i]
        short_onsets = short_onsets + win_starts[i]
        all_peaks = np.append(all_peaks, short_peaks)
        all_onsets = np.append(all_onsets, short_onsets)

    # in the overlap if two peaks are referenced twice, remove them
    all_peaks = np.unique(all_peaks)
    all_onsets = np.unique(all_onsets)

    # delete peaks that are within 20ms range of one another
    thresh = min_time_between_ms / fs
    all_peaks = np.delete(all_peaks, np.argwhere(np.ediff1d(all_peaks) <= thresh) + 1)
    all_onsets = np.delete(all_onsets, np.argwhere(np.ediff1d(all_onsets) <= thresh) + 1)

    all_peaks =  (all_peaks*F).astype(int)
    all_onsets = (all_onsets*F).astype(int)
    
    if correct_points:
        all_peaks = _correct_upsample(raw_signal, all_peaks, int(F), 'max')
        all_onsets = _correct_upsample(raw_signal, all_onsets, int(F), 'min')
    
    return all_peaks, all_onsets

def _correct_upsample(ppg_signal, points, F, minmax):
    
    for i in range(points.shape[0]):
        start = points[i] - F
        end = points[i] + F + 1
        short_segment = ppg_signal[start:end]
        if len(short_segment) < F:
            continue
        if minmax=='min':
            correction = np.argmin(short_segment)-F
        else:
            correction = np.argmax(short_segment)-F
        
        points[i] = points[i] + correction
    return points


def _bishop_peak_detector(data):
    '''
    #Data : input data as vector
    #Sampling frequency (optional):sampling frequency of input
    #Returns: vectors [peaks, troughs, maximagram, minimigram] containing indices
    #of the peaks and troughs and the maxima/minima scalograms
    '''

    N = len(data)
    L = math.ceil(N / 2) - 1

    # Detrend the data
    data[np.isnan(data)] = np.nanmean(data)
    data = scipy.signal.detrend(data)

    Mx, Mn = _jit_compute(data, N, L)

    # maximagram = Mx
    # minimigram = Mn

    '''
    #Form Y the column-wise count of where Mx is 0, a scale-dependent distribution of local maxima.
    #Find d, the scale with the most maxima (== most number of zeros in row)
    #Redimension Mx to contain only the first d scales
    '''

    Y = np.count_nonzero(Mx, axis=0)
    d = np.argmax(Y)
    Mx = Mx[:, 0:d]

    '''
    #Form Y2 the column-wise count of where Mn is 0, a scale-dependent distribution of local maxima.
    #Find d2, the scale with the most minima (== most number of zeros in row)
    #Redimension Mn to contain only the first d scales
    '''

    Y2 = np.count_nonzero(Mn, axis=0)
    d2 = np.argmax(Y2)
    Mn = Mn[:, 0:d2]

    # Form Zx and Zn the row-wise counts of Mx and Mn's non-zero elements.
    # Any row with a zero count contains entirely zeros, thus indicating the presence of a peak or trough

    Zx = d - np.count_nonzero(Mx, axis=1)
    # Zx = len(Mx) - np.count_nonzero(Mx, axis=1)
    Zn = d2 - np.count_nonzero(Mn, axis=1)

    # Find all the zeros in Zx and Zn. The indices of the zero counts correspond to the position
    # of peaks and troughs respectively

    peaks = np.array(np.where(Zx == 0)) + 1
    troughs = np.array(np.where(Zn == 0)) + 1

    return peaks, troughs


@jit(nopython=True, cache=True)
def _jit_compute(data, N, L):
    Mx = np.zeros((N, L))
    Mn = np.zeros((N, L))
    # Produce the local maxima scalogram
    for j in range(0, L + 1):
        k = j
        for i in range(k + 2 + 1, N - k + 2):
            if data[i - 2] > data[i - k - 2] and data[i - 2] > data[i + k - 2]:
                Mx[i - 2, j - 1] = True
            if data[i - 2] < data[i - k - 2] and data[i - 2] < data[i + k - 2]:
                Mn[i - 2, j - 1] = True
    return Mx, Mn

