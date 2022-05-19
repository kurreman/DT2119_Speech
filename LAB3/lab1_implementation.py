# DT2119, Lab 1 Feature Extraction

import numpy as np
import scipy
from scipy.signal import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances

from lab1_tools import (
    trfbank,
    lifter as tools_lifter,
    tidigit2labels as tools_tidigit2labels,
)


def pairwise_distance(A, B):
    return euclidean_distances(A, B)


def tidigit2labels(tidigitsarray):
    return tools_tidigit2labels(tidigitsarray)


def GMM(data, n_components, gm=None, **kvargs):
    if gm is not None:
        gm.fit(data)
    else:
        gm = GaussianMixture(n_components=n_components, **kvargs).fit(data)
    return gm


def lifter(mfcc, l=22):
    return tools_lifter(mfcc, l)


def mspec(
    samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000
):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(
    samples,
    winlen=400,
    winshift=200,
    preempcoeff=0.97,
    nfft=512,
    nceps=13,
    samplingrate=20000,
    liftercoeff=22,
):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    return np.lib.stride_tricks.sliding_window_view(samples, (winlen,))[::winshift, :]

    # if copy:
    #     return view.copy() and set writeable=True
    # else:
    #     return view


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    return lfilter([1, -p], [1], input)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    return input * hamming(input.shape[1], sym=0)


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    return np.abs(scipy.fft.fft(input, nfft)) ** 2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    return trfbank(samplingrate, input.shape[1])


def cepstrum(input, nceps, **kvargs):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    # n : int, optional
    #   Length of the transform. If n < x.shape[axis], x is truncated. If n > x.shape[axis], x is zero-padded. The default results in n = x.shape[axis].
    return dct(input, **kvargs)[:, :nceps]


def dtw(x, y, dist=pairwise_distance, calculate_path=False):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    N = x.shape[0]
    M = y.shape[0]

    LD = pairwise_distance(x, y)
    AD = np.ones((N, M)) * np.inf
    path = np.empty((N, M, 2))
    path[:] = np.nan

    AD[:, 0] = LD[:, 0].cumsum()
    AD[0, :] = LD[0, :].cumsum()

    for i in range(1, N):
        for j in range(1, M):
            cost = LD[i, j]
            costs = [AD[i - 1, j], AD[i, j - 1], AD[i - 1, j - 1]]
            path[i, j, :] = [(i - 1, j), (i, j - 1), (i - 1, j - 1)][np.argmin(costs)]
            AD[i, j] = cost + costs[np.argmin(costs)]

    # calcualte shortest path
    if calculate_path:
        currents = [np.array([path.shape[0], path.shape[1]]) - 1]
        while currents[-1][0] > 0 and currents[-1][1] > 0:
            currents.append(path[currents[-1][0], currents[-1][1], :].astype(int))
        path = np.array(currents)

    return AD[-1, -1], LD, AD, path
