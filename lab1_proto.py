# DT2119, Lab 1 Feature Extraction
from multiprocessing.sharedctypes import Value
import numpy as np
import scipy as sp
from scipy import signal, fftpack
import lab1_tools as tools
# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
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
    return logMelSpectrum(spec, samplingrate)[0] #!! Added indexing since my function returns two variables

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
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
    return tools.lifter(ceps, liftercoeff)

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
    samplelength = len(samples)
    N = 1+int((samplelength-1*winlen)/(winlen-winshift))
    enframedsignal = np.zeros((N,winlen))
    start = 0
    stop = winlen
    for i in range(N):
        curr_win = samples[start:stop]
        enframedsignal[i] = curr_win
        start += winlen-winshift
        stop +=  winlen-winshift
    return enframedsignal


    
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
    return signal.lfilter([1, -p], [1], input)

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
    num_rows, num_cols = input.shape
    ham_window = signal.hamming(num_cols,sym=0)
    for i in range(num_rows):
        input[i] = np.multiply(input[i],ham_window) #element wise multiplication
    return input

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
    fft_input = np.zeros((input.shape[0],nfft),dtype=np.complex_)
    for i in range(input.shape[0]):
        fft_input[i] = fftpack.fft(input[i],nfft)
    return abs(fft_input)**2



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
    nfft = input.shape[1]
    trfil = tools.trfbank(samplingrate,nfft)
    input_filtered = np.matmul(input,trfil.T)
    return np.log(input_filtered), trfil


def cepstrum(input, nceps):
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
    input_dct = fftpack.dct(input)
    input_dct = tools.lifter(input_dct)
    input_dct_cut = input_dct[:,0:nceps]
    return input_dct_cut

def locD(x,y):
    """Input: two matrices of MFCC coeffs for a speech sample. n_windows X n_MFCCs
        Output: single matrix with local euclidian distance matric of size n_wind_x X n_wind_y"""

    locD = np.zeros((x.shape[0],y.shape[0]))
    for i, MFFC_vec_i in enumerate(x):
        for j, MFFC_vec_j in enumerate(y):
            locD[i,j] = np.linalg.norm(MFFC_vec_i-MFFC_vec_j)
    return locD



def dtw(x, y, dist=locD):
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
    LD = locD(x,y)
    AD = np.full(LD.shape, None)
    path = [[AD.shape[0],0]]
    for i in reversed(range(LD.shape[0])):
        for j in range(LD.shape[1]):
            neighbours = [] #neighbour index 0 = left, 1 = diag, 2 = under
            neighbours_index = [[i,j-1],[i+1,j-1],[i+1,j]]
            
            if j-1 >= 0: #handling out of bounds
                neighbours.append(AD[i,j-1])
            else:
                neighbours.append(np.Inf)
            try: #handling out of bounds
                if j-1 >= 0: #handling out of bounds
                    neighbours.append(AD[i+1,j-1])
                else:
                    neighbours.append(np.Inf)
            except IndexError:
                neighbours.append(np.Inf)

            try: #handling out of bounds
                neighbours.append(AD[i+1,j])
            except IndexError:
                neighbours.append(np.Inf)

            index_min = np.argmin(neighbours)
            AccD_min = neighbours[index_min]
            if AccD_min == np.Inf:
                AccD_min = 0
            else: 
                path.append(neighbours_index[index_min])
            AD[i,j] = LD[i,j] + AccD_min
    d = AD[-1,-1]/(AD.shape[0]+AD.shape[1])
    return d, LD, AD, path




