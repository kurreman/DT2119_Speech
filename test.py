#Koray Amico Kulbay, DT2119, Lab 1, tests

from unittest import TextTestResult
import lab1_proto as proto
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd

def tests():

    example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
    data = np.load('lab1_data.npz', allow_pickle=True)['data']

    fsampling = 20E3 #Hz
    winlen = int(20E-3*fsampling)
    winshift = int(10E-3*fsampling)

    testsignal = example["samples"]

    #testing enframe
    testsignal = proto.enframe(testsignal,winlen,winshift)
    if not np.array_equal(example["frames"],testsignal):
        raise Exception("enframe broken")

    #testing preemp 
    testsignal = proto.preemp(testsignal)
    if not np.array_equal(example["preemph"],testsignal):
        raise Exception("preemp broken")

    #testing windowing
    testsignal = proto.windowing(testsignal)
    if not np.allclose(example["windowed"],testsignal): #True if relative difference for each element is less than 0.001%
        raise Exception("windowing broken")

    #testing powerSpectrum 
    testsignal = proto.powerSpectrum(testsignal,512)
    if not np.allclose(example["spec"],testsignal):
        raise Exception("powerSpectrum broken")

    #testing logMelSpectrum 
    testsignal, trfil = proto.logMelSpectrum(testsignal,fsampling)
    if not np.allclose(example["mspec"],testsignal):
        raise Exception("powerSpectrum broken")

    #testing cepstrum
    testsignal = proto.cepstrum(testsignal,13)
    if not np.allclose(example["lmfcc"],testsignal):
        raise Exception("powerSpectrum broken")

    #Custom tests
    x = 2*np.ones((80,4))#rnd.rand(3,4)
    y = np.ones((5,4))#rnd.rand(5,4)
    locD = proto.locD(x,y)
    assert np.array_equal(locD,2*np.ones((80,5)))

    print("All tests OK")
    LD = np.array([[1,1,1,1,1,1,0],[1,1,1,1,1,0,1],[1,1,1,1,0,1,1],[1,1,1,0,1,1,1],[1,0,0,1,1,1,1],[0,1,1,1,1,1,1]]) #testing
    LD = np.zeros((9,9))
    d, LD, AD, path = proto.dtw(x,y,LD=LD)
    print("d = {} LD = {} AD = {} path = {}".format(d, LD, AD, path))

tests()
