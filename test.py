#Koray Amico Kulbay, DT2119, Lab 1, tests

def tests():
    import lab1_proto as proto
    import numpy as np
    import matplotlib.pyplot as plt

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
    print(testsignal.shape)
    testsignal = proto.logMelSpectrum(testsignal,fsampling)
    print(example["mspec"].shape)

    if not np.allclose(example["mspec"],testsignal):
        raise Exception("powerSpectrum broken")

    print("All tests OK")

tests()
