#Koray Amico Kulbay, DT2119, Lab 1, tests

import lab1_proto as proto
import numpy as np
import matplotlib.pyplot as plt

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
data = np.load('lab1_data.npz', allow_pickle=True)['data']
fsampling = 20E3 #Hz
winlen = int(20E-3*fsampling)
winshift = int(10E-3*fsampling)

#testing enframe
if not np.array_equal(example["frames"],proto.enframe(example["samples"],winlen,winshift)):
    raise Exception("Enframe broken")


