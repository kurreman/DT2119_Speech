#Koray Amico Kulbay, DT2119, Lab 1, tests

import lab1_proto as l
import numpy as np
import matplotlib.pyplot as plt

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

data = np.load('lab1_data.npz', allow_pickle=True)['data']
plt.pcolormesh(l.enframe(example["samples"],20,10))
plt.show()
if example["frames"] != l.enframe(example["samples"],20,10):
    raise Exception("Enframe broken")
#print(example["samples"])
#print(example["frames"])
#print(data[0]["samples"])

