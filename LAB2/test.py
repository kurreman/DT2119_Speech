import lab2_proto as proto
import numpy as np

def test():
    data_raw = np.load('lab2_data.npz', allow_pickle=True)['data']
    example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()
    phoneHMMs_fem = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
    phoneHMMs_all = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    
    proto.concatTwoHMMs(phoneHMMs_fem["ow"],phoneHMMs_fem["ah"])
    
test()