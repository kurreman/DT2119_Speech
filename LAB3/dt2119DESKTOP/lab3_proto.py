from importlib.machinery import WindowsRegistryFinder
import numpy as np
from lab3_tools import *
import os
import lab3_tools as tools
import lab3_proto as proto
import lab1_proto as proto1
import lab1_tools as tools1
import lab2_proto as proto2
import lab2_tools as tools2
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.utils import np_utils
from keras.utils import np_utils
import re

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
   """ word2phones: converts word level to phone level transcription adding silence

   Args:
      wordList: list of word symbols
      pronDict: pronunciation dictionary. The keys correspond to words in wordList
      addSilence: if True, add initial and final silence
      addShortPause: if True, add short pause model "sp" at end of each word
   Output:
      list of phone symbols
   """
   wordList = ["sil"] + wordList + ["sil"]
   phoneList = ["sil"]
   for i, word in enumerate(wordList[1:-1]):
      word = pronDict[word] + ["sp"]
      phoneList = phoneList + word
   
   return phoneList + ["sil"]
    

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
   """ forcedAlignmen: aligns a phonetic transcription at the state level

   Args:
      lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
            computed the same way as for the training of phoneHMMs
      phoneHMMs: set of phonetic Gaussian HMM models
      phoneTrans: list of phonetic symbols to be aligned including initial and
                  final silence

   Returns:
      list of strings in the form phoneme_index specifying, for each time step
      the state from phoneHMMs corresponding to the viterbi path.
   """

def extractDATASET(path):
   """extracts features and targets of each .wav file in dataset. Also force aligns"""
   # if dataset == "training":
   #    path = "tidigits/train"
   # elif dataset == "testing":
   #    path = "tidigits/test"

   #SETUP
   phoneHMMs = np.load('lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
   phones = sorted(phoneHMMs.keys())
   nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
   stateList = np.load("stateList.npy", allow_pickle=True).tolist()

   prondict = {} 
   prondict['o'] = ['ow']
   prondict['z'] = ['z', 'iy', 'r', 'ow']
   prondict['1'] = ['w', 'ah', 'n']
   prondict['2'] = ['t', 'uw']
   prondict['3'] = ['th', 'r', 'iy']
   prondict['4'] = ['f', 'ao', 'r']
   prondict['5'] = ['f', 'ay', 'v']
   prondict['6'] = ['s', 'ih', 'k', 's']
   prondict['7'] = ['s', 'eh', 'v', 'ah', 'n']
   prondict['8'] = ['ey', 't']
   prondict['9'] = ['n', 'ay', 'n']

   data = []
   counter = 0
   for root, dirs, files in os.walk(path):
      for file in files:
         if file.endswith('.wav'):
            counter += 1
            print("currently on .wav file number:",counter)
            print("file",file)

            filename = os.path.join(root, file)
            samples, samplingrate = loadAudio(filename)

            lmfcc,mspec = proto1.mfcc(samples=samples,samplingrate=samplingrate,mspecOutput=True)

            wordTrans = list(tools.path2info(filename)[2])
            phoneTrans = proto.words2phones(wordTrans, prondict)
            utteranceHMM = proto2.concatHMMs(phoneHMMs, phoneTrans)
            stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]
            
            #Alignment
            obsloglik = tools2.log_multivariate_normal_density_diag(lmfcc,utteranceHMM["means"],utteranceHMM["covars"])
            vloglik, vpath = proto2.viterbi(obsloglik,np.log(utteranceHMM["startprob"]),np.log(utteranceHMM["transmat"][:-1,:-1]))

            viterbiStateTrans = []
            for i,stateID in enumerate(vpath):
               viterbiStateTrans.append(stateList.index(stateTrans[stateID]))

            targets = viterbiStateTrans #NOT SURE IF THIS IS CORRECT 
            #...your code for feature extraction and forced alignment
            data.append({'filename': filename, 'lmfcc': lmfcc,
                              'mspec': mspec, 'targets': targets})
   return data


def _stateIDs2stateNames(array,stateList):
   """Converts elements from state index to actual state name according to stateList"""

   for i,index in enumerate(array):
      array[i] = stateList[index]
   return array

def _getGenderIdFile(filename):
   """returns gender, ID, file of filename (which is a path)"""
   remainder, file = os.path.split(filename)
   remainder, ID = os.path.split(remainder)
   remainder, gender = os.path.split(remainder)

   return gender, ID, file

def _getNspeakerNgender(dataset,getspeakerList=False):
   """Returns numbert of unique speakers in dataset and number of men vs women"""

   Nmen = 0
   Nwomen = 0
   speakerList = []
   for datapt in dataset:
      filename = datapt["filename"]
      gender, ID, file = _getGenderIdFile(filename)
      if gender == "man":
         Nmen +=1 
      elif gender == "woman":
         Nwomen += 1
      if not ID in speakerList:
         speakerList.append(ID)
   if getspeakerList:
      return len(speakerList), Nmen, Nwomen, speakerList
   else:
      return len(speakerList), Nmen, Nwomen

def getSubset(dataset,desiredGender):
   """Returns a subset of dataset where each datapt is of a certain gender, man or woman"""
   subset = []
   for datapt in dataset: 
      filename = datapt["filename"]
      gender, ID, file = _getGenderIdFile(filename)
      if gender == desiredGender:
         subset.append(datapt)
   return subset

# def splitDatasetUnique(dataset,P):
#    """Roughly splits dataset into desired proportions P and 1-P, where no speaker ID is common between the subsets"""
#    N = len(dataset)
#    N1 = int(P*N)
#    N2 = N-N1

#    subset1 = []
#    subset2 = []

#    return subset1, subset2

def splitSpeakerSubsets(dataset):
   """Splits a dataset into multiple subsets belonging to only a single speaker"""
   subsets = {}
   for datapt in dataset:
      filename = datapt["filename"]
      gender, ID, file = _getGenderIdFile(filename)
      try: 
         subsets[ID].append(datapt)
      except KeyError:
         subsets[ID] = [datapt]
   return subsets


def stackMatrix(M):
   """Stacks a matrix with 3 values back and 3 values infront of time for each time step. timeXfeatures.
   End cases are handled with just zero feature vectors"""
   Mtemp = np.zeros((M.shape[0]+6,M.shape[1]))
   Mtemp[3:-3,:] = M
   Mstack = np.zeros((M.shape[0],M.shape[1]*7))
   #print(Mtemp[0:3,:])
   #print(Mtemp[-3:,:])

   for n in range(Mstack.shape[0]):
      ntemp = n + 3
      Mstack[n,:] = np.concatenate((Mtemp[ntemp-3,:],Mtemp[ntemp-2,:],Mtemp[ntemp-1,:],
                                    Mtemp[ntemp,:],
                                    Mtemp[ntemp+1,:],Mtemp[ntemp+2,:],Mtemp[ntemp+3,:]),
                                    axis=None)
   return Mstack

def stackDataset(dataset):
   """stacks matrices lmfcc and mspec in datasets, generating new keys called lmfccStacked, mspecStacked"""

   for datapt in dataset: 
      datapt["lmfccStacked"] = stackMatrix(datapt["lmfcc"])
      datapt["mspecStacked"] = stackMatrix(datapt["mspec"])

def standardiseEachUtterance(dataset):
   """Standardises whole dataset such that mean is 0 and variance is 1 for each utterance"""
   counter = 0
   for datapt in dataset:
      datapt["lmfcc"] = _standardiseData(datapt["lmfcc"])
      datapt["lmfccStacked"] = _standardiseData(datapt["lmfccStacked"])
      datapt["mspec"] = _standardiseData(datapt["mspec"])
      datapt["mspecStacked"] = _standardiseData(datapt["mspecStacked"])
      
      counter+=1
      print("datapt:",counter)

def _standardiseData(data):
   scaler = StandardScaler().fit(data)
   dataScaled = scaler.transform(data)
   return dataScaled

def flattenData(dataset):
   """Returns 4 large matrices of dataset lmfcc and mspec also each in stacked version"""

   lmfcc_x = dataset[0]["lmfcc"]
   mspec_x = dataset[0]["mspec"]
   dlmfcc_x = dataset[0]["lmfccStacked"]
   dmspec_x = dataset[0]["mspecStacked"]
   _y = dataset[0]["targets"]
   counter = 0
   for i,datapt in enumerate(dataset):
      counter +=1
      print("currently on datapt:", counter, "of",len(dataset))
      if i != 0:
         lmfcc_x = np.concatenate((lmfcc_x,datapt["lmfcc"]))
         mspec_x = np.concatenate((mspec_x,datapt["mspec"]))
         dlmfcc_x = np.concatenate((dlmfcc_x,datapt["lmfccStacked"]))
         dmspec_x = np.concatenate((dmspec_x,datapt["mspecStacked"]))
         _y = np.concatenate((_y,datapt["targets"]))
   return lmfcc_x,mspec_x,dlmfcc_x,dmspec_x,_y

def trainDNN(features,labels,EPOCHS,BATCH_SIZE):
   """Creates and trains on features and using labels DNN using Keras"""
   stateList = np.load("stateList.npy", allow_pickle=True).tolist()
   feature_dim = features.shape[1]
   output_dim = len(stateList)
   #Create model
   model = tf.keras.Sequential()
   model.add(tf.keras.layers.Dense(80,activation=tf.nn.relu,input_shape=(feature_dim,)))
   model.add(tf.keras.layers.Dense(70,activation=tf.nn.relu)) #choose ReLu since it's faster to compute compared to Sigmoid
   model.add(tf.keras.layers.Dense(output_dim,activation=tf.nn.softmax)) #softmax distributes probabilitites across our states

   model.compile(loss="categorical_crossentropy",
               optimizer="sgd",
               metrics=["accuracy"])

   #EPOCHS = 2 #Goes through data set EPOCHS number of times 
   #BATCH_SIZE = 256 #Each training step model will see BATCH_SIZE number of examples to guide and adjust parameters
   #train model 
   model.fit(features,labels,epochs=EPOCHS,batch_size=BATCH_SIZE)
   return model

def convert2indexArray(M):
   array = np.zeros(M.shape[0])
   for i in range(M.shape[0]):
      for j in range(M.shape[1]):
         if M[i,j] == 1:
            array[i] = j
   return array

def indexArray2stringArray(arr):
   stateList = np.load("stateList.npy", allow_pickle=True).tolist()
   newarr = []
   for i,el in enumerate(arr):
      newarr.append(stateList[int(el)])
   return newarr
def stateArray2phonemeArray(arr):
   phonemes = ['ow','z', 'iy', 'r','w', 'ah', 'n', 'uw','th','f', 'ao','ay', 'v','s', 'ih', 'k','eh','ey', 't','sil','sp']
   
   for i,state in enumerate(arr):
      for phoneme in phonemes:
         if re.search("^%s" % phoneme,state):
            arr[i] = phoneme
   return arr
      
def mergePhonemes(arr):
   newarr = [arr[0]]
   oldel = arr[0]
   for el in arr:
      if el != oldel:
         newarr.append(el)
      oldel = el
   return newarr

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
