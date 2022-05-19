from importlib.machinery import WindowsRegistryFinder
import numpy as np
from lab3_tools import *
import os
import lab3_toolsSPYDER as tools
import lab3_proto as proto
import lab1_proto as proto1
import lab1_tools as tools1
import lab2_proto as proto2
import lab2_tools as tools2

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