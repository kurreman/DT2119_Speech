import numpy as np
from lab2_tools import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """

    twoHMMs = {}
    PI = hmm1["startprob"]
    P = hmm2["startprob"]
    A = hmm1["transmat"]
    B = hmm2["transmat"]
    
    M1 = len(hmm1["startprob"])-1 #-1 for non emitting state
    M2 = len(hmm2["startprob"])-1
    K = M1+M2
    PIconcat = np.zeros((K+1))
    Aconcat = np.zeros((K+1,K+1))

    #OBS this "hardcoded" approach will create issues if used with a model with other than 3 states. 
    #Example "sp" has only 2 states, important for lab 3 if reusing code. 

    PIconcat[0:M1] = PI[0:M1]
    PIconcat[M1:] = PI[M1]*P
    
    Aconcat[0:M1,0:M1] = A[0:M1,0:M1]
    Aconcat[M1:,0:M1] = np.zeros((4,M1))
    for i in range(M1+1):
        Aconcat[i,M1:] = A[i,M1]*P
        #Aconcat[1,3:] = A[1,3]*P
        #Aconcat[2,3:] = A[2,3]*P
    Aconcat[M1:-1,M1:] = B[0:M2,:]
    temp = np.zeros(K-M1+1)
    temp[-1] = 1
    Aconcat[-1,M1:] = temp

    twoHMMs["startprob"] = PIconcat
    twoHMMs["transmat"] = Aconcat
    twoHMMs["means"] = np.concatenate((hmm1["means"][0:M1+1,:],hmm2["means"][0:M2+1,:]))
    twoHMMs["covars"] = np.concatenate((hmm1["covars"][0:M1+1,:],hmm2["covars"][0:M2+1,:]))

    return twoHMMs


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states | log phi_j(x_i)
        log_startprob: log probability to start in state i | log pi_i
        log_transmat: log transition probability from state i to j | log a_ij

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

    N = log_emlik.shape[0]
    M = log_emlik.shape[1]

    forward_prob = np.zeros((N,M))

    for j in range(M):
        forward_prob[0,j] = log_startprob[j] + log_emlik[0,j]
        for n in range(1,N): #skips n=0 since the init state is defined above
            #print(forward_prob[n-1,:].shape, log_transmat[:-1,j].shape)
            forward_prob[n,j] = logsumexp(forward_prob[n-1,:]+log_transmat[:-1,j]) + log_emlik[n,j] #-1 on transmat to skip the last
    
    return forward_prob
            


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """

def loglik(logalphaN):
    """ Input: forward log probabilities for each of the M states in the model in the last step N | log_alpha
        Output: log likelihood of whole sequence X until last step N-1"""
    return logsumexp(logalphaN)
