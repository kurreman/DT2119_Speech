import numpy as np
from lab2_tools import *
import time

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

    N = log_emlik.shape[0]
    M = log_emlik.shape[1]

    backward_prob = np.zeros((N,M))

    for i in reversed(range(M)):
        backward_prob[N-1,i] = 0
        for n in reversed(range(N-1)): #skips n=0 since the init state is defined above
            backward_prob[n,i] = logsumexp(log_transmat[i,:-1]+log_emlik[n+1,:]+backward_prob[n+1,:]) #-1 on transmat to skip the last
    
    return backward_prob

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
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]

    viterbi_loglik = np.zeros((N,M))
    viterbi_path = np.zeros((N,M)) #TODO 1 !!!!
    #TODO 2 forceFinalState!! 

    for j in range(M):
        viterbi_loglik[0,j] = log_startprob[j] + log_emlik[0,j]
        viterbi_path[0,j] = 0
        for n in range(1,N):
            viterbi_loglik[n,j] = np.max(viterbi_loglik[n-1,:]+log_transmat[:-1,j]) + log_emlik[n,j] #-1 on transmat to skip the last
            viterbi_path[n,j] = np.argmax(viterbi_loglik[n-1,:]+log_transmat[:-1,j])
            #viterbi_path.append(np.argmax(viterbi_loglik[n-1,:]+log_transmat[:-1,j]))
            #viterbi_path[j].append(np.argmax(viterbi_loglik[n-1,:]+log_transmat[:-1,j]))
    
    return viterbi_loglik,viterbi_path

def viterbi2(log_emission_lik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emission_lik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    n_states = log_startprob.shape[0] - 1
    n_frames = log_emission_lik.shape[0]

    log_prop_path = np.zeros((n_frames, n_states))
    argmax_paths = np.zeros((n_frames, n_states), dtype=int)

    for j in range(n_states):
        log_prop_path[0][j] = log_startprob[j] + log_emission_lik[0][j]

    for i in range(1, n_frames):
        for j in range(n_states):
            temp = log_prop_path[i - 1, :] + log_transmat[:, j]
            log_prop_path[i][j] = temp.max() + log_emission_lik[i][j]
            argmax_paths[i][j] = temp.argmax()

    viterbi_path = np.zeros(n_frames, dtype=int)
    if forceFinalState:
        viterbi_path[-1] = n_states - 1
    else:
        viterbi_path[-1] = log_prop_path[-1, :].argmax()

    for i in range(2, n_frames + 1):
        viterbi_path[-i] = argmax_paths[-i + 1][viterbi_path[-i + 1]]

    return log_prop_path[-1, :].max(), viterbi_path

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N = log_alpha.shape[0]
    M = log_alpha.shape[1]
    log_gamma = np.zeros((N,M))
    for i in range(M):
        for n in range(N):
            log_gamma[n,i] = log_alpha[n,i] + log_beta[n,i] - logsumexp(log_alpha[N-1,:])
    for tstep in range(N):
        if not np.isclose(np.sum(np.exp(log_gamma[tstep,:])),1):
            print("State posteriors don't sum to 1!!")
    return log_gamma

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
    # N = X.shape[0]
    # D = X.shape[1]
    # M = log_gamma.shape[1]
    # gamma = np.exp(log_gamma)

    # print(X.shape, log_gamma.shape)

    # means = np.zeros((M,D))
    # covars = np.zeros((M,D))
    # for j in range(D):
    #     means[:,j] = np.sum(np.multiply(gamma[:,j],X[:,j]))/np.sum(gamma[:,j])
    #     # covars[:,j] = np.sum(np.multiply(gamma[:,j],np.multiply(X[:,j]-means[:,j],(X[:,j]-means[:,j]).T)))/np.sum(gamma[:,j])
    #     covars[:,j] = np.sum(np.multiply(gamma[:,j],np.multiply(X[j,:]-means[j,:],(X[j,:]-means[j,:]).T)))/np.sum(gamma[:,j])
    # # for j in range(D):
    # #     for m in range(M):

    # # for m in range(M):
    # #     means[m,:] = 
    # #     covars[m,:] = 

    # return means, covars

    n_states = log_gamma.shape[1]
    n_features = X.shape[1]

    normalizer = np.diag(1 / np.exp(logsumexp(log_gamma)))

    new_means = np.dot(normalizer, np.dot(np.exp(log_gamma.T), X))

    new_covars = np.zeros((n_states, n_features))

    for index in range(n_states):
        new_covars[index, :] = np.dot(
            np.exp(log_gamma.T[index, :]), ((X - new_means[index]) ** 2)
        )

    new_covars = np.dot(normalizer, new_covars)

    new_covars = np.clip(new_covars, varianceFloor, None)

    return new_means, new_covars



def loglik(logalphaN):
    """ Input: forward log probabilities for each of the M states in the model in the last step N | log_alpha
        Output: log likelihood of whole sequence X until last step N-1"""
    return logsumexp(logalphaN)

def score_data(data,wordHMMs,algorithm="forward"):
    """Scores data with final loglikelihood.
        Args:
            data: Contains 44 samples of different speakers
            wordHMMs: HMMs for words"""
    Tstart = time.time()
    score_table = {}
    for datapt in data: 
        utt_file = datapt["filename"]
        score_table[utt_file] = {}
        for word in wordHMMs.keys():
            obsloglik = log_multivariate_normal_density_diag(datapt['lmfcc'],wordHMMs[word]["means"],wordHMMs[word]["covars"])
            if algorithm=="forward":
                logalpha = forward(obsloglik,np.log(wordHMMs[word]["startprob"]),np.log(wordHMMs[word]["transmat"]))
                loglik_val = loglik(logalpha[-1,:])
            elif algorithm=="viterbi":
                v_val = viterbi(obsloglik,np.log(wordHMMs[word]["startprob"]),np.log(wordHMMs[word]["transmat"]))[0]
                loglik_val = np.max(v_val[-1,:])
            score_table[utt_file][word] = loglik_val
            score_table[utt_file]["digit"] = datapt["digit"]
    Tend = time.time()
    print("Runtime: ",Tend-Tstart, "s", " | Algorithm: ", algorithm)
    return score_table

def table2matrix(score_table):
    """"""
    score_matrix = np.zeros((44,11))
    word_list = []
    digit_list = []
    i=0
    for utt_id in score_table.keys():
        score_list = []
        word_list = []
        digit_list.append(score_table[utt_id]["digit"])
        for word in score_table[utt_id].keys():
            if word != "digit":
                score_list.append(score_table[utt_id][word])
                word_list.append(word)
        score_matrix[i,:] = np.array(score_list)
        i+=1
    return score_matrix, word_list, digit_list

def pickbestfit(score_matrix, word_list):
    """"""
    best_fit_list = []
    for i in range(score_matrix.shape[0]):
        highest_lik_id = np.argmax(score_matrix[i,:])
        best_fit_list.append(word_list[highest_lik_id])
    return best_fit_list
    
def percentCorrect(list1,list2):
    """Returns equality percentage of elementwise comparison between list 1 and 2 of same length"""
    max_score = len(list1)
    score = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            score += 1
    return score/(max_score)

def getlogAlphaBetaGamma(utterance,HMMmodel): 
    obsloglik = log_multivariate_normal_density_diag(utterance['lmfcc'],HMMmodel["means"],HMMmodel["covars"])
    logalpha = forward(obsloglik,np.log(HMMmodel["startprob"]),np.log(HMMmodel["transmat"]))
    logbeta = backward(obsloglik,np.log(HMMmodel["startprob"]),np.log(HMMmodel["transmat"]))
    loggamma = statePosteriors(logalpha,logbeta)

    return logalpha, logbeta,loggamma


