# numpy class for multimodal log-bilinear LM

import numpy as np
import gnumpy as gpu
import sys
from utils import stop
from utils import lm_tools
from scipy.optimize import check_grad
from scipy.sparse import vstack
from numpy.random import RandomState
import time
gpu.max_memory_usage = 5454000000


class MLBL(object):
    """
    Multimodal Log-bilinear language model trained using SGD
    """
    def __init__(self,
                 name='lbl',
                 loc='models/mlbl.pkl',
                 seed=1234,
                 criteria='validation_pp',
                 k=5,
                 V=1000,
                 K=20,
                 D=4096,
                 h=256,
                 context=2,
                 batchsize=100,
                 maxepoch=500,
                 eta_t=0.1,
                 gamma_r=1e-4,
                 gamma_c=1e-5,
                 f=0.998,
                 p_i=0.5,
                 p_f=0.99,
                 T=500.0,
                 verbose=1):
        """
        name: name of the network
        loc: location to save model files
        seed: random seed
        criteria: when to stop training
        k: validation interval before stopping
        V: vocabulary size
        K: embedding dimensionality
        D: dimensionality of the image features
        h: intermediate layer dimensionality
        context: word context length
        batchsize: size of the minibatches
        maxepoch: max number of training epochs
        eta_t: learning rate
        gamma_r: weight decay for representations
        gamma_c: weight decay for contexts
        f: learning rate decay
        p_i: initial momentum
        p_f: final momentum
        T: number of epochs until p_f is reached (linearly)
        verbose: display progress
        """
        self.name = name
        self.loc = loc
        self.criteria = criteria
        self.seed = seed
        self.k = k
        self.V = V
        self.K = K
        self.D = D
        self.h = h
        self.context = context
        self.batchsize = batchsize
        self.maxepoch = maxepoch
        self.eta_t = eta_t
        self.gamma_r = gamma_r
        self.gamma_c = gamma_c
        self.f = f
        self.p_i = p_i
        self.p_f = p_f
        self.T = T
        self.verbose = verbose
        self.p_t = (1 - (1 / T)) * p_i + (1 / T) * p_f


    def init_params(self, embed_map, count_dict, L):
        """
        Initializes embeddings and context matricies
        """
        prng = RandomState(self.seed)

        # Pre-trained word embedding matrix
        if embed_map != None:
            R = np.zeros((self.K, self.V))
            for i in range(self.V):
                word = count_dict[i]
                if word in embed_map:
                    R[:,i] = embed_map[word]
                else:
                    R[:,i] = embed_map['*UNKNOWN*']
            R = gpu.garray(R)
        else:
            r = np.sqrt(6) / np.sqrt(self.K + self.V + 1)
            R = prng.rand(self.K, self.V) * 2 * r - r
            R = gpu.garray(R)
        bw = gpu.zeros((1, self.V))

        # Context 
        C = 0.01 * prng.randn(self.context, self.K, self.K)
        C = gpu.garray(C)

        # Image context
        M = 0.01 * prng.randn(self.h, self.K)
        M = gpu.garray(M)

        # Hidden layer
        r = np.sqrt(6) / np.sqrt(self.D + self.h + 1)
        J = prng.rand(self.D, self.h) * 2 * r - r
        J = gpu.garray(J)
        bj = gpu.zeros((1, self.h))

        # Initial deltas used for SGD
        deltaR = gpu.zeros(np.shape(R))
        deltaC = gpu.zeros(np.shape(C))
        deltaB = gpu.zeros(np.shape(bw))
        deltaM = gpu.zeros(np.shape(M))
        deltaJ = gpu.zeros(np.shape(J))
        deltaBj = gpu.zeros(np.shape(bj))

        self.R = R
        self.C = C
        self.bw = bw
        self.M = M
        self.J = J
        self.bj = bj
        self.deltaR = deltaR
        self.deltaC = deltaC
        self.deltaB = deltaB
        self.deltaM = deltaM
        self.deltaJ = deltaJ
        self.deltaBj = deltaBj


    def forward(self, X, Im, test=False):
        """
        Feed-forward pass through the model
        X: ('batchsize' x 'context') matrix of word indices
        """
        batchsize = X.shape[0]
        Im = gpu.garray(Im)
        R = self.R
        C = self.C
        M = self.M
        bw = self.bw
        J = self.J
        bj = self.bj

        # Forwardprop images
        Im = gpu.concatenate((Im, gpu.ones((batchsize, 1))), 1)
        IF = gpu.dot(Im, gpu.concatenate((J, bj)))
        IF = IF * (IF > 0)

        # Obtain word features
        tmp = R.as_numpy_array()[:,X.flatten()].flatten(order='F')
        tmp = tmp.reshape((batchsize, self.K * self.context))
        words = np.zeros((batchsize, self.K, self.context))
        for i in range(batchsize):
            words[i,:,:] = tmp[i,:].reshape((self.K, self.context), order='F')
        words = gpu.garray(words)

        # Compute the hidden layer (predicted next word representation)
        acts = gpu.zeros((batchsize, self.K))
        for i in range(self.context):
            acts = acts + gpu.dot(words[:,:,i], C[i,:,:])
        acts = acts + gpu.dot(IF, M)
        acts = gpu.concatenate((acts, gpu.ones((batchsize, 1))), 1)

        # Compute softmax
        preds = gpu.dot(acts, gpu.concatenate((R, bw)))
        preds = gpu.exp(preds - preds.max(1).reshape(batchsize, 1))
        denom = preds.sum(1).reshape(batchsize, 1)
        preds = gpu.concatenate((preds / denom, gpu.ones((batchsize, 1))), 1)

        return (words, acts, IF, preds.as_numpy_array())


    def objective(self, Y, preds):
        """
        Compute the objective function
        """
        batchsize = Y.shape[0]

        # Cross-entropy
        C = -np.sum(Y.multiply(np.log(preds[:,:-1] + 1e-20))) / batchsize
        return C


    def backward(self, Y, preds, IF, acts, words, X, Im):
        """
        Backward pass through the network
        """
        batchsize = preds.shape[0]
        Im = gpu.garray(Im)

        # Compute part of df/dR
        Ix = gpu.garray(preds[:,:-1] - Y) / batchsize
        delta = gpu.dot(acts.T, Ix)
        dR = delta[:-1,:] + self.gamma_r * self.R
        db = delta[-1,:]
        dR = dR.as_numpy_array()

        # Compute df/dC and word inputs for df/dR
        Ix = gpu.dot(Ix, self.R.T)
        dC = gpu.zeros(np.shape(self.C))
        for i in range(self.context):
            delta = gpu.dot(words[:,:,i].T, Ix)
            dC[i,:,:] = delta + self.gamma_c * self.C[i,:,:]
            delta = gpu.dot(Ix, self.C[i,:,:].T)
            delta = delta.as_numpy_array()
            for j in range(X.shape[0]):
                dR[:,X[j,i]] = dR[:,X[j,i]] + delta.T[:,j]

        # Compute df/dM
        dM = gpu.dot(IF.T, Ix) + self.gamma_c * self.M

        # Compute df/dJ
        Ix = gpu.dot(Ix, self.M.T) * (IF > 0)
        Im = gpu.concatenate((Im, gpu.ones((batchsize, 1))), 1)
        delta = gpu.dot(Im.T, Ix)
        dJ = delta[:-1,:] + self.gamma_c * self.J
        dBj = delta[-1,:]

        self.dR = gpu.garray(dR)
        self.dM = dM
        self.db = db
        self.dC = dC
        self.dJ = dJ
        self.dBj = dBj


    def update_params(self, X):
        """
        Update the network parameters using the computed gradients
        """
        batchsize = X.shape[0]
        self.deltaC = self.p_t * self.deltaC - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dC
        self.deltaR = self.p_t * self.deltaR - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dR
        self.deltaB = self.p_t * self.deltaB - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.db
        self.deltaM = self.p_t * self.deltaM - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dM
        self.deltaJ = self.p_t * self.deltaJ - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dJ
        self.deltaBj = self.p_t * self.deltaBj - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dBj

        self.C = self.C + self.deltaC
        self.R = self.R + self.deltaR
        self.bw = self.bw + self.deltaB
        self.M = self.M + self.deltaM
        self.J = self.J + self.deltaJ
        self.bj = self.bj + self.deltaBj


    def update_hyperparams(self):
        """
        Updates the learning rate and momentum schedules
        """
        self.eta_t = self.eta_t * self.f
        if self.epoch < self.T:
            self.p_t = (1 - ((self.epoch + 1) / self.T)) * self.p_i + \
                ((self.epoch + 1) / self.T) * self.p_f
        else:
            self.p_t = self.p_f


    def compute_obj(self, X, Im, Y):
        """
        Perform a forward pass and compute the objective
        """
        preds = self.forward(X, Im)[-1]
        obj = self.objective(Y, preds)
        return obj


    def compute_pp(self, Xp, Im, word_dict):
        """
        Compute the model perplexity
        """
        return lm_tools.perplexity(self, Xp, word_dict, Im, self.context)


    def train(self, X, indX, XY, V, indV, VY, IM, count_dict, word_dict, embed_map):
        """
        Trains the LBL
        """
        self.start = self.seed
        self.init_params(embed_map, count_dict, XY)
        inds = np.arange(len(X))
        numbatches = len(inds) / self.batchsize
        curr = 1e20
        counter = 0
        target=None
        num = 15000

        # Main loop
        stop.display_phase(1)
        for epoch in range(self.maxepoch):
            self.epoch = epoch
            tic = time.time()
            prng = RandomState(self.seed + epoch + 1)
            prng.shuffle(inds)
            for minibatch in range(numbatches):

                batchX = X[inds[minibatch::numbatches]]
                batchY = XY[inds[minibatch::numbatches]]
                batchindX = indX[inds[minibatch::numbatches]].astype(int).flatten()
                batchIm = IM[batchindX]
            
                (words, acts, IF, preds) = self.forward(batchX, batchIm)
                self.backward(batchY, preds, IF, acts, words, batchX, batchIm)
                self.update_params(batchX)

            self.update_hyperparams()
            toc = time.time()

            # Results and stopping criteria
            obj = self.compute_obj(X[:num], IM[indX[:num].astype(int).flatten()], XY[:num])
            obj_val = self.compute_obj(V[:num], IM[indV[:num].astype(int).flatten()], VY[:num])

            if self.verbose > 0:
                stop.display_results(epoch, toc-tic, obj, obj_val)
            (curr, counter) = stop.update_result(curr, obj_val, counter)
            if counter == 0:
                stop.save_model(self, self.loc)
                stopping_target = obj

            if stop.criteria_complete(self, epoch, curr, obj, counter, 
                self.k, obj_val, target):
                if self.criteria == 'maxepoch':
                    break
                elif self.criteria == 'validation_pp':
                    self = stop.load_model(self.loc)
                    counter = 0
                    X = np.r_[X, V]
                    XY = vstack([XY, VY]).tocsr()
                    indX = np.r_[indX, indV]
                    self.criteria = 'll_train_heldout'
                    target = stopping_target   #obj
                    stop.display_phase(2)
                    inds = range(X.shape[0])
                    prng.shuffle(inds)
                    numbatches = len(inds) / self.batchsize
                elif self.criteria == 'll_train_heldout':
                    break
          
      
def main():
    pass

if __name__ == '__main__':
    main()


        
