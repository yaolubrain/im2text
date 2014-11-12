# numpy class for multimodal log-bilinear LM

import numpy as np
import theano
import theano.tensor as T
import sys
from utils import stop
from utils import lm_tools
from scipy.optimize import check_grad
from scipy.sparse import vstack
from numpy.random import RandomState
import time


class MLBL_WORD2VEC(object):
    """
    Multimodal Log-bilinear language model trained using SGD
    """
    def __init__(self,
                 name='lbl',
                 loc='models/mlbl_word2vec.pkl',
                 seed=1234,
                 criteria='validation_pp',
                 k=5,
                 V=1000,
                 K=300,
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
#                else:
#                    R[:,i] = embed_map['*UNKNOWN*']
        else:
            r = np.sqrt(6) / np.sqrt(self.K + self.V + 1)
            R = prng.rand(self.K, self.V) * 2 * r - r

        bw = np.zeros((1, self.V))

        # Context 
        C = 0.01 * prng.randn(self.context, self.K, self.K)

        # Image context
        M = 0.01 * prng.randn(self.h, self.K)

        # Hidden layer
        r = np.sqrt(6) / np.sqrt(self.D + self.h + 1)
        J = prng.rand(self.D, self.h) * 2 * r - r
        bj = np.zeros((1, self.h))

        R = theano.shared(R.astype(theano.config.floatX), borrow=True)
        C = theano.shared(C.astype(theano.config.floatX), borrow=True)
        bw = theano.shared(bw.astype(theano.config.floatX), borrow=True)
        M = theano.shared(M.astype(theano.config.floatX), borrow=True)
        J = theano.shared(J.astype(theano.config.floatX), borrow=True)
        bj = theano.shared(bj.astype(theano.config.floatX), borrow=True)

        self.R = R
        self.C = C
        self.bw = bw
        self.M = M
        self.J = J
        self.bj = bj

    def forward(self, X, Im):
        """
        Feed-forward pass through the model
        X: ('batchsize' x 'context') matrix of word indices
        """
        batchsize = X.shape[0]
        R = self.R
        C = self.C
        M = self.M
        bw = self.bw
        J = self.J
        bj = self.bj

        # Forwardprop images
        Im = T.concatenate((Im, T.ones((batchsize, 1))), 1)
        IF = T.dot(Im, T.concatenate((J, bj)))
        IF = IF * (IF > 0)

        # Obtain word features
#        tmp = R.as_numpy_array()[:,X.flatten()].flatten(order='F')
#        tmp = tmp.reshape((batchsize, self.K * self.context))
#        words = np.zeros((batchsize, self.K, self.context))
#        for i in range(batchsize):
#            words[i,:,:] = tmp[i,:].reshape((self.K, self.context), order='F')
#        words = gpu.garray(words)

        words = R[:,X.flatten()].transpose().flatten().reshape((batchsize, self.context, self.K)).dimshuffle([0, 2, 1])
         
        # Compute the hidden layer (predicted next word representation)
#        acts = gpu.zeros((batchsize, self.K))
#        for i in range(self.context):
#            acts = acts + gpu.dot(words[:,:,i], C[i,:,:])
#        acts = acts + gpu.dot(IF, M)
#        acts = gpu.concatenate((acts, gpu.ones((batchsize, 1))), 1)
#        def oneMatMult(i, A, B):
#            return T.dot(A[:,:,i], B[i,:,:])

#        matMults, updates = theano.scan(fn=oneMatMult, sequences=T.arange(self.context), non_sequences=[words, C])

        acts = T.dot(words[:,:,0], C[0,:,:]) \
             + T.dot(words[:,:,1], C[1,:,:]) \
             + T.dot(words[:,:,2], C[2,:,:]) \
             + T.dot(words[:,:,3], C[3,:,:]) \
             + T.dot(words[:,:,4], C[4,:,:])

#        acts = matMults.sum()
        acts = acts + T.dot(IF, M)
        acts = T.concatenate((acts, T.ones((batchsize, 1))), 1)

#        # Compute softmax
#        preds = gpu.dot(acts, gpu.concatenate((R, bw)))
#        preds = gpu.exp(preds - preds.max(1).reshape(batchsize, 1))
#        denom = preds.sum(1).reshape(batchsize, 1)
#        preds = gpu.concatenate((preds / denom, gpu.ones((batchsize, 1))), 1)
        preds = T.dot(acts, T.concatenate((R, bw)))
        preds = T.exp(preds - preds.max(1).reshape((batchsize, 1)))
        denom = preds.sum(1, keepdims=True)
        preds = T.concatenate((preds / denom, T.ones((batchsize, 1))), 1)

        return (words, acts, IF, preds)


    def objective(self, Y, preds):
        """
        Compute the objective function
        """
        batchsize = Y.shape[0]

        # Cross-entropy
#        C = -np.sum(Y.multiply(np.log(preds[:,:-1] + 1e-20))) / batchsize
        C = -T.sum(T.mul(Y, (T.log(preds[:,:-1] + 1e-20)))) / batchsize
        return C

    def update_params(self, objective, X, lr, mom):
        """
        Update the network parameters using the computed gradients
        """
        batchsize = X.shape[0]
#        self.deltaC = self.p_t * self.deltaC - \
#            (1 - self.p_t) * (self.eta_t / batchsize) * self.dC
#        self.deltaR = self.p_t * self.deltaR - \
#            (1 - self.p_t) * (self.eta_t / batchsize) * self.dR
#        self.deltaB = self.p_t * self.deltaB - \
#            (1 - self.p_t) * (self.eta_t / batchsize) * self.db
#        self.deltaM = self.p_t * self.deltaM - \
#            (1 - self.p_t) * (self.eta_t / batchsize) * self.dM
#        self.deltaJ = self.p_t * self.deltaJ - \
#            (1 - self.p_t) * (self.eta_t / batchsize) * self.dJ
#        self.deltaBj = self.p_t * self.deltaBj - \
#            (1 - self.p_t) * (self.eta_t / batchsize) * self.dBj
#        self.C = self.C + self.deltaC
#        self.R = self.R + self.deltaR
#        self.bw = self.bw + self.deltaB
#        self.M = self.M + self.deltaM
#        self.J = self.J + self.deltaJ
#        self.bj = self.bj + self.deltaBj

#        params = [self.R, self.C, self.bw, self.M, self.J, self.bj] 
        params = [self.C, self.bw, self.M, self.J, self.bj] 
        updates = []
        for param in params:
            delta = theano.shared(param.get_value()*0.)
            updates.append((delta, mom * delta - ((1. - mom) * (lr / batchsize) * T.grad(objective, param)).astype(theano.config.floatX)))
            updates.append((param, param + delta))

        return updates

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

        x = T.matrix('x', dtype='int32')
        y = T.matrix('y')
        im = T.matrix('im')
        lr = T.scalar('lr')
        mom = T.scalar('mom')
        (words, acts, IF, preds) = self.forward(x, im)
        obj_T = self.compute_obj(x, im, y)
        compute_obj_T = theano.function([x, im, y], obj_T)
        train_batch = theano.function([x, im, y, lr, mom], obj_T, 
                                      updates=self.update_params(obj_T, x, lr, mom), 
                                      on_unused_input='warn')
        # Main loop
        stop.display_phase(1)
        for epoch in range(self.maxepoch):
            self.epoch = epoch
            tic = time.time()
            prng = RandomState(self.seed + epoch + 1)
            prng.shuffle(inds)
            obj = 0.0
            for minibatch in range(numbatches):
#            for minibatch in range(1):
                batchX = X[inds[minibatch::numbatches]].astype(np.int32)
                batchY = XY[inds[minibatch::numbatches]].toarray().astype(theano.config.floatX)
                batchindX = indX[inds[minibatch::numbatches]].astype(np.int32).flatten()
                batchIm = IM[batchindX].astype(theano.config.floatX)
                
                one_obj = train_batch(batchX, batchIm, batchY, self.eta_t, self.p_t)
                obj += one_obj

                print self.C[0,0,0].eval(), self.bw[0,0].eval(), self.bj[0,0].eval(), self.J[0,0].eval(), self.M[0,0].eval()

            self.update_hyperparams()

            toc = time.time()
            # Results and stopping criteria
#            obj_train = compute_obj_T(X[:num].astype(np.int32), 
#                              IM[indX[:num].astype(int).flatten()].astype(theano.config.floatX), 
#                              XY[:num].toarray().astype(theano.config.floatX))
            obj_val = compute_obj_T(V[:num].astype(np.int32), 
                                  IM[indV[:num].astype(int).flatten()].astype(theano.config.floatX), 
                                  VY[:num].toarray().astype(theano.config.floatX))

            stop.display_results(epoch, toc-tic, obj, obj_val)
'''            
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
#                    self = stop.load_model(self.loc)

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
'''          
      
def main():
    pass

if __name__ == '__main__':
    main()


        
