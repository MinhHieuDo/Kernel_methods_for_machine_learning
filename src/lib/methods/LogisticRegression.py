import numpy as np
import math
import matplotlib.pyplot as plt


from .baseKernel import baseKernel
from ..tools.utils import sigmoid
from ..tools.utils import timeit





# Weighted Kernel Ridge Regression
class  WKRR():
    def __init__(self):
        self.alpha_ = None

    def fit(self,K_train,y_train,w,lambda_reg=0.01):
        #y_train = y_train.reshape(-1,1)
        n = K_train.shape[0]
        if w is None:
            w = np.ones(n)
        # compute W12
        W12 = np.diag(np.sqrt(w))
        inv = np.linalg.inv(W12.dot(K_train.dot(W12))+ n*lambda_reg*np.eye(n))
        self.alpha_ = W12.dot(inv.dot(W12.dot(y_train)))
        return self.alpha_

class KLR(baseKernel):
    def __init__(self,lambda_reg=0.01, niters=1000,tolerance=1.e-5, **kwargs):
        super(KLR, self).__init__(**kwargs)
        self.alpha_ = 0
        self.lambda_reg_ = kwargs.get('lambda_reg',lambda_reg)
        self.niters_ = niters
        self.tolerance_ = tolerance

    def get_coef(self):
        return list(self.alpha_)

    def fit_use_K(self,K_train,y_train):
        '''
        K: gram matrix  np.array(n_samples_train,n_samples_train)
        Y: label        np.array(n_sample_train)
        niters    : the number of iteration
        tolerance :  stopping criteria
        lambda_reg : lambda regularization 

        '''
        with timeit('Fit with Logistic Regression ', font_style='bold', bg='Red', fg='White'):
            print('lambda_reg =',self.lambda_reg_)
            n= K_train.shape[0]
            self.alpha_ = np.random.rand(n)
            
            # solving KLP by IRLS
            for i in range(self.niters_):
                alpha_old = self.alpha_

                M = K_train.dot(self.alpha_)
                sig_pos, sig_neg = sigmoid(M * y_train), sigmoid(-M * y_train)
                W = sig_neg * sig_pos
                Z = M + y_train / np.maximum(sig_pos, 1.e-6)
                wkrr = WKRR()
                self.alpha_ = wkrr.fit(K_train=K_train,y_train=Z,w=W,lambda_reg=self.lambda_reg_)
                if np.linalg.norm(self.alpha_ - alpha_old) < self.tolerance_:
                    break
                if i == self.niters_-1:
                    print('Warning: please increase the number of iteration to ensure the convergence')
        return self.alpha_
    
    def predict_prob_use_K(self,K_test):
        '''
        K_test : gram matrix for test set np.array(n_samples_test,n_samples_train)
        return : probability for each data
        '''
        prediction = sigmoid(K_test.dot(self.alpha_))
        return prediction

    def predict_use_K(self,K_test):
        with timeit('Predict with Logistic Regression ', font_style='bold', bg='Red', fg='White'):
            prediction = np.array(self.predict_prob_use_K(K_test)>0.5,dtype=int)
            prediction[prediction ==0] = -1
        return prediction

    # def fit(self,X_train,y_train):
    #     self.X_train = X_train
    #     K_train = self.kernel_function_(X_train, X_train)
    #     return self.fit_use_K(K_train, y_train)
    #
    # def predict(self,X_test):
    #     K_test = self.kernel_function_(X_test, self.X_train)
    #     return self.predict_use_K(K_test)






