import numpy as np
from ..kernels.kernelsForNum import linear_kernel, polynomial_kernel, rbf_kernel
from ..kernels.kernelsForString import get_spectrum_kernel,get_mismatch_kernel


class baseKernel(object):
    dict_numeric_kernels = {'linear': linear_kernel,'polynomial': polynomial_kernel,'rbf': rbf_kernel,}
    dict_string_kernels = {'spectrum': get_spectrum_kernel,'mismatch': get_mismatch_kernel}

    def __init__(self, kernel_name='rbf', **kwargs):
        self.kernel_name_     = kernel_name

        self.build_kernel_function()
        self.get_kernel_params(**kwargs)

    def build_kernel_function(self):
        if self.kernel_name_ in self.dict_numeric_kernels.keys():
            self.kernel_function_ = self.dict_numeric_kernels[self.kernel_name_]
        elif self.kernel_name_ in self.dict_string_kernels.keys():
            self.kernel_function_ = self.dict_string_kernels[self.kernel_name_]
        else:
            print("PLease check again your kernel! It is not implemented")

    def get_kernel_params(self,**kwargs):

        self.kernel_params_ = {}
        # numeric kernel
        if self.kernel_name_ == 'rbf':
            self.kernel_params_['gamma'] = kwargs.get('gamma', 1)
        if self.kernel_name_ == 'polynomial':
            self.kernel_params_['degree'] = kwargs.get('degree',2)
        # string kernel
        if self.kernel_name_ == 'spectrum':
            self.kernel_params_['k'] = kwargs.get('k',5)
        if self.kernel_name_ == 'mismatch':
            self.kernel_params_['k'] = kwargs.get('k',5)
            self.kernel_params_['m'] = kwargs.get('m',1)

    def build_gram_matrix(self,X,Y=None):
        #print('params',self.kernel_params_)
        K = self.kernel_function_(X,Y,**self.kernel_params_)
        return K


    # use data
    def fit(self,X_train,y_train):
        self.X_train = X_train
        K_train = self.build_gram_matrix(X_train,None)
        return self.fit_use_K(K_train,y_train)

    def predict(self,X_test):
        K_test = self.build_gram_matrix(X_test,self.X_train)
        return self.predict_use_K(K_test)

    def fit_use_K(self,K_train, y_train):
        pass
    def predict_use_K(self,K_test):
        pass

