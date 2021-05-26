import numpy as np
import cvxopt
import quadprog
from .baseKernel import baseKernel
from ..tools.utils import timeit

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    n = P.shape[1]
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A, (1, n), "d"), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args,options={'show_progress': False})
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


class KSVM(baseKernel):
    '''
    Solve the KSVM problem:
        minimize    (1/2)*x^T*K*x - y^T*x
        subject to  0 <= yi*xi <= C
    '''
    def __init__(self,C=0.1, tolerance=1.e-5, optimization='cvxopt', **kwargs):
        super(KSVM, self).__init__(**kwargs)
        self.alpha_ = 0
        self.C_ = kwargs.get('C',C)
        self.tol = tolerance
        self.optim = optimization
        
    def get_coef(self):
        '''
        get_coef: Return the alpha parameter of the class
        Paramaters: -
        Return: the model parameters
        '''
        return list(self.alpha_)

    def fit_use_K(self,K_train,y_train):
        with timeit('Fit with SVM ', font_style='bold', bg='Red', fg='White'):
            print('C=',self.C_)
            n = K_train.shape[0]
            P = np.diag(y_train).dot(K_train).dot(np.diag(y_train))
            eps = 1.e-12
            P = P + eps*np.eye(n)
            q = -np.ones(n)

            G = np.vstack([-np.eye(n),np.eye(n)])
            h = np.hstack([np.zeros(n), self.C_ * np.ones(n)])

            A = y_train[np.newaxis, :]
            b = np.array([0.])

            if self.optim == 'cvxopt':
                alpha = cvxopt_solve_qp(P,q,G,h,A,b)
            else:
                alpha = quadprog_solve_qp(P,q,G,h,A,b)
                print('Use QP solver! ')

            self.alpha_ = alpha * y_train
            # print('alpha', self.alpha_)
            # compute support vector and bias
            sv = np.logical_and((alpha > self.tol), (self.C_ - alpha > self.tol))
            self.bias_ = np.mean(y_train[sv] - K_train[sv].dot(self.alpha_))



    def predict_prob_use_K(self, K_test):
        prediction = K_test.dot(self.alpha_) + self.bias_
        prediction = np.sign(prediction)
        return prediction

    def predict_use_K(self, K_test):
        with timeit('Predict with SVM ', font_style='bold', bg='Red', fg='White'):
            prediction = np.array(self.predict_prob_use_K(K_test)>=0,dtype=int)
            prediction[prediction ==0]=-1

        return prediction
        

    # def fit(self,X_train,y_train):
    #     self.X_train = X_train
    #     K_train = self.kernel_function_(X_train, X_train)
    #     return self.fit_use_K(K_train, y_train)
    #
    # def predict(self,X_test):
    #     K_test = self.kernel_function_(X_test, self.X_train)
    #     return self.predict_use_K(K_test)
    








