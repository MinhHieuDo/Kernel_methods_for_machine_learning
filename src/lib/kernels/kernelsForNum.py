import numpy as np

def linear_kernel(X1, X2=None,**kwargs):
    if X2 is None:
        X2 = X1
    K = X1.dot(X2.T)
    return K

def polynomial_kernel(X1, X2=None, **kwargs):
    degree = kwargs.get('degree',2)
    if X2 is None:
        X2 = X1
    return (1 + linear_kernel(X1, X2))**degree


def rbf_kernel(X1, X2=None,**kwargs):
    gamma = kwargs.get('gamma',1)
    if X2 is None:
        X2 = X1
    X1_norm = np.sum(X1**2,axis=-1)
    X2_norm = np.sum(X2**2,axis=-1)
    X1_dot_X2 = np.matmul(X1, X2.T)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * X1_dot_X2))
    return K