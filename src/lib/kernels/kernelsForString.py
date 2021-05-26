import numpy as np
import itertools
from tqdm import tqdm
from ..tools.utils import timeit


def get_spectrum_feature_map(x,k,Ak):
    '''
    x: string DNA (string)
    k: length of kmer (int)
    Ak: all combination of kmer from the alphabet A

    return feature value at x
    '''
    n = len(Ak)
    phi_k = np.zeros(n)
    
    for id, u in enumerate(Ak):
        for i in range(len(x)-k+1):
            kmer = x[i:i+k]
            if u == kmer:
                phi_k[id] +=1
    return phi_k

def get_spectrum_kernel(X,Y=None,**kwargs):
    '''
    Compute the spectrum kernel K(x,y)
    X and Y ar list of string DNA
    return the gram matrix K(x,y)
    '''
    k = kwargs.get('k',5)
    
    with timeit('Building Gram matrix for spectrum kernel', font_style='bold', bg='Blue', fg='White'):
        print('k=',k)
        Ak = [''.join(c) for c in itertools.product('ACGT', repeat=k)]
        nx = X.shape[0]
        list_phi_x = []
        for i in tqdm(range(nx), desc='Feature vector'):
            xi = X[i]
            list_phi_x.append(get_spectrum_feature_map(xi,k,Ak))

        if Y is None:
            ny = nx
            list_phi_y = list_phi_x
            K = np.zeros((nx,ny))
            for i in tqdm(range(nx), desc='Gram matrix'):
                for j in range(ny):
                    if j>=i:
                        K[i,j] = np.dot(list_phi_x[i], list_phi_y[j])
                        K[j,i] = K[i,j]
        else:
            ny = Y.shape[0]
            list_phi_y = []
            for j in range(ny):
                yj = Y[j]
                list_phi_y.append(get_spectrum_feature_map(yj,k,Ak))
            K = np.zeros((nx,ny))
            for i in tqdm(range(nx),desc='Gram matrix'):
                for j in range(ny):
                    K[i,j] = np.dot(list_phi_x[i], list_phi_y[j])

    return K

# ====================================================================================================================
def is_mismatch_neighbor(m,alpha,beta):
    '''
    alpha, beta : string 
    m : int
    '''
    list_alpha = list(alpha)
    list_beta  = list(beta)
    mismatch = np.sum([list_alpha[i]!= list_beta[i] for i in range(len(list_alpha))])
    if mismatch <=m:
        return True
    else:
        return False
def get_mismatch_feature_map(x,k,m,Ak):
    n = len(Ak)
    phi_km = np.zeros(n)
    
    for id, u in enumerate(Ak):
        for i in range(len(x)-k+1):
            kmer = x[i:i+k]
            if is_mismatch_neighbor(m,kmer,u):
                phi_km[id] +=1
    return phi_km

def get_mismatch_kernel(X,Y=None,**kwargs):
    '''
    Compute the mismatch kernel K(x,y)
    X and Y ar list of string DNA
    return the gram matrix K(x,y)
    '''

    k = kwargs.get('k',5)
    m = kwargs.get('m',1)
    
    with timeit('Building Gram matrix for mismatch kernel', font_style='bold', bg='Blue', fg='White'):
        print('k={} and m={}'.format(k,m))
        Ak = [''.join(c) for c in itertools.product('ACGT', repeat=k)]
        nx = X.shape[0]
        list_phi_x = []
        for i in tqdm(range(nx), desc='Feature vector'):
            xi = X[i]
            list_phi_x.append(get_mismatch_feature_map(xi,k,m,Ak))

        if Y is None:
            ny = nx
            list_phi_y = list_phi_x
            K = np.zeros((nx,ny))
            for i in tqdm(range(nx), desc='Gram matrix'):
                for j in range(ny):
                    if j>=i:
                        K[i,j] = np.dot(list_phi_x[i], list_phi_y[j])
                        K[j,i] = K[i,j]
        else:
            ny = Y.shape[0]
            list_phi_y = []
            for j in range(ny):
                yj = Y[j]
                list_phi_y.append(get_mismatch_feature_map(yj,k,m,Ak))
            K = np.zeros((nx,ny))
            for i in tqdm(range(nx),desc='Gram matrix'):
                for j in range(ny):
                    K[i,j] = np.dot(list_phi_x[i], list_phi_y[j])

    return K

#=====================================================================================
def normalize_K(K):
    """
    Normalize kernel
    
    """
    norms = np.sqrt(K[np.diag_indices_from(K)])
    return (K / norms[np.newaxis,:]) / norms[:,np.newaxis]




            
if __name__ == "__main__":
    print('We want to test some kermels for string here !')
    k=2
    Ak = [''.join(c) for c in itertools.product('ACGT', repeat=k)]
    print(Ak)
    X = np.array(['ACTAGAA','GACTTCCAA','GCTTAAGAA'])
    Y = np.array(['TTACGGAA','ATT','GCTTAG'])
    # X = ['ACTAGAA','GACTTCCAA']
    # Y = ['TTACGGAA','ATT']
    K = get_spectrum_kernel(X,X,k=2)
    print(K)
    
    print(get_spectrum_feature_map('AGTCGGAG',k,Ak))
    K2 = get_mismatch_kernel(X,k=2,m=1)
    print(K2)
    print(normalize_K(K2))
    k=2;m=1
    print(get_mismatch_feature_map('AGTCGGAGAAA',k,m,Ak))
    
    





