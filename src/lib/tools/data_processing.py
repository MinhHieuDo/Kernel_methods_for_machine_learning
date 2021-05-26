from ...config import data_dir,ext, fit_dir
import pandas as pd
import os
import numpy as np
from .utils import train_test_split, check_create_dir
import pickle

def load_data(phase, k=0, numeric_data=False):
    
    if phase == "train":
        st = "tr"
    elif phase == "test":
        st = "te"
    else:
        print('Please check your phase !')

    datafilename = "X" + st + str(k) + ext
    dataPath = os.path.join(data_dir, datafilename)
    df_data = pd.read_csv(dataPath, sep=',')

    Id = df_data["Id"].values
    if numeric_data:
        datafilename = "X" + st + str(k) + "_mat100" + ext
        dataPath = os.path.join(data_dir, datafilename)
        df_mat = pd.read_csv(dataPath, sep=' ', dtype='float64', header=None)
        X = df_mat.values
    else:
        X= df_data["seq"].values

    if phase == "train":
        labelfilename = "Y" + st + str(k) + ext
        labelPath = os.path.join(data_dir, labelfilename)
        df_labels = pd.read_csv(labelPath)
        y = df_labels["Bound"].values

        return (Id, X, y)
    else:
        return (Id, X, None)

def load_all_data(phase='train', numeric_data=False):
    if phase == 'train':
        Id_0, X_0, y_0 = load_data(phase, k=0, numeric_data=numeric_data)
        Id_1, X_1, y_1 = load_data(phase, k=1, numeric_data=numeric_data)
        Id_2, X_2, y_2 = load_data(phase, k=2, numeric_data=numeric_data)
        Id = np.concatenate([Id_0,Id_1,Id_2])
        X  = np.concatenate([X_0,X_1,X_2])
        y  = np.concatenate([y_0,y_1,y_2])
        return (Id,X,y)

    else:
        Id_0, X_0,_ = load_data(phase, k=0, numeric_data=numeric_data)
        Id_1, X_1,_ = load_data(phase, k=1, numeric_data=numeric_data)
        Id_2, X_2,_ = load_data(phase, k=2, numeric_data=numeric_data)
        Id = np.concatenate([Id_0,Id_1,Id_2])
        X  = np.concatenate([X_0,X_1,X_2])
        return (Id,X,None)

def save_Gram_matrix(K,phase,Id,kernel_name,**kwargs):
    check_create_dir(fit_dir)
    if kernel_name == 'spectrum':
        param = '_k'+str(kwargs.get('k',6))
    elif kernel_name == 'mismatch':
        param = '_k'+str(kwargs.get('k',6))+'_'+'m'+str(kwargs.get('m',1))
    else:
        param =''
    filename = 'K'+'_'+phase+'_'+'set'+str(Id)+'_'+kernel_name+ param
    pickle.dump(K, open(fit_dir + filename, "wb"))

def load_Gram_matrix(phase,Id,kernel_name,**kwargs):
    if kernel_name == 'spectrum':
        param = '_k'+str(kwargs.get('k',6))
    elif kernel_name == 'mismatch':
        param = '_k'+str(kwargs.get('k',6))+'_'+'m'+str(kwargs.get('m',1))
    else:
        param =''
    filename = 'K'+'_'+phase+'_'+'set'+str(Id)+'_'+kernel_name+ param
    K = pickle.load(open(fit_dir + filename, "rb"))
    return K

if __name__ == "__main__":
    print('Test some functions')
    Id, data, labels = load_data(phase='train', k=2, numeric_data=False)
    print('Id',type(Id))
    print('data',data.shape[0])
    print('labels',type(labels))
    print(data[2])
    print(len(data[0]))


    Id, X, y = load_all_data(phase='train', numeric_data=True)
    print('Id',Id.shape)
    print('X',X.shape)
    print('y',y.shape)
    print(Id)
    X_train, y_train, X_test, y_test = train_test_split(X,y)
    print(X_train.shape)
    print(X_test.shape)
    
    
    