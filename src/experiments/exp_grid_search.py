import pandas as pd
import numpy as np
import os
from ..config import data_dir, results_dir, fit_dir
from ..lib.tools.utils import accuracy_score, train_test_split
from ..lib.methods.SupportVectorMachine import KSVM
from ..lib.methods.LogisticRegression import  KLR
from ..lib.tools.data_processing import load_data,load_all_data, save_Gram_matrix,load_Gram_matrix
from ..lib.kernels.kernelsForString import get_spectrum_kernel, get_mismatch_kernel



for id in range(3):
    Id_train, X_train_full, y_train_full = load_data(phase='train',k=id, numeric_data=False)
    X_train, y_train, X_val, y_val = train_test_split(X_train_full,y_train_full,test_size=0.2,shuffle=True)
    Id_test, X_test, _ = load_data(phase='test', k=id, numeric_data=False)
    for k in [5, 6, 7]:
        kernel_name = 'spectrum'
        Gram_matrix_train = get_spectrum_kernel(X_train,None, k=k)
        Gram_matrix_val = get_spectrum_kernel(X_val,X_train, k=k)
        Gram_matrix_test = get_spectrum_kernel(X_test,X_train, k=k)
        
        save_Gram_matrix(Gram_matrix_train,phase='train',Id=id,kernel_name=kernel_name,k=k)
        save_Gram_matrix(Gram_matrix_val,phase='val',Id=id,kernel_name=kernel_name,k=k)
        save_Gram_matrix(Gram_matrix_test,phase='test',Id=id,kernel_name=kernel_name,k=k)

        K_train = load_Gram_matrix(phase='train',Id=id,kernel_name=kernel_name,k=k)
        K_val   = load_Gram_matrix(phase='val',Id=id,kernel_name=kernel_name,k=k)
        K_test  = load_Gram_matrix(phase='test',Id=id,kernel_name=kernel_name,k=k)

        y_train = y_train*2 -1
        y_val   = y_val*2-1
        # SVC
        for Cv in [0.1,0.5,1,10,100]:
            model = KSVM(C=Cv, kernel_name=kernel_name,k=k)
            # model = KLR(lambda_reg=0.001, kernel_name='rbf',gamma=0.5)
            model.fit_use_K(K_train,y_train)
            pred_val = model.predict_use_K(K_val)
            print('SVM for Set = {}, k = {}, C = {}, Precision = {}'.format(id,k, Cv, accuracy_score(pred_val, y_val)))
            pred_test = model.predict_use_K(K_test)
            pred_test = np.array(pred_test >= 0, dtype=int)
            print(pred_test[:10])
        for lambda_reg in [0.01,0.1, 0.5, 1, 2]:
            model = KLR(lambda_reg=lambda_reg, kernel_name=kernel_name,k=k)
            # model = KLR(lambda_reg=0.001, kernel_name='rbf',gamma=0.5)
            model.fit_use_K(K_train,y_train)
            pred_val = model.predict_use_K(K_val)
            print('KLR for  Set = {}, k = {}, lambda = {}, Precision = {}'.format(id, k,lambda_reg, accuracy_score(pred_val, y_val)))
            pred_test = model.predict_use_K(K_test)
            pred_test = np.array(pred_test >= 0, dtype=int)
            print(pred_test[:10])


