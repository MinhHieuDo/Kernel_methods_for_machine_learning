'''
This experiment aim to train each data set
'''

import pandas as pd
import numpy as np
import os
from ..config import data_dir, results_dir, fit_dir
from ..lib.tools.utils import accuracy_score, train_test_split, check_create_dir
from ..lib.methods.SupportVectorMachine import KSVM
from ..lib.methods.LogisticRegression import  KLR
from ..lib.tools.data_processing import load_data,load_all_data, save_Gram_matrix,load_Gram_matrix
from ..lib.kernels.kernelsForString import get_spectrum_kernel, get_mismatch_kernel,normalize_K



id = 0
Id_train, X_train_full, y_train_full = load_data(phase='train',k=id, numeric_data=False)
X_train, y_train, X_val, y_val = train_test_split(X_train_full,y_train_full,test_size=0.2,shuffle=True)
Id_test, X_test, _ = load_data(phase='test', k=id, numeric_data=False)

# n_train = X_train.shape[0]
# n_val = X_val.shape[0]
# n_test = X_test.shape[0]


K_train = 0
K_val   = 0
K_test  = 0
kernel_name = 'spectrum'
for k in [7]:
    K_train_temp = get_spectrum_kernel(X_train,None, k=k)
    K_val_temp   = get_spectrum_kernel(X_val,X_train, k=k)
    K_test_temp  = get_spectrum_kernel(X_test,X_train, k=k)

    # save_Gram_matrix(K_train_temp,phase='train',Id=id,kernel_name=kernel_name,k=k)
    # save_Gram_matrix(K_val_temp,phase='val',Id=id,kernel_name=kernel_name,k=k)
    # save_Gram_matrix(K_test_temp,phase='test',Id=id,kernel_name=kernel_name,k=k)

    K_train = K_train + K_train_temp
    K_val   = K_val  + K_val_temp
    K_test  = K_test + K_test_temp

# K_train = load_Gram_matrix(phase='train',Id=id,kernel_name=kernel_name,k=k)
# K_val   = load_Gram_matrix(phase='val',Id=id,kernel_name=kernel_name,k=k)
# K_test  = load_Gram_matrix(phase='test',Id=id,kernel_name=kernel_name,k=k)

K_train = normalize_K(K_train)


y_train = y_train*2 -1
y_val   = y_val*2-1

model = KSVM(C=2, kernel_name=kernel_name)
#model = KLR(lambda_reg=0.01, kernel_name=kernel_name,k=k)
model.fit_use_K(K_train,y_train)

pred_train = model.predict_use_K(K_train)
print('Accuracy train', accuracy_score(pred_train,y_train))
pred_val = model.predict_use_K(K_val)
print('Precision =', accuracy_score(pred_val, y_val))
pred_test = model.predict_use_K(K_test)
pred_test = np.array(pred_test >= 0, dtype=int)
print(pred_test[:10])
k=8
check_create_dir(results_dir)
filename = 'submission'+'_'+ kernel_name +'_'+'k'+str(k)+'_'+'id'+str(id)+'.csv'
with open(results_dir+filename, 'w') as f:
    f.write('ID,Bound\n')
    for i in range(len(pred_test)):
        f.write(str(1000*id+i)+','+str(pred_test[i])+'\n')

