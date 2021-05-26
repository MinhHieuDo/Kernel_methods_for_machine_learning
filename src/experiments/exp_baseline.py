import pandas as pd
import numpy as np
import os
from ..config import data_dir, results_dir
from ..lib.tools.utils import accuracy_score, train_test_split, check_create_dir
from ..lib.methods.SupportVectorMachine import KSVM
from ..lib.methods.LogisticRegression import  KLR
from ..lib.tools.data_processing import load_data,load_all_data
from ..lib.kernels.kernelsForString import get_spectrum_kernel, get_mismatch_kernel


'''
Read all the datasets
'''


Id, X_train_full, y_train_full = load_all_data(phase='train', numeric_data=False)
Id, X_test,_ = load_all_data(phase='test', numeric_data=False)

X_train, y_train, X_val, y_val = train_test_split(X_train_full,y_train_full,test_size=0.2)
y_train = y_train*2 -1
y_val   = y_val*2-1

# X_train_kernel = get_spectrum_kernel(X_train,k=6)
# X_val_kernel   = get_spectrum_kernel(X_val,X_train,k=6)
# X_test_kernel  = get_spectrum_kernel(X_test,X_train,k=6)

X_train_kernel = get_mismatch_kernel(X_train,k=5,m=1)
X_val_kernel   = get_mismatch_kernel(X_val,X_train,k=5,m=1)
X_test_kernel  = get_mismatch_kernel(X_test,X_train,k=5,m=1)


# model = KSVM(C=1000, kernel_name='rbf')
model  = KLR(lambda_reg=0.001, kernel_name='rbf')



model.fit_use_K(X_train_kernel,y_train)
pred_val = model.predict_use_K(X_val_kernel)

print('Precision =',accuracy_score(y_val,pred_val))
print('y_val',y_val[:20])
print('pred_val',pred_val[:20])

pred_test = model.predict_use_K(X_test_kernel).reshape(-1)
pred_test = np.array(pred_test>=0,dtype=int)
print('pred_test',pred_test)

check_create_dir(results_dir)

with open(results_dir+"submission_baseline.csv", 'w') as f:
    f.write('ID,Bound\n')
    for i in range(len(pred_test)):
        f.write(str(i)+','+str(pred_test[i])+'\n')