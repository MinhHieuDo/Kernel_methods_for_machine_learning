import pandas as pd
import numpy as np
import os
import argparse
from ..config import data_dir, results_dir, fit_dir
from ..lib.tools.utils import accuracy_score, train_test_split, check_create_dir
from ..lib.methods.SupportVectorMachine import KSVM
from ..lib.methods.LogisticRegression import  KLR
from ..lib.tools.data_processing import load_data,load_all_data, save_Gram_matrix, load_Gram_matrix
from ..lib.kernels.kernelsForString import get_spectrum_kernel, get_mismatch_kernel
from ..lib.kernels.kernelsForNum import rbf_kernel,linear_kernel,polynomial_kernel



# Configuration of the test case
best_param_kernel = [{'kernel_name':'spectrum', 'kernel_function':get_spectrum_kernel, 'k':8},
                    {'kernel_name': 'mismatch', 'kernel_function':get_spectrum_kernel, 'k':8},
                    {'kernel_name': 'spectrum', 'kernel_function':get_spectrum_kernel, 'k':8}]

best_param_model = [{'model_name': KSVM, 'C':2},
                    {'model_name': KSVM, 'C':10},
                    {'model_name': KSVM, 'C':1}]


# Load data
X_total = {'train':[],'val':[],'test':[]}
y_total = {'train':[], 'val':[]}

for id in range(3):
    Id_train, X_train_full, y_train_full = load_data(phase='train',k=id, numeric_data=False)
    X_train, y_train, X_val, y_val = train_test_split(X_train_full,y_train_full,test_size=0.2,shuffle=True)
    y_train = y_train * 2 - 1
    y_val = y_val * 2 - 1
    X_total['train'].append(X_train)
    y_total['train'].append(y_train)
    X_total['val'].append(X_val)
    y_total['val'].append(y_val)
    Id_test, X_test, _ = load_data(phase='test', k=id, numeric_data=False)
    X_total['test'].append(X_test)




# compute some Gram matrix :
Gram_matrix_total = {'train':[], 'val':[], 'test':[]}

for id in range(3):
    for phase in ['train','val','test']:
        if phase == 'train':
            Gram_matrix_total[phase].append(best_param_kernel[id]['kernel_function'](X_total[phase][id], None, **best_param_kernel[id]))
            
        else:
            Gram_matrix_total[phase].append(best_param_kernel[id]['kernel_function'](X_total[phase][id], X_total['train'][id], **best_param_kernel[id]))
            



model_total = []
for id in range(3):
    model_total.append( best_param_model[id]['model_name'](**best_param_model[id],**best_param_kernel[id]))

# train and predict for each set
prediction_test = []
prediction_val  = []
for id in range(3):
    print('For set ',id)
    model_total[id].fit_use_K(Gram_matrix_total['train'][id],y_total['train'][id])
    pred_val  = model_total[id].predict_use_K(Gram_matrix_total['val'][id])
    print('Precision =', accuracy_score(pred_val, y_total['val'][id]))
    pred_test = model_total[id].predict_use_K(Gram_matrix_total['test'][id])
    pred_test = np.array(pred_test >= 0, dtype=int)
    prediction_val.append(pred_val)
    prediction_test.append(pred_test)
prediction_test = np.concatenate(prediction_test)
print(prediction_test)


# submission 

check_create_dir(results_dir)
with open(results_dir+"Yte.csv", 'w') as f:
    f.write('ID,Bound\n')
    for i in range(len(prediction_test)):
        f.write(str(i)+','+str(prediction_test[i])+'\n')





