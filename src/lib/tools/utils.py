import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time
from contextlib import contextmanager
from typing import List
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles.named_colors import NAMED_COLORS
from prompt_toolkit.styles import Style

# Performance utils
# =================
@contextmanager
def timeit(msg, font_style='', fg='Black', bg='White'):
    pprint(msg, font_style=font_style, fg=fg, bg=bg)
    t0 = time.time()
    yield
    pprint('Finished {}. Run time: {}'.format(msg, time.time() - t0), font_style=font_style, fg=fg, bg=bg)

# Print utils
# ============
def pprint(text, font_style='', fg='Black', bg='White'):
    """ANSI color labels: from prompt_toolkit.styles.named_colors import NAMED_COLORS
    """
    if font_style not in ['', 'underline', 'italic', 'bold']:
        print('Warning: Invalid font_style '+font_style+'.')
    if fg not in NAMED_COLORS:
        print('Warning: fg color '+fg+' not in ANSI NAMED_COLORS. We use Black instead.')
        fg = 'Black'
    if bg not in NAMED_COLORS:
        print('Warning: bg color '+bg+' not in ANSI NAMED_COLORS. We use White instead.')
        bg = 'White'
    
    style = Style.from_dict({'a': '{} fg:{} bg:{}'.format(font_style, fg, bg)})
    return print_formatted_text(HTML('<a>{}</a>'.format(text)), style=style)

def sigmoid(x):
    """Return the sigmoid of x
    Parameters
    ----------
    x : float
        input value
    Returns
    -------
    float
	The sigmoid of the input
    """
    return 1/(1 + np.exp(-x))

def accuracy_score(y_pred,y_true):
    '''
    Function that computes the accuracy score for prediction
    Param: @y_true: true label
    @y_pred: predict label
    Return: accuracy score
    '''
    return (y_true == y_pred).mean()

def error (y_pred,y_true):
    '''
    Function that computes the error for prediction
    Param: @y_true: true label
    @y_pred: predict label
    Return: error
    '''
    return (y_true != y_pred).mean()



def train_test_split(X, y, test_size=0.2, shuffle=True):
    Id_total = list(range(X.shape[0]))

    if shuffle:
        #split = np.random.choice(range(X.shape[0]), int((1-test_size)*X.shape[0]))
        random.shuffle(Id_total)
    Id_train, Id_test = Id_total[:int((1-test_size)*X.shape[0])], Id_total[int((1-test_size)*X.shape[0]):]

    Xtrain =  X[Id_train]
    y_train = y[Id_train]
    X_test =  X[Id_test]
    y_test =  y[Id_test]

    return Xtrain, y_train, X_test, y_test



def get_colors(n):
    cmap = plt.get_cmap('jet')
    return cmap(np.linspace(0, 1.0, n))

def check_create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name