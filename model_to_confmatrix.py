# encoding: utf-8

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.models import load_model,model_from_json
from keras.utils import np_utils


argv = sys.argv
timestep = None

# 配列を変形[[x,x..],[y,y..],[z,z..]] -> [[x,y,z],[x,y,z],...]
def combine_xyz(arr):
    res = []
    for i in range(len(arr[0])):
        temp = []
        for j in range(len(arr)):
            temp.append(arr[j][i])
        res.append(temp)
    return res


# データをスライスしてミニバッチの配列に
def slice_truncated(data, label):
    
    X = []
    y = []
    
    data = combine_xyz(data)

    for i in range(len(data)-timestep):
        X.append(data[i:i + timestep])
        y.append(label[i + timestep])
    
    re_X = np.array(X).reshape(len(X), timestep, len(X[0][0]))
    re_y = np.array(y).reshape(len(X), 1)

    return re_X, re_y

# csvからのデータの読み込み
def data_process():
    X_test, y_test = [], []
    X_x_test, X_y_test, X_z_test = [], [], []
    
    test_csv = argv[2:]
    
    for csv_name in test_csv:
        print "reading test... '{}'".format(csv_name)
        df = pd.read_csv(csv_name)
        #print df.head(3)
            
        X_x_test.extend(df['Acc_x'].values.flatten())
        X_y_test.extend(df['Acc_y'].values.flatten())
        X_z_test.extend(df['Acc_z'].values.flatten())
            
        y_test.extend(df['Label'].values.flatten())
    
    X_test, y_test = slice_truncated([X_x_test, X_y_test, X_z_test], y_test)
    print "Testing data ---->"
    print "X.shape: ",
    print X_test.shape
    print "y.shape: ",
    print y_test.shape
    
    return X_test, y_test

# クラスをone hot 配列に
def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def plot_confmatrix(model, labels, X_test, y_test):
    #confusion matrix生成
    y_p = model.predict(X_test)
    
    y_pred = []
    
    for arr in y_p:
        y_pred.append(np.argmax(arr))

    cm = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)
    plt.figure()

    title='Confusion matrix'
    cmap=plt.cm.Blues
    tick_marks = np.arange(len(labels))

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    

    thresh = cm.max() / 2.
    fmt = '.2f'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__=="__main__":
    if len(argv)<2:
        print 'need at least 1 arg'
        sys.exit()

    filepath = argv[1]
    
    model = None
    if argv[1].find('.h5')!=-1:
        model = load_model(filepath)
    elif argv[1].find('.json')!=-1:
        model = model_from_json(open(filepath).read())
        model.load_weights(filepath.replace('.json','.hdf5'))
    else:
        print 'need .h5 or .json'
        sys.exit()


    model.summary()
    if len(argv)==2:
        sys.exit()
    
    layer =  model.get_layer(index=0)
    timestep = int(layer.input.get_shape()[1])
    X_test, y_test = data_process()
    
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    score = model.evaluate(X_test, one_hot(y_test), verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1]) 
    labels = ["nothing","lying","lifting","subtle"]
    plot_confmatrix(model, labels, X_test, y_test)
