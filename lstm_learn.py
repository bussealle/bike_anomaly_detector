# encoding: utf-8

from keras.models import Sequential
from keras.layers import LSTM, Dense
#from keras.layers.core import Activation, Dropout
#from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import Tkinter
import datetime
import itertools


# Truncated Time
TRUNCATED_TIME = 30
#seed = 7
np.random.seed()



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
    
    timestep = TRUNCATED_TIME
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
def data_process(train_csv,test_csv):
    X_train, y_train = [], []
    X_test, y_test = [], []

    X_x_train, X_y_train, X_z_train = [], [], []
    X_x_test, X_y_test, X_z_test = [], [], []
    
    test_frag = False
    if len(test_csv)>0:
        test_frag = True

    for csv_name in train_csv:
        print "reading train... '{}'".format(csv_name)
        df = pd.read_csv('./labeled/'+csv_name+'.csv')
        #print df.head(3)

        X_x_train.extend(df['Acc_x'].values.flatten())
        X_y_train.extend(df['Acc_y'].values.flatten())
        X_z_train.extend(df['Acc_z'].values.flatten())
        
        y_train.extend(df['Label'].values.flatten())
    
    if test_frag==True:
        for csv_name in test_csv:
            print "reading test... '{}'".format(csv_name)
            df = pd.read_csv('./labeled/'+csv_name+'.csv')
            #print df.head(3)
            
            X_x_test.extend(df['Acc_x'].values.flatten())
            X_y_test.extend(df['Acc_y'].values.flatten())
            X_z_test.extend(df['Acc_z'].values.flatten())
            
            y_test.extend(df['Label'].values.flatten())
    
    X_train, y_train = slice_truncated([X_x_train, X_y_train, X_z_train], y_train)
    if test_frag==True:
        X_test, y_test = slice_truncated([X_x_test, X_y_test, X_z_test], y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
    print "Training data ---->"
    print "X.shape: ",
    print X_train.shape
    print "y.shape: ",
    print y_train.shape
    print "Testing data ---->"
    print "X.shape: ",
    print X_test.shape
    print "y.shape: ",
    print y_test.shape
    
    return X_train, y_train, X_test, y_test

# クラスをone hot 配列に
def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

# 設定
class Config(object):
    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)
        self.test_count = len(X_test)
        self.n_steps = len(X_train[0])
        # Training
        self.batch_size = 1500
        self.training_epochs = 90
        # LSTM structure
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 64
        self.n_classes = 4

# LSTM
def go_LSTM(X_train, y_train, X_test, y_test, config):

     
    model = Sequential()
    model.add(LSTM(config.n_hidden, return_sequences=True,
        input_shape=(config.n_steps, config.n_inputs)))
    model.add(LSTM(config.n_hidden, return_sequences=True))
    model.add(LSTM(config.n_hidden))
    model.add(Dense(config.n_classes, activation='softmax'))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    
    arg = sys.argv
    if len(arg)==2 and arg[1].find('.hdf5')!=-1:
        print "loading weight {} --->".format(arg[1])
        model.load_weights(arg[1])
        print "--DONE!--"

    #early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    history  = model.fit(X_train, y_train,
            batch_size=config.batch_size, epochs=config.training_epochs,
            validation_data=(X_test, y_test)
            #validation_split=0.1,
            #callbacks=[early_stopping]
            )

    return model, history

# 結果をプロット
def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

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



# パラメタを保存
def save_parameters(model):
    dir_name = "./weight"
    if not(os.path.exists(dir_name)):
        os.mkdir(dir_name)
    filepath = "weight_{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(dir_name,"{}.json".format(filepath)), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join(dir_name,"{}.yaml".format(filepath)), 'w').write(yaml_string)
    print('save weights')
    model.save_weights(os.path.join(dir_name,"{}.hdf5".format(filepath)))

if __name__ == "__main__":
    # csvデータを開く
    TRAIN_CSV = [
            "accel_nothing_robust1_labeled",
            "accel_nothing_robust2_labeled",
            #"accel_seq2_labeled"
            #"accel_nothing1_labeled",
            #"accel_nosthing2_labeled",
            #"accel_nothing3_labeled",
            #"accel_subtle-attack_labeled",
            #"accel_soft-attack_labeled",
            #"accel_lying-right1_labeled",
            #"accel_lying-right2_labeled",
            #"accel_lying-left1_labeled",
            #"accel_lying-left2_labeled",
            #"accel_liftup-move1_labeled",
            #"accel_liftup-move2_labeled",
            #"accel_lift-right_labeled",
            #"accel_lift-left_labeled",
            #"accel_ijirare_labeled",
            #"accel_dondon_labeled",
            #"accel_lying-left3_labeled",
            #"accel_hard-attack_labeled"
            ]
    TEST_CSV = [
            #"accel_seq1_labeled",
            #"accel_seq2_labeled"
            ]
    LABELS = [
            "nothing",
            "lying",
            "lifting",
            "subtle"
            ]
    X_train, y_train, X_test, y_test = data_process(TRAIN_CSV,TEST_CSV)
    y_train = one_hot(y_train)
    y_test_o = one_hot(y_test)
    config = Config(X_train, X_test)
    
    #LSTM
    model, history = go_LSTM(X_train, y_train, X_test, y_test_o, config)
    #Save model
    save_parameters(model)

    #Eval
    score = model.evaluate(X_test, y_test_o, verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])
    
    #Plot result
    plot_confmatrix(model,LABELS,X_test,y_test)
    plot_history(history)
