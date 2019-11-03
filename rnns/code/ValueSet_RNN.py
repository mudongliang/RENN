import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Activation, LSTM, concatenate, Input, GRU, SimpleRNN
from keras.layers.embeddings import Embedding
from scipy import io
from sklearn.model_selection import train_test_split
import re
from os.path import join

"""
Model structure:
input: Decimal
input layer: 256 (Embedding)
hidden layer: 8 (Bidirectional RNN)
output layer: 2
"""
"""
0: global
1: heap
2: stack
3: other
"""
np.random.seed(1234)

def process_data(data_file, label_file):
    with open(label_file) as f:
        label_txt = f.readlines()
    with open(data_file) as f:
        data_txt = f.readlines()
    data = []
    label = []
    for i in xrange(len(data_txt)):
        data_tmp = re.sub('\s', '', data_txt[i])
        label_tmp = label_txt[i]
        if len(label_tmp) > 2:
            label_tmp = 10*(int(label_tmp[0])+1)+int(label_tmp[2])+1
        else:
            label_tmp = 10*(int(label_tmp[0])+1)
        label_tmp = int(label_tmp)
        for j in xrange((len(data_tmp))/2):
            dec = int(data_tmp[j*2:(j+1)*2], 16)
            data.append(dec)
            label.append(label_tmp)
    return data, label

def perf_measure(yy_true, yy_pred, pos_label, neg_label, option ):
    y_true = yy_true.copy()
    y_pred = yy_pred.copy()

    TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
    for j in neg_label:
        y_true[np.where(y_true==j)] = 0
        y_pred[np.where(y_pred==j)] = 0

    y_true[np.where(y_true==pos_label)] = 1
    y_pred[np.where(y_pred==pos_label)] = 1

    if option == 1:
        for id in xrange(y_true.shape[0]):
            pos_id = np.where(y_true[id, :] == 1)[0]
            i = 0
            while (i != pos_id.shape[0]):
                idx = []
                for j in xrange(pos_id.shape[0]):
                    if j >= i:
                        if j != pos_id.shape[0] - 1:
                            if (pos_id[j] + 1 == pos_id[j + 1]):
                                idx.append(j)
                                i = i + 1
                            elif (pos_id[j] - 1 == pos_id[j - 1]) or j == pos_id.shape[0] - 1:
                                i = i + 1
                                idx.append(j)
                                for j in idx:
                                    if y_pred[id, pos_id[j]] == 1:
                                        y_pred[id, pos_id[idx]] = 1
                                        break
                                break
                        else:
                            i = i + 1
                            idx.append(j)
                            for j in idx:
                                if y_pred[id, pos_id[j]] == 1:
                                    y_pred[id, pos_id[idx]] = 1
                                    break
                            break

    for id in xrange(y_true.shape[0]):
        pos_id = np.where(y_true[id, :] == 1)[0]
        aa = np.where(y_pred[id, pos_id] == 1)[0]
        TP += aa.shape[0]

        aa = np.where(y_pred[id, pos_id] == 0)[0]
        FN += aa.shape[0]

        neg_id = np.where(y_true[id, :] == 0)[0]
        aa = np.where(y_pred[id, neg_id] == 0)[0]
        TN = aa.shape[0]

        aa = np.where(y_pred[id, neg_id] == 1)[0]
        FP += aa.shape[0]

    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    f1 = (2 * precision * recall) / (precision + recall + 1e-5)
    return (precision, recall, accuracy, f1)

def predict_classes(proba):
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')

def load_data(data_set, seq_len, label_train):
    data = []
    label = []
    for folder in data_set:
        #data_file = '/Users/Henryguo/Desktop/new_result'
        data_file = os.path.join('/home/wzg13/Data/new_fse_traces', folder)
        inst_file = os.path.join(data_file, 'binary')
        label_file = os.path.join(data_file, 'region')
        data_temp, label_temp = process_data(inst_file, label_file)
        data.extend(data_temp)
        label.extend(label_temp)
    
    data = np.asarray(data)
    label = np.asarray(label)

    if label_train == 0:
        data_1 = data[np.where(label == 10)]
        label_1 = label[np.where(label == 10)]

        data_2 = data[np.where(label == 14)]
        label_2 = label[np.where(label == 14)]

        data_3 = data[np.where(label == 13)]
        label_3 = label[np.where(label == 13)]

        data_12 = np.concatenate((data_1, data_2))
        label_12 = np.concatenate((label_1, label_2))

        data = np.concatenate((data_12, data_3))
        label = np.concatenate((label_12, label_3))

        label[np.where(label == 10)] = 0
        label[np.where(label == 13)] = 1
        label[np.where(label == 14)] = 2

        data = data.tolist()
        label = label.tolist()
        n_class = 3

    elif label_train == 1:

        data_1 = data[np.where(label == 20)]
        label_1 = label[np.where(label == 20)]
        
        data_2 = data[np.where(label == 22)]
        label_2 = label[np.where(label == 22)]
        
        data_3 = data[np.where(label == 23)]
        label_3 = label[np.where(label == 23)]
        
        data_4 = data[np.where(label == 24)]
        label_4 = label[np.where(label == 24)]

        data_12 = np.concatenate((data_1, data_2))
        data_123 = np.concatenate((data_12, data_3))

        label_12 = np.concatenate((label_1, label_2))
        label_123 = np.concatenate((label_12, label_3))

        data = np.concatenate((data_123, data_4))
        label = np.concatenate((label_123, label_4))

        label[np.where(label == 20)] = 0
        label[np.where(label == 22)] = 1
        label[np.where(label == 23)] = 2
        label[np.where(label == 24)] = 3

        data = data.tolist()
        label = label.tolist()
        n_class = 4

    elif label_train == 2:

        data_1 = data[np.where(label == 30)]
        label_1 = label[np.where(label == 30)]
        
        data_2 = data[np.where(label == 32)]
        label_2 = label[np.where(label == 32)]
        
        data_3 = data[np.where(label == 33)]
        label_3 = label[np.where(label == 33)]
        
        data_4 = data[np.where(label == 34)]
        label_4 = label[np.where(label == 34)]

        data_12 = np.concatenate((data_1, data_2))
        data_123 = np.concatenate((data_12, data_3))

        label_12 = np.concatenate((label_1, label_2))
        label_123 = np.concatenate((label_12, label_3))

        data = np.concatenate((data_123, data_4))
        label = np.concatenate((label_123, label_4))

        label[np.where(label == 30)] = 0
        label[np.where(label == 32)] = 1
        label[np.where(label == 33)] = 2
        label[np.where(label == 34)] = 3

        data = data.tolist()
        label = label.tolist()
        n_class = 4

    elif label_train == 3:

        data_1 = data[np.where(label == 41)]
        label_1 = label[np.where(label == 41)]
        
        data_2 = data[np.where(label == 42)]
        label_2 = label[np.where(label == 42)]
        
        data_3 = data[np.where(label == 43)]
        label_3 = label[np.where(label == 43)]
        
        data_4 = data[np.where(label == 44)]
        label_4 = label[np.where(label == 44)]

        data_12 = np.concatenate((data_1, data_2))
        data_123 = np.concatenate((data_12, data_3))

        label_12 = np.concatenate((label_1, label_2))
        label_123 = np.concatenate((label_12, label_3))

        data = np.concatenate((data_123, data_4))
        label = np.concatenate((label_123, label_4))

        label[np.where(label == 41)] = 0
        label[np.where(label == 42)] = 1
        label[np.where(label == 43)] = 2
        label[np.where(label == 44)] = 3

        data = data.tolist()
        label = label.tolist()
        n_class = 4


    data_num = len(data)
    data_truncated = []
    label_truncated = []
    for i in xrange((data_num/seq_len)+1):
        data_truncated.append(data[i*seq_len:(i+1)*seq_len])
        label_truncated.append(label[i * seq_len:(i + 1) * seq_len])

    X = pad_sequences(data_truncated, maxlen=seq_len, dtype='int32',
                        padding='post', truncating='post', value=0)
    Y = pad_sequences(label_truncated, maxlen=seq_len, dtype='int32',
                        padding='post', truncating='post', value=0)
    
    Y_one_hot = np.zeros((Y.shape[0], seq_len, n_class), dtype=Y.dtype)

    for id in xrange(Y.shape[0]):
        Y_one_hot[id, np.arange(seq_len), Y[id]] = 1
    
    return X, Y_one_hot, Y, n_class

def ValueSet_RNN(training_set, testing_set, train_flag, batch_size, epochs, label_train, seq_len):

    X_train, Y_train, YY_train, n_class = load_data(training_set, seq_len, label_train)
    X_test, Y_test, YY_test, n_class = load_data(testing_set, seq_len, label_train)
     
    YY_train_input_f = np.concatenate((np.zeros((YY_train.shape[0],1)),YY_train[:,0:-1]), axis=1).reshape(YY_train.shape[0], seq_len, 1)
  
    YY_train_input_b = np.concatenate((YY_train[:,1:], np.zeros((YY_train.shape[0],1))), axis=1).reshape(YY_train.shape[0], seq_len, 1)
   
    YY_test_input_f = np.concatenate((np.zeros((YY_test.shape[0],1)),YY_test[:,0:-1]), axis=1).reshape(YY_test.shape[0], seq_len, 1)
    YY_test_input_b = np.concatenate((YY_test[:,1:], np.zeros((YY_test.shape[0],1))), axis=1).reshape(YY_test.shape[0], seq_len, 1)

    print "Counting the number of data in each category..."
    if label_train == 0:
        print "training..."
        print '0: global...'
        print np.where(YY_train==0)[0].shape[0]
        print '1: global&Stack...'
        print np.where(YY_train==1)[0].shape[0]
        print '2: global&Other...'
        print np.where(YY_train==2)[0].shape[0]

        print "testing....."
        print '0: global...'
        print np.where(YY_test==0)[0].shape[0]
        print '1: global&Stack...'
        print np.where(YY_test==1)[0].shape[0]
        print '2: global&Other...'
        print np.where(YY_test==2)[0].shape[0]

    if label_train == 1:
        print "training..."
        print '0: heap...'
        print np.where(YY_train==0)[0].shape[0]
        print '1: heap&heap...'
        print np.where(YY_train==1)[0].shape[0]
        print '2: heap&Stack...'
        print np.where(YY_train==2)[0].shape[0]
        print '3: heap&Other...'
        print np.where(YY_train==3)[0].shape[0]

        print "testing....."
        print '0: heap...'
        print np.where(YY_test==0)[0].shape[0]
        print '1: heap&heap...'
        print np.where(YY_test==1)[0].shape[0]
        print '2: heap&Stack...'
        print np.where(YY_test==2)[0].shape[0]
        print '3: heap&Other...'
        print np.where(YY_test==3)[0].shape[0]

    if label_train == 2:
        print "training..."
        print '0: Stack...'
        print np.where(YY_train==0)[0].shape[0]
        print '1: Stack&heap...'
        print np.where(YY_train==1)[0].shape[0]
        print '2: Stack&Stack...'
        print np.where(YY_train==2)[0].shape[0]
        print '3: Stack&Other...'
        print np.where(YY_train==3)[0].shape[0]

        print "testing....."
        print '0: Stack...'
        print np.where(YY_test==0)[0].shape[0]
        print '1: Stack&heap...'
        print np.where(YY_test==1)[0].shape[0]
        print '2: Stack&Stack...'
        print np.where(YY_test==2)[0].shape[0]
        print '3: Stack&Other...'
        print np.where(YY_test==3)[0].shape[0]
        
    if label_train == 3:
        print "training..."
        print '0: other&global...'
        print np.where(YY_train==0)[0].shape[0]
        print '1: other&heap...'
        print np.where(YY_train==1)[0].shape[0]
        print '2: other&Stack...'
        print np.where(YY_train==2)[0].shape[0]
        print '3: other&other...'
        print np.where(YY_train==3)[0].shape[0]


        print "testing....."
        print '0: other&global...'
        print np.where(YY_test==0)[0].shape[0]
        print '1: other&heap...'
        print np.where(YY_test==1)[0].shape[0]
        print '2: other&Stack...'
        print np.where(YY_test==2)[0].shape[0]
        print '3: other&other...'
        print np.where(YY_test==3)[0].shape[0]


    if train_flag == 0:

        print "Using Bi-SimpleRNN >>>>>>>>>>>>>>>>>>>>>>>>"
        model = Sequential()
        model.add(Embedding(input_dim= 256, output_dim=16, input_length=seq_len))
        model.add(Bidirectional(SimpleRNN(units=8, activation='relu', dropout=0.5, return_sequences=True)))
        model.add(TimeDistributed(Dense(n_class, activation='softmax'), input_shape=(seq_len, 16)))
        model.summary()
        model.compile('rmsprop', 'categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=[X_test, Y_test], verbose = 1)

        name = str(label_train)+'__BidirtionalSimpleRnn.h5' 
        model.save(name)

    elif train_flag == 1:
        print "Using Bi-GRU >>>>>>>>>>>>>>>>>>>>>>>>"
        model = Sequential()
        model.add(Embedding(input_dim= 256, output_dim=16, input_length=seq_len))
        model.add(Bidirectional(GRU(units=8, recurrent_dropout=0.5, dropout=0.5, return_sequences=True)))
        model.add(TimeDistributed(Dense(n_class, activation='softmax'), input_shape=(seq_len, 16)))
        model.summary()
        model.compile('rmsprop', 'categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=[X_test, Y_test], verbose = 1)

        name = str(label_train)+'__BidirtionalGRU.h5'
        model.save(name)

    elif train_flag == 2:
        print "Using Bi-LSTM >>>>>>>>>>>>>>>>>>>>>>>>"
        model = Sequential()
        model.add(Embedding(input_dim= 256, output_dim=16, input_length=seq_len))
        model.add(Bidirectional(LSTM(units=8, dropout=0.25, recurrent_dropout = 0.25, return_sequences=True)))
        model.add(TimeDistributed(Dense(n_class, activation='softmax'), input_shape=(seq_len, 16)))
        model.summary()
        model.compile('rmsprop', 'categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=[X_test, Y_test], verbose = 1)

        name = str(label_train) + '__BidirtionalLSTM.h5'
        model.save(name)
    
    elif train_flag == 3:
        print "Using conditional-Bi-GRU or LSTM>>>>>>>>>>>>>>>>>>>>>>>>"

        X_input = Input(shape=(seq_len,), dtype='int32', name='X_input')
        X_embeded = Embedding(output_dim=16, input_dim=256, input_length=seq_len)(X_input)
        Y_input_f = Input(shape=(seq_len, 1,), name='Y_input_f')
        Embeded_out_f = concatenate([X_embeded, Y_input_f])

        Y_input_b = Input(shape=(seq_len, 1,), name='Y_input_b')
        Embeded_out_b = concatenate([X_embeded, Y_input_b])

        # GRU_out_f = (GRU(units=8, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))(Embeded_out_f)
        # GRU_out_b = (GRU(units=8, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))(Embeded_out_b)
        LSTM_out_f = (LSTM(units=8, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))(Embeded_out_f)
        LSTM_out_b = (LSTM(units=8, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))(Embeded_out_b)

        recurrent_out = concatenate([LSTM_out_f, LSTM_out_b])
        model_out = TimeDistributed(Dense(n_class, activation='softmax'), input_shape=(seq_len, 16), name='model_out')(
            recurrent_out)

        model = Model(inputs=[X_input, Y_input_f, Y_input_b], outputs=model_out)
        model.summary()
        model.compile('rmsprop', 'categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit({'X_input': X_train, 'Y_input_f': YY_train_input_f, 'Y_input_b': YY_train_input_b},
                  {'model_out': Y_train}, batch_size=batch_size,
                  epochs=epochs,
                  validation_data=[{'X_input': X_test, 'Y_input_f': YY_test_input_f, 'Y_input_b': YY_test_input_b},
                                   {'model_out': Y_test}], verbose=1)
        name = str(label_train) + '__conditional_BidirtionalLSTM.h5'
        model.save(name)

    if train_flag == 3:
        P_train = predict_classes(model.predict({'X_input':X_train, 'Y_input_f':YY_train_input_f,'Y_input_b':YY_train_input_b}))
        P_test = predict_classes(model.predict({'X_input':X_test, 'Y_input_f':YY_test_input_f, 'Y_input_b':YY_test_input_b}))
    
    else:
        P_train = model.predict_classes(X_train)
        P_test = model.predict_classes(X_test)
        
    print('****************************************')
    
    if label_train == 0:
        print('evaluating global....')
        print('evaluating train data....')
        print ('Global...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=0, neg_label= [1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Global&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=1, neg_label= [0, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Global&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=2, neg_label= [0, 1], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'

        print('****************************************')
        print('evaluating test data.....')
  
        print ('Global...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label = [1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Global&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Global&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label = [0, 1], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))


    elif label_train == 1:
        print('evaluating heap....')
        print('evaluating train data....')
        print ('Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'


        print('****************************************')
        print('evaluating test data.....')
        print ('Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'


    elif label_train == 2:

        print('evaluating Stack....')
        print('evaluating train data....')

        print ('Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'

        print('****************************************')
        print('evaluating test data.....')
  

        print ('Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'

    elif label_train == 3:
        print('evaluating other....')
        print('evaluating train data....')

        print ('Other&global...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Other&Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Other&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Other&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_train, yy_pred = P_train, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'

        print('****************************************')
        print('evaluating test data.....')
  
        print ('Other&global...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Other&Heap...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Other&Stack...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'
        print ('Other&Other...')
        (precision, recall, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, f1))
        print '-----------------------------'

    return (0)

