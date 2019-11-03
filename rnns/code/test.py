import os
import argparse
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

   
    pos_id = np.where(y_true == 1)[0]
    aa = np.where(y_pred[pos_id] == 1)[0]
    TP = aa.shape[0]

    aa = np.where(y_pred[pos_id] == 0)[0]
    FN = aa.shape[0]


    neg_id = np.where(y_true == 0)[0]
    aa = np.where(y_pred[neg_id] == 0)[0]
    TN = aa.shape[0]


    aa = np.where(y_pred[neg_id] == 1)[0]
    FP = aa.shape[0]


    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-5)
    f1 = (2 * precision * recall) / (precision + recall + 1e-5)
    return (precision, recall, accuracy, f1)

def predict_classes(proba):
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')

def load_data(data_file, seq_len, label_train):
    data_all = []
    label_all = []
    inst_file = os.path.join(data_file, 'binary')
    label_file = os.path.join(data_file, 'region')
    data_temp, label_temp = process_data(inst_file, label_file)
    data_all.extend(data_temp)
    label_all.extend(label_temp)
    
    data = np.asarray(data_all)
    label = np.asarray(label_all)
    label_all_truth = label

    if label_train == 0:
        data_1 = data[np.where(label == 10)]
        label_1 = label[np.where(label == 10)]

        data_2 = data[np.where(label == 13)]
        label_2 = label[np.where(label == 13)]

        data_3 = data[np.where(label == 14)]
        label_3 = label[np.where(label == 14)]

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
    
    return X, Y_one_hot, Y, n_class, label_all_truth


def Testing(data_file, train_flag, label_train, seq_len):

    X_test, Y_test, YY_test, n_class, label_all_truth = load_data(data_file, seq_len, label_train)

    YY_test_input_f = np.concatenate((np.zeros((YY_test.shape[0], 1)), YY_test[:, 0:-1]), axis=1).reshape(
        YY_test.shape[0], seq_len, 1)
    YY_test_input_b = np.concatenate((YY_test[:, 1:], np.zeros((YY_test.shape[0], 1))), axis=1).reshape(
        YY_test.shape[0], seq_len, 1)

    # Counting the number of data in each category
    if label_train == 0:
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
        print '*************************** Global ***************************'
        print '0: global...'
        print np.where(YY_test==0)[0].shape[0]
        print '1: global&Stack...'
        print np.where(YY_test==1)[0].shape[0]
        print '2: global&Other...'
        print np.where(YY_test==2)[0].shape[0]

    if label_train == 1:
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
        print '*************************** Heap ***************************'
        print '0: heap...'
        print np.where(YY_test==0)[0].shape[0]
        print '1: heap&heap...'
        print np.where(YY_test==1)[0].shape[0]
        print '2: heap&Stack...'
        print np.where(YY_test==2)[0].shape[0]
        print '3: heap&Other...'
        print np.where(YY_test==3)[0].shape[0]

    if label_train == 2:
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
        print '*************************** Stack ***************************'
        print '0: Stack...'
        print np.where(YY_test==0)[0].shape[0]
        print '1: Stack&heap...'
        print np.where(YY_test==1)[0].shape[0]
        print '2: Stack&Stack...'
        print np.where(YY_test==2)[0].shape[0]
        print '3: Stack&Other...'
        print np.where(YY_test==3)[0].shape[0]
        
    if label_train == 3:
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
        print '*************************** Other ***************************'
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
        name = str(label_train) + '__BidirtionalSimpleRnn.h5'
        name = 'models/'+name
        model = load_model(name)

    elif train_flag == 1:
        print "Using Bi-GRU >>>>>>>>>>>>>>>>>>>>>>>>"
        name = str(label_train) + '__BidirtionalGRU.h5'
        name = 'models/'+name
        model = load_model(name)

    elif train_flag == 2:
        print "Using Bi-LSTM >>>>>>>>>>>>>>>>>>>>>>>>"

        name = str(label_train) + '__BidirtionalLSTM.h5'
        name = 'models/'+name
        model = load_model(name)

    elif train_flag == 3:
        print "Using Unconditional-Bi-LSTM >>>>>>>>>>>>>>>>>>>>>>>>"
        name = str(label_train) + '__conditional_BidirtionalLSTM.h5'
        # name = str(label_train) + '__conditional_BidirtionalGRU.h5'
        name = 'models_new/'+name
        model = load_model(name)

    if train_flag == 3:
        P_test = predict_classes(
            model.predict({'X_input': X_test, 'Y_input_f': YY_test_input_f, 'Y_input_b': YY_test_input_b}))

    else:
         P_test = model.predict_classes(X_test)


    print('****************************************************************')
    if label_train == 0:
        print('evaluating global....')
        print('****************************************')
        P_test = P_test.flatten()[0:(np.where(label_all_truth == 10)[0].shape[0]
                                     + np.where(label_all_truth == 13)[0].shape[0]
                                     + np.where(label_all_truth == 14)[0].shape[0])]

        YY_test = YY_test.flatten()[0:P_test.shape[0]]
 
        print ('Global...')
        (precision_00, recall_00, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label = [1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_00, recall_00, accuracy, f1))
        print '-----------------------------'
        print ('Global&Stack...')
        (precision_01, recall_01, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_01, recall_01, accuracy, f1))
        print '-----------------------------'
        print ('Global&Other...')
        (precision_02, recall_02, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label = [0, 1], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_02, recall_02, accuracy, f1))
        P_test[np.where(P_test == 0)] = 10
        P_test[np.where(P_test == 1)] = 13
        P_test[np.where(P_test == 2)] = 14
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
        result = [precision_00, recall_00, precision_01, recall_01, precision_02, recall_02]

    if label_train == 1:
        print('****************************************************************')
        print('evaluating heap....')
        print('****************************************')
        P_test = P_test.flatten()[0:(np.where(label_all_truth == 20)[0].shape[0]
                                     + np.where(label_all_truth == 22)[0].shape[0]
                                     + np.where(label_all_truth == 23)[0].shape[0]
                                     + np.where(label_all_truth == 24)[0].shape[0])]
        YY_test = YY_test.flatten()[0:P_test.shape[0]]

        print ('Heap...')
        (precision_10, recall_10, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_10, recall_10, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Heap...')
        (precision_11, recall_11, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_11, recall_11, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Stack...')
        (precision_12, recall_12, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_12, recall_12, accuracy, f1))
        print '-----------------------------'
        print ('Heap&Other...')
        (precision_13, recall_13, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_13, recall_13, accuracy, f1))
        print '-----------------------------'

        P_test[np.where(P_test == 0)] = 20
        P_test[np.where(P_test == 1)] = 22
        P_test[np.where(P_test == 2)] = 23
        P_test[np.where(P_test == 3)] = 24
        result = [precision_10, recall_10, precision_11, recall_11, precision_12, recall_12, precision_13, recall_13]
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'

    elif label_train == 2:
        print('****************************************************************')
        print('evaluating Stack....')
        print('****************************************')
        P_test = P_test.flatten()[0:(np.where(label_all_truth == 30)[0].shape[0]
                                     + np.where(label_all_truth == 32)[0].shape[0]
                                     + np.where(label_all_truth == 33)[0].shape[0]
                                     + np.where(label_all_truth == 34)[0].shape[0])]
        YY_test = YY_test.flatten()[0:P_test.shape[0]]
        (precision_20, recall_20, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_20, recall_20, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Heap...')
        (precision_21, recall_21, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_21, recall_21, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Stack...')
        (precision_22, recall_22, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_22, recall_22, accuracy, f1))
        print '-----------------------------'
        print ('Stack&Other...')
        (precision_23, recall_23, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_23, recall_23, accuracy, f1))
        print '-----------------------------'
        P_test[np.where(P_test == 0)] = 30
        P_test[np.where(P_test == 1)] = 32
        P_test[np.where(P_test == 2)] = 33
        P_test[np.where(P_test == 3)] = 34
        result = [precision_20, recall_20, precision_21, recall_21, precision_22, recall_22, precision_23, recall_23]
        
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'

    if label_train == 3:
        print('****************************************************************')
        print('evaluating other....')
        print('****************************************')
        P_test = P_test.flatten()[0:(np.where(label_all_truth == 40)[0].shape[0]
                                     + np.where(label_all_truth == 42)[0].shape[0]
                                     + np.where(label_all_truth == 43)[0].shape[0]
                                     + np.where(label_all_truth == 44)[0].shape[0])]
        YY_test = YY_test.flatten()[0:P_test.shape[0]]
      
        print ('Other&global...')
        (precision_30, recall_30, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=0, neg_label= [1, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_30, recall_30, accuracy, f1))
        print '-----------------------------'
        print ('Other&Heap...')
        (precision_31, recall_31, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=1, neg_label= [0, 2, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_31, recall_31, accuracy, f1))
        print '-----------------------------'
        print ('Other&Stack...')
        (precision_32, recall_32, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=2, neg_label= [0, 1, 3], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_32, recall_32, accuracy, f1))
        print '-----------------------------'
        print ('Other&Other...')
        (precision_33, recall_33, accuracy, f1) = perf_measure(yy_true = YY_test, yy_pred = P_test, pos_label=3, neg_label= [0, 1, 2], option = 0)
        print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision_33, recall_33, accuracy, f1))
        print '-----------------------------'
        P_test[np.where(P_test == 0)] = 41
        P_test[np.where(P_test == 1)] = 42
        P_test[np.where(P_test == 2)] = 43
        P_test[np.where(P_test == 3)] = 44

        result = [precision_30, recall_30, precision_31, recall_31, precision_32, recall_32, precision_33, recall_33]
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'

    return P_test, label_all_truth, label_train, result

def write_label(data_file, label_pred, train_flag):
    name = 'region_DL_'+str(train_flag)
    inst_file = os.path.join(data_file, 'binary')
    region_file = os.path.join(data_file , name)

    f_1 = open(region_file, 'w')
    with open(inst_file) as f:
        data_txt = f.readlines()
    ll = 0
    for i in xrange(len(data_txt)):
        data_tmp = re.sub('\s', '', data_txt[i])
        len_inst = (len(data_tmp)) / 2
        label_tmp = label_pred[ll : ll+len_inst]
        ll = ll + len_inst
        label_byte = np.unique(label_tmp)[0]
        max = np.where(label_tmp == label_byte)[0].shape[0]

        for j in np.unique(label_tmp):
            a = np.where(label_tmp == j)[0].shape[0]
            if a >= max:
                max = a
                label_byte = j
        if label_byte % 10 == 0:
            string = str((label_byte/10 -1)) + '\n'
        else:
            string = str((label_byte/10 -1)) + ':' + str((label_byte%10 -1)) + '\n'
        f_1.writelines(string)
    f_1.close()
    return (0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-binary_file', help='the path of malicious', type=str)
    parser.add_argument('-train_flag', type = int)
    args = parser.parse_args()
    testing_set = args.binary_file 
    
    train_flag = args.train_flag
    seq_len = 200
    print '**********************************************'
    print '**********************************************'
    print testing_set
    print '**********************************************'
    print '**********************************************'

    data_file = os.path.join('/home/wzg13/Data/newest_fse_traces', testing_set)
    #data_file = os.path.join('/Users/Henryguo/Desktop/testing', testing_set)

    inst_file = os.path.join(data_file, 'binary')
    label_file = os.path.join(data_file, 'region')
    data, label = process_data(inst_file, label_file)
    label = np.asarray(label)
    label_pred = np.zeros_like(label)

    result_all = []
    for label_train in xrange(4):
        P_test, label_all_truth, label_train, result = Testing(data_file, train_flag, label_train, seq_len)
        result_all.append(result)
        P_test = P_test.flatten()
        if label_train == 0:
            num_1 = np.where(label_all_truth==10)[0].shape[0]
            num_2 = np.where(label_all_truth == 13)[0].shape[0]
            num_3 = np.where(label_all_truth == 14)[0].shape[0]
            label_pred[np.where(label_all_truth==10)] = P_test[0:num_1]
            label_pred[np.where(label_all_truth==13)] = P_test[num_1:(num_1+num_2)]
            label_pred[np.where(label_all_truth==14)] = P_test[(num_1+num_2):(num_1+num_2+num_3)]
        
        elif label_train == 1:
            num_1 = np.where(label_all_truth == 20)[0].shape[0]
            num_2 = np.where(label_all_truth == 22)[0].shape[0]
            num_3 = np.where(label_all_truth == 23)[0].shape[0]
            num_4 = np.where(label_all_truth == 24)[0].shape[0]

            label_pred[np.where(label_all_truth == 20)] = P_test[0:num_1]
            label_pred[np.where(label_all_truth == 22)] = P_test[num_1:(num_1 + num_2)]
            label_pred[np.where(label_all_truth == 23)] = P_test[(num_1 + num_2):(num_1 + num_2+num_3)]
            label_pred[np.where(label_all_truth == 24)] = P_test[(num_1 + num_2 + num_3):(num_1 + num_2+num_3+num_4)]

        elif label_train == 2:
            num_1 = np.where(label_all_truth == 30)[0].shape[0]
            num_2 = np.where(label_all_truth == 32)[0].shape[0]
            num_3 = np.where(label_all_truth == 33)[0].shape[0]
            num_4 = np.where(label_all_truth == 34)[0].shape[0]

            label_pred[np.where(label_all_truth == 30)] = P_test[0:num_1]
            label_pred[np.where(label_all_truth == 32)] = P_test[num_1:(num_1 + num_2)]
            label_pred[np.where(label_all_truth == 33)] = P_test[(num_1 + num_2):(num_1 + num_2+num_3)]
            label_pred[np.where(label_all_truth == 34)] = P_test[(num_1 + num_2 + num_3):(num_1 + num_2+num_3+num_4)]

        elif label_train == 3:
            num_1 = np.where(label_all_truth == 41)[0].shape[0]
            num_2 = np.where(label_all_truth == 42)[0].shape[0]
            num_3 = np.where(label_all_truth == 43)[0].shape[0]
            num_4 = np.where(label_all_truth == 44)[0].shape[0]

            label_pred[np.where(label_all_truth == 41)] = P_test[0:num_1]
            label_pred[np.where(label_all_truth == 42)] = P_test[num_1:(num_1 + num_2)]
            label_pred[np.where(label_all_truth == 43)] = P_test[(num_1 + num_2):(num_1 + num_2+num_3)]
            label_pred[np.where(label_all_truth == 44)] = P_test[(num_1 + num_2 + num_3):(num_1 + num_2+num_3+num_4)]

    label_pred[np.where(label_pred==0)] = 40

    write_label(data_file, label_pred, train_flag)
    inst_file = os.path.join(data_file, 'binary')
    name_aa = 'region_DL_' + str(train_flag)
    label_file = os.path.join(data_file, name_aa)
    data, label_1 = process_data(inst_file, label_file)
    error = [ ]
    error.append(np.count_nonzero(label_1 - label))
    print error
    print np.count_nonzero(label_pred - label_1)
    result_all.append(error)
    final_stat = np.zeros((1, 31))
    k = 0
    print result_all
    for cell in result_all:
        for j in cell:
            final_stat[0,k] = j
            k = k+1
    name = testing_set + '__' + str(train_flag)
    io.savemat(name, {'result':final_stat})
