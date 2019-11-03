import os
os.environ["THEANO_FLAGS"] = "device=gpu0,floatX=float32"
import numpy as np
from ValueSet_RNN import ValueSet_RNN
from os import listdir
mypath = '/home/wzg13/Data/newest_fse_traces'

cases = [f for f in listdir(mypath)]

cases.sort()
training_set = cases[0:45]
testing_set = cases[45:64]

batch = 200
epochs = 18
seq_len = 200

train_flag  = 3

for label in xrange(4):
	ValueSet_RNN(training_set = training_set, testing_set = [], train_flag = train_flag, batch_size=batch, epochs=epochs, label_train =label, seq_len=seq_len)
