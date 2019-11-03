import os
import numpy as np
import re
from os import listdir
from os.path import join


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
        #print label_tmp
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

mypath = '/home/wzg13/Data/newest_fse_traces'
onlyfiles = [ f for f in listdir(mypath)]
print len(onlyfiles)
data = []
label = []
i = 0
for file in onlyfiles:
    i = i+1
    print (i)
    data_file = os.path.join(mypath, file)
    inst_file = os.path.join(data_file, 'binary')
    label_file = os.path.join(data_file, 'region')
    data_temp, label_temp = process_data(inst_file, label_file)
    data.extend(data_temp)
    label.extend(label_temp)

data = np.asarray(data)
label = np.asarray(label)
print data.shape
print label.shape

print '10: global...'
print np.where(label==10)[0].shape[0]
print np.where(label==11)[0].shape[0]
print np.where(label==12)[0].shape[0]
print np.where(label==13)[0].shape[0]
print np.where(label==14)[0].shape[0]
        
print '20: heap...'
print np.where(label==20)[0].shape[0]
print np.where(label==21)[0].shape[0]
print np.where(label==22)[0].shape[0]
print np.where(label==23)[0].shape[0]
print np.where(label==24)[0].shape[0]

print '30: stack...'
print np.where(label==30)[0].shape[0]
print np.where(label==31)[0].shape[0]
print np.where(label==32)[0].shape[0]
print np.where(label==33)[0].shape[0]
print np.where(label==34)[0].shape[0]

print '40: other...'
print np.where(label==40)[0].shape[0]
print np.where(label==41)[0].shape[0]
print np.where(label==42)[0].shape[0]
print np.where(label==43)[0].shape[0]
print np.where(label==44)[0].shape[0]
