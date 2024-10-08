import numpy as np
import pandas as pd 
import os
from transformers import TFBertModel,BertTokenizer
from sklearn.model_selection import train_test_split,ShuffleSplit
import scipy as sp
import time
from sklearn.metrics import roc_auc_score
from model import DeepIndel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model_name = "./bert-base-uncased"
bert_model = TFBertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("./vocab.txt",do_lower_case=False)

file = "./dataset/K562 (n=35129).csv"
datafile = pd.read_csv(file)
seqs = datafile['seqs']
labels = np.array(pd.concat((datafile['delf'],datafile['1insf'],datafile['1delf'],datafile['1fsf'],datafile['2fsf'],datafile['fsf']),axis=1,names=None),dtype='float')

def data_process(seqs,labels,tokenizer):
   Y = []
   kmers = []
   for seq,label in zip(seqs,labels):
      kmer1 = [seq[i:i+1] for i in range(len(seq) - 1 + 1)]
      kmer2 = [seq[i:i+2] for i in range(len(seq) - 2 + 1)]
      kmer3 = [seq[i:i+3] for i in range(len(seq) - 3 + 1)]
      kmers.append(" ".join(kmer1 + kmer2 + kmer3))#
      Y.append(label)
   token = tokenizer(kmers,return_tensors="tf",padding=True,truncation=True)
   return np.array(token["input_ids"]),np.array(Y)

def cross_valiadtion(X, Y,n_split=5):
   train_test_data = []
   cv = ShuffleSplit(n_splits=n_split,test_size=0.1,random_state=50)
   for train_index,test_index in cv.split(X):
      X1_train,X1_test = X[train_index],X[test_index]
      Y_train,Y_test = Y[train_index],Y[test_index]
      train_test_data.append((X1_train,X1_test,Y_train,Y_test))
   return train_test_data

config = bert_model.config
config.vocab_size = 86
X, Y = data_process(seqs, labels)
data_list = cross_valiadtion(X, Y)

for data in data_list:
   X_train, X_test, Y_train, Y_test = data[0], data[1], data[2], data[3]
   model = DeepIndel(bert_model)
   early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
   adam = Adam(learning_rate=0.00001)
   model.compile(loss='binary_crossentropy', optimizer=adam)
   model.fit([X_train],Y_train, batch_size=64,epochs=200, verbose=2, validation_split=0.1 ,shuffle=False, callbacks=[early_stopping])