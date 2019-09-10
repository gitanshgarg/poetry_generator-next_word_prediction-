# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam




# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 2000
LATENT_DIM = 25

# load in the data
input_texts = []
target_texts = []
for line in open('robert_frost.txt'):
  line = line.rstrip()
  if not line:
    continue

  input_line = '<sos> ' + line
  target_line = line + ' <eos>'

  input_texts.append(input_line)
  target_texts.append(target_line)


all_lines = input_texts + target_texts

# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# find max seq length
max_sequence_length_from_data = max(len(s) for s in input_sequences)
print('Max sequence length:', max_sequence_length_from_data)


# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))
assert('<sos>' in word2idx)
assert('<eos>' in word2idx)


# pad sequences so that we get a N x T matrix
max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
print('Shape of data tensor:', input_sequences.shape)



# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('glove/glove.6B.%sd.txt' % EMBEDDING_DIM),encoding="utf-8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))



# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector



# one-hot the targets (can't use sparse cross-entropy)
one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
for i, target_sequence in enumerate(target_sequences):
  for t, word in enumerate(target_sequence):
    if word > 0:
      one_hot_targets[i, t, word] = 1
      
      
      
      
      
#creating a embeding layer with pre trained weights
      
embedding=Embedding(num_words,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False);


input1=Input(shape=(max_sequence_length,));
input_c=Input(shape=(LATENT_DIM,));
input_h=Input(shape=(LATENT_DIM,));


x=embedding(input1);#N*T*D;
lstm=LSTM(LATENT_DIM,return_state=True,return_sequences=True);
x,_,_=lstm(x,initial_state=[input_c,input_h]);#N*T*LATENT_DIM
dense=Dense(num_words,activation="softmax");
output=dense(x);#N*T*V

model_train=Model([input1,input_c,input_h],[output])

model_train.compile(
  loss='categorical_crossentropy',
  # optimizer='rmsprop',
  optimizer=Adam(lr=0.01),
  
  metrics=['accuracy']
)
print("TRAINING MOEL.....");
z=np.zeros((len(input_sequences),LATENT_DIM));
r=model_train.fit([input_sequences,z,z],one_hot_targets, batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT)


print("results ....");



# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()


input_s=Input(shape=(1,));#1*1;
x1=embedding(input_s);
out,h,c=lstm(x1,initial_state=[input_c,input_h]);
out=dense(out);#1*1*V;


model_sample=Model([input_s,input_c,input_h],[out,h,c]);

idx2word={k:v for v,k in word2idx.items()};


def sample_line():
    
    h=np.zeros((1,LATENT_DIM));
    c=np.zeros((1,LATENT_DIM));
    
    inp=np.array([[word2idx['<sos>']]]);
    eos=word2idx['<eos>'];
    
    output_sen=[]
    for i in range(max_sequence_length):
        o,h,c=model_sample.predict([inp,h,c]);
        
        prob=o[0,0];
        
        if(np.argmax(prob)==0):
            print("wtf");
        
        
        prob[0]=0;
        prob/=prob.sum();
        
        idx=np.random.choice(len(prob),p=prob);
        
        if(idx==eos):
            break;
            
        output_sen.append(idx2word.get(idx));
        inp[0,0]=idx;
        
    return ' '.join(output_sen);

while(True):
    for i in range(4):
        print(sample_line());
        
    j=input("continue Y/N");
    if(j=='N'):
        break;
    
    
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    








































































































      
      
      
      
      
      
      
      
      
      
      