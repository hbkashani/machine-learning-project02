#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:00:01 2017

@author: haniehkashani
"""

import pandas as pd
import numpy as np
import random
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import string


train_input = pd.read_csv('/Users/haniehkashani/Documents/Machine learning mcgill/project02/train_input.csv')
#training_input = training_input[['id','conversation']]
train_output = pd.read_csv('/Users/haniehkashani/Documents/Machine learning mcgill/project02/train_output.csv') 

# Computing Priors (i.e percentage of each class in traning data)
counts = pd.crosstab(index=train_output['category'], columns="count") 
prior = counts['count']/len(train_output.axes[0])


# Choosing a random set of conversations   
from random import sample
random.seed(123)
l = np.size(train_input['conversation']) #length of data
f = 1000 #coversation to train on quickly
indices = sample(range(1,l),f)
train_input_sample = train_input.iloc[indices] #use iloc to consider rows instead of columns
train_output_sample = train_output.iloc[indices]

# Attaching input and output
#train_IO = pd.concat([train_input,train_output['category']],axis=1)
train_IO_sample = pd.concat([train_input_sample,train_output_sample['category']],axis=1)

# Concatenate all words in the same class
#attached_train = train_IO.groupby(['category']).agg(lambda col:','.join(col))
attached_train_sample = train_IO_sample.groupby(['category']).agg(lambda col:','.join(col))



# Sanitizing once Concatenated all words in same Class
numCat = np.size(attached_train_sample['conversation'])
# 1) Get rid of all words starting with "<" (i.e. <speaker_#>, </s>, etc.)
clean_train_attached_sample = pd.DataFrame(index = attached_train_sample.index, columns = attached_train_sample.columns) 
for i in range(0,numCat):
    clean_train_attached_sample.iloc[i] = ' '.join(word for word in attached_train_sample['conversation'][i].split(' ') if not word.startswith('<'))
# 2) Get rid of user names, i.e. starting with "@"
clean_train_attached_sample_2 = pd.DataFrame(index = attached_train_sample.index, columns = attached_train_sample.columns)    
for i in range(0,numCat):
    clean_train_attached_sample_2.iloc[i] = ' '.join(word for word in clean_train_attached_sample['conversation'][i].split(' ') if not word.startswith('@'))
# 3) Get rid of all Stopwords
clean_train_attached_sample_3 = pd.DataFrame(index = attached_train_sample.index, columns = attached_train_sample.columns)      
for i in range(0,numCat):
    clean_train_attached_sample_3.iloc[i] = ' '.join(word for word in clean_train_attached_sample_2['conversation'][i].split(' ') if word not in stopwords.words('english'))
# 4) Removing punctuation
clean_train_attached_sample_4 = pd.DataFrame(index = attached_train_sample.index, columns = attached_train_sample.columns)      
for i in range(0,numCat):
    clean_train_attached_sample_4.iloc[i] = ' '.join(word for word in clean_train_attached_sample_3['conversation'][i].split(' ') if word not in string.punctuation)
# 5) Removing "com", "'s" and "'t"
clean_train_attached_sample_5 = pd.DataFrame(index = attached_train_sample.index, columns = attached_train_sample.columns)      
for i in range(0,numCat):
    clean_train_attached_sample_5.iloc[i] = clean_train_attached_sample_4['conversation'][i].replace('com','').replace('\'s','').replace('\'t','')

   
    
    
    
    
# SKLEARN: Step 1 - Counting word occurrences
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(clean_train_attached_sample_5['conversation'])
print(X_train_counts)
#word_freq = count_vect.vocabulary_.get(u'the')
#print(word_freq)

# SKLEARN: Step 2 - Getting word frequency, i.e. our Feature Matrix for Naive Bayes (=condprob)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)




for index, row in clean_train_attached_sample_3.iterrows():
    clean_train_attached_sample_3.loc[index, 'conversation'] = row['conversation'].replace('. com', '').replace('<speaker_1>', '').replace('<speaker_2>', '').replace('</s>','').replace('i','').replace(',','').replace('the','').replace('.','').replace('to','').replace('a','').replace('?','').replace('of','').replace('n','').replace('t','').replace('<speaker_3>','').replace('-','').replace('tht','').replace('on','').replace('d','').replace('*','').replace('</d>','')
    #print row['conversation']
    from collections import Counter
    print (index, Counter(row['conversation'].split()).most_common()[:15])



# =========== Naive Bayes Classifier ============

