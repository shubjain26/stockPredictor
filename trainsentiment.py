#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import svm
from sklearn.model_selection import cross_val_score

from dateutil.parser import parse
from datetime import timedelta

import random
import re


# In[ ]:





# In[23]:




def import_data(filename):
	return pd.read_csv(filename, header=0, sep = '\t').fillna('').values


def pre_process(data):
	return data


def add_sentiment(data):
    sid = SentimentIntensityAnalyzer()
    avgs = np.empty((len(data),1))
    print(len(data))
    for i in range(0,len(data)):
        field = data[i][8]
        avgs[i] = sid.polarity_scores(field)['compound']
    return np.append(data, avgs, axis=1)


# In[24]:


def calc_avg_sentiments(data):
	previousDate = parse(data[0,8]) - timedelta(days=1)
	previousMs = -1
	previousMsTom = -1
	tempSents = [0]
	output = np.empty((1,3))
	for line in data:
		date = parse(line[8])
		sentiment = line[12]
		ms = line[9]
		msTom = line[10]
		if date > previousDate:
			avgSent = sum(tempSents)/len(tempSents)
			output = np.append(output,[[ms, msTom, avgSent]], axis=0)

			tempSents = []
			tempSents.append(sentiment)
			previousDate = date
			previousMs = ms
			previousMsTom = msTom
		elif date == previousDate:
			tempSents.append(sentiment)
		else:
			print("Wrong date order.")
	return output[1:]	# First row was to initialize. Ugly, but works.


# In[25]:


def filter_data(data):
	output = np.empty((1,3))
	for line in data:
		ms = line[6]
		msTom = line[7]
		#print(line[9])
		sentiment = line[9]

		output = np.append(output, [[ms, msTom, sentiment]], axis=0)
	return output[1:]


def divide_train_test(data):
	random.seed(2)
	random.shuffle(data)

	split = .7 * len(data)
	train = data[:int(split)]
	test = data[int(split):]
	return train, test


# In[26]:


def train(data, labels):
	return svm.SVC(kernel='linear', C=1).fit(data, labels)


def cross_validate(data, labels, clsfr):
	return cross_val_score(clsfr, data, labels, cv=2)


# In[31]:


date_to_search = "2019-09-13"
def Get_sentiment_results(date_to_search):
    test_specific_date_sent = []
    tweets_caused = []
    data = import_data("combined_tesla_news_stocks.csv")
    #print(data)
    pre_processed = pre_process(data)
    sentiment_included = add_sentiment(pre_processed)
    #print(sentiment_included)
    for j in range(0,len(sentiment_included)):
        date = sentiment_included[j][2]
        if(date_to_search == date):
            sentiment = sentiment_included[j][9]
            tweets_cause = sentiment_included[j][8]
            tweets_caused.append(tweets_cause)            
            test_specific_date_sent.append(sentiment)
    
    test_date_sentiments = np.array([test_specific_date_sent]).reshape(len(test_specific_date_sent),1)
    #print(np.array([test_specific_date_sent]).reshape(len(test_specific_date_sent),1))
    
    #print(len(sentiment_included))
    avg_sentiments = filter_data(sentiment_included)

    print("Data exists of " + str(avg_sentiments.shape[0]) + " cases.\n")

    # Predict today's stock
    train_set, test_set = divide_train_test(avg_sentiments)

    
    train_labels = train_set[:,0].ravel()
    train_sentiments = train_set[:,2].reshape(len(train_set), 1)
    test_labels = test_set[:,0].ravel()
    #print("test_labels::")
    #print(test_labels)
    test_sentiments = test_set[:,2].reshape(len(test_set),1)
    clsfr = train(train_sentiments, train_labels.astype('int'))
    print("Training done.")
    print("Pedicting today's stock...")
    #print(test_sentiments)
    scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
    print(scores)
    #print(test_labels)
    print(clsfr.predict(test_sentiments))
    print("Average: " + str(scores.mean()))
    print("Cross validation done.\n")
    
    print("Prediction for specific date!!!")
    
    pred_arr = clsfr.predict(test_date_sentiments)
    
    print("Average score for date:"+ str(pred_arr.mean()))
    if(pred_arr.mean() > 0.5):
        final_result = "Price is going to increasee!"
        print("Price is going to increasee!")
    else:
        final_result = "Price is going to decrease!"
        print("Price is going to decrease!")
    return final_result, tweets_caused
    
    
    
    

    # Predict tomorrow's stock
#     train_set, test_set = divide_train_test(avg_sentiments)
#     train_labels = train_set[:,1].ravel()
#     train_sentiments = train_set[:,].reshape(len(train_set), 1)
#     test_labels = test_set[:,1].ravel()
#     test_sentiments = test_set[:,].reshape(len(test_set),1)

#     clsfr = train(train_sentiments, train_labels.astype('int'))
#     print("Training done.")
#     print("Predicting tomorrow's stock...")
# 	scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
# 	print(scores)
# 	print("Average: " + str(scores.mean()))
# 	print("Cross validation done.\n")
# 	# Predict the day after tomorrow's stock
# 	train_set, test_set = divide_train_test(avg_sentiments)
# 	train_labels = train_set[:,2].ravel()
# 	train_sentiments = train_set[:,3].reshape(len(train_set), 1)
# 	test_labels = test_set[:,2].ravel()
# 	test_sentiments = test_set[:,3].reshape(len(test_set),1)

# 	clsfr = train(train_sentiments, train_labels.astype('int'))
# 	print("Training done.")
# 	print("Predicting the day after tomorrow's stock...")
# 	scores = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)
# 	print(scores)
# 	print("Average: " + str(scores.mean()))
# 	print("Cross validation done.")
# 	print("Done.")


# In[32]:


Get_sentiment_results(date_to_search)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




