#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas
from pandas import Series
from datetime import datetime
import numpy as np
import datetime as dt


# In[37]:


df_msft = pandas.read_csv("TSLA.csv")
df_news = pandas.read_csv("E:/hackathon_devengers/tesla_1-1-18_14-9-19.csv")


# In[ ]:





# In[38]:



df_msft


# In[39]:


df_news


# In[40]:


df_msft = df_msft[(df_msft['Date'] > '2018-09-13') & (df_msft['Date'] < '2019-09-13')]
df_msft


# In[41]:


#t = technology
#df_technology = df_news[df_news['CATEGORY'] == 't']
df_technology = df_news[df_news['text'].str.contains('tesla', case=False)]
sLength = len(df_technology['date'])
#df_technology['today'] = Series(np.zeros(sLength), index=df_technology.index, dtype=np.int32)
#df_technology['tomorrow'] = Series(np.zeros(sLength), index=df_technology.index, dtype=np.int32)

#df_technology['date'] = df_technology['date'].apply(lambda x: datetime.fromtimestamp(int(int(x)/1000)).strftime('%Y-%m-%d'))


# In[42]:


df_technology


# In[43]:


df_msft


# In[44]:


def get_data(date,window,sentiment,df,normalize = True):
    
    #print(df)
    all_dates = df['Date']
    value = df['Close'].where(df['Date'] == date)
    print(value)
    c = np.where(df['Close'] == value)
    
    search_index = int(c[0])
    print(search_index)
    val1 = df['Close'][search_index]
    val2 = df['Open'][search_index]
    print(val1,val2)
    #test_set = list(df['Close'][search_index - window:search_index])
    #test_set = np.array(test_set)
    #print(test_set)
    #thresh_val = test_set[0]
#     if normalize == True:
#         test_set = np.array([_get_cummulative_return(test_set)])
    return val1


# In[45]:


value = get_data("2019-08-30",200,"pos",df_msft,True)


# In[46]:


def findStockChange(row, dataset = df_msft, dateOffset = 0):
    currentstockDay = None
    date = datetime.strptime(row, '%Y-%m-%d')
    date = date + dt.timedelta(days=dateOffset)
    row = date.strftime('%Y-%m-%d')
    currentstockDay = dataset[dataset['Date'] == row]
    if not currentstockDay.empty:
        return currentstockDay.iloc[0]['Close'] > currentstockDay.iloc[0]['Open']
    else:
        return False

    
df_technology['ms_today'] = df_technology['date'].apply(lambda row: findStockChange(row)).astype(np.int32)


# In[48]:


df_technology


# In[49]:


df_technology['ms_tomorrow'] = df_technology['date'].apply(lambda row: findStockChange(row, dateOffset=1)).astype(np.int32)


# In[50]:


df_technology


# In[51]:


def normalize_headline(row):
    result = row.lower()
    #Delete useless character strings
    result = result.replace('...', '')
    whitelist = set('abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?')
    result = ''.join(filter(whitelist.__contains__, result))
    return result

df_technology['normalized_headline'] = df_technology['text'].apply(normalize_headline)


# In[52]:


df_technology


# In[55]:


df_technology.to_csv('combined_tesla_news_stocks.csv', sep='\t')


# In[ ]:




