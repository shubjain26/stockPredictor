#!/usr/bin/env python
# coding: utf-8


import bulbea as bb
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas 
import pandas_datareader.data as web
import collections
import numbers
from bulbea.learn.models import RNN
from sklearn.metrics import mean_squared_error


class ShareObj():

    def to_get_data(self,source, tinker):
        end=dt.datetime.now()
        start = dt.datetime(end.year-1,end.month,end.day)
        df = web.DataReader(tinker,source,start,end)
        df.to_csv(str(tinker+".csv"))
        #df.head()
        #df[['High','Low','Open','Close']].plot()
        return df

    def __init__(self,source,code):
        self.code=code
        self.data=self.to_get_data(source,code)
    

def split(share,
          attrs     = 'Close',
          window    = 0.01,
          train     = 0.80,
          shift     = 1,
          normalize = False):
    '''
    :param attrs: `str` or `list` of attribute names of a share, defaults to *Close* attribute
    :type attrs: :obj: `str`, :obj:`list`
    '''
    #_check_type(share, type_ = bb.Share, raise_err = True, expected_type_name = 'bulbea.Share')
    _check_iterable(attrs, raise_err = True)
    _check_int(shift, raise_err = True)
    _check_real(window, raise_err = True)
    _check_real(train, raise_err = True)

    _validate_in_range(window, 0, 1, raise_err = True)
    _validate_in_range(train, 0, 1, raise_err = True)

    data   = share.data[attrs]
    #print(data)

    length = len(share.data)
    #print(length)

    window = int(np.rint(length * window))
    offset = shift - 1
    #print(window,offset)

    splits = np.array([data[i if i is 0 else i + offset: i + window] for i in range(length - window)])

    #print(splits.shape)

    if normalize:
        splits = np.array([_get_cummulative_return(split) for split in splits])

    size   = len(splits)
    split  = int(np.rint(train * size))

    train  = splits[:split,:]
    test   = splits[split:,:]

    Xtrain, Xtest = train[:,:-1], test[:,:-1]
    ytrain, ytest = train[:, -1], test[:, -1]

    return (Xtrain, Xtest, ytrain, ytest)
    

def _check_iterable(o, raise_err = False):
    return _check_type(o, collections.Iterable, raise_err = raise_err, expected_type_name = '(str, list, tuple)')
def _check_int(o, raise_err = False):
    return _check_type(o, numbers.Integral, raise_err = raise_err, expected_type_name = 'int')
def _check_real(o, raise_err = False):
    return _check_type(o, numbers.Real, raise_err = raise_err, expected_type_name = '(int, float)')
def _validate_in_range(value, low, high, raise_err = False):
    if not low <= value <= high:
        if raise_err:
            raise ValueError('{value} out of bounds, must be in range [{low}, {high}].'.format(
                value = value,
                low   = low,
                high  = high
            ))
        else:
            return False
    else:
        return True
def _check_type(o, type_, raise_err = False, expected_type_name = None):
    if not isinstance(o, type_):
        if raise_err:
            _raise_type_error(
                expected_type_name = expected_type_name,
                recieved_type_name = _get_type_name(o)
            )
        else:
            return False
    else:
        return True


def _get_cummulative_return(data):
    cumret  = (data / data[0]) - 1
    return cumret

def get_data(date,window,sentiment,df,normalize = True):
    #print(df)
    all_dates = df['Date']
    try:             
        value = df['Close'].where(df['Date'] == date)
        c = np.where(df['Close'] == value)
        search_index = int(c[0])
        test_set = list(df['Close'][search_index - window:search_index])
        test_set = np.array(test_set)
        print(test_set)
        thresh_val = test_set[0]
        if normalize == True:
            test_set = np.array([_get_cummulative_return(test_set)])
            
        return test_set, thresh_val
    except:
        print("date is not there!!")
        date = pandas.to_datetime(date)
        date = date - np.timedelta64(1,'D')
        #print(date)
        new_date = pandas.to_datetime(date).date()
        #print(str(new_date))
        test_set, thresh_val = get_data(str(new_date),window,sentiment,df,normalize = True)
        return test_set, thresh_val
        

def get_prediction_by_date(input_date, stock_name):
    share = ShareObj("yahoo",stock_name)
    print("DATA OF GIVEN STOCK PRICE FOR 1 YEAR")
    print(share.data)
    len(share.data)
    
    Xtrain, Xtest, ytrain, ytest = split(share, 'Close',window= 0.02, normalize=True)
    print(Xtrain.shape)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))



    rnn = RNN([1, 100, 100, 1]) # number of neurons in each layer
    rnn.fit(Xtrain, ytrain, batch_size = 64, nb_epoch= 10, validation_split= 0.01)
    
    print("Analysing closing price for before few days of given date..")
    dates = share.data.index
    df = share.data.reset_index()
    test_data, thresh_val= get_data(input_date,Xtrain.shape[1],"positive",df, True)
    print(test_data, thresh_val)
    print(test_data.shape)
    
    p = rnn.predict(Xtest)
    mean_squared_error(ytest, p)
    

    test_data  = np.reshape( test_data, ( test_data.shape[0],  test_data.shape[1], 1))
    pred = rnn.predict(test_data)
    print(pred)
    print("predicted value for given date:")
    print((pred+1)*thresh_val)
    predicted_val = (pred+1)*thresh_val
    return predicted_val[0][0] 


if __name__ == "__main__":
    input_date = "2019-09-14"
    stock_name = "AMZN"
    # source_csv = {stock_name: str(stock_name+'.csv')}
    predicted_val = get_prediction_by_date(input_date, stock_name)
    print(predicted_val)

