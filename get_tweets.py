import tweepy
import datetime
import pandas as pd
from tqdm import tqdm,trange
from dateutil.relativedelta import relativedelta


consumerKey = "0dS9NjpnigQFJZuXyjCogecw0" # (API key)
consumerSecret = "b2pf0DTvxnNsiOeAjgy6B6PPDb6J42W4jDNeHeyHEWnyw0ttJr" # (API secret key)
accessToken = "764037378-QCU4yGtyN4C9ZTK9hgg76Kr7JmAL1QiiZqA2moMC" # (Access token)NEW
accessTokenSecret = "HyVapADFE9cYRl72wRPPUXrRu8E1TKAUU1G9JltPg8m0O" # (Access token secret)NEW

def get_owner_tweets(handles,startDate,endDate,pages=20):
    global consumerKey, consumerSecret, accessToken, accessTokenSecret

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)

    api = tweepy.API(auth)
    pages = 20 # number of tweets 20 per page

    owner_usernames = handles #["@tesla","@elonmusk"]
    
    #startDate = datetime.datetime(year=2014,month=9,day=1,hour=0,minute=0,second=0)
    #endDate = datetime.datetime(year=2019,month=9,day=14,hour=0,minute=0,second=0)
    
    dict_tweets = []
    tweets = []
    skipped_twts_counter = 0
    skipped_calls = []
    for owner in tqdm(owner_usernames):
        for i in range(pages):
            try:
                page_twt = api.user_timeline(owner,page=i)
            except Exception as err:
                print(str(err))
                skipped_twts_counter +=1
                skipped_calls.append(owner)
                continue
            for twt in page_twt:
                if twt.created_at > startDate and twt.created_at < endDate:
                    tweets.append(twt)
                    date = twt.created_at.strftime("%Y-%m-%d")
                    time_stamp = twt.created_at.timestamp()
                    _dict_twt = {"text":twt.text,
                                "date":date,
                                "time_stamp":time_stamp,
                                "handle":owner}
                    dict_tweets.append(_dict_twt)
                else:
                    skipped_twts_counter +=1
    
    df = pd.DataFrame(dict_tweets)
    return df


def get_related_tweets(handles,keywords,startDate,endDate,pages=20):
    global consumerKey, consumerSecret, accessToken, accessTokenSecret

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)

    api = tweepy.API(auth)
    pages = 20 # number of tweets 20 per page

    related_usernames = handles #["@tesla","@elonmusk"]
    
    #startDate = datetime.datetime(year=2014,month=9,day=1,hour=0,minute=0,second=0)
    #endDate = datetime.datetime(year=2019,month=9,day=14,hour=0,minute=0,second=0)
    
    dict_tweets = []
    tweets = []
    skipped_twts_counter = 0
    for related_handle in tqdm(related_usernames):
        for i in range(pages):
            try:
                page_twt = api.user_timeline(related_handle,page=i)
            except Exception as err:
                print(str(err))
                print("err in handle:{} page:{}".format(related_handle,i))
                skipped_calls.append(owner)
                skipped_twts_counter +=1
                continue
            for twt in page_twt:
                txt = twt.text
                if not any(word in txt for word in keywords):
                    skipped_twts_counter+=1
                    continue
                if twt.created_at > startDate and twt.created_at < endDate:
                    tweets.append(twt)
                    date = twt.created_at.strftime("%Y-%m-%d")
                    time_stamp = twt.created_at.timestamp()
                    _dict_twt = {"text":twt.text,
                                "date":date,
                                "time_stamp":time_stamp,
                                "handle":related_handle}
                    dict_tweets.append(_dict_twt)
                else:
                    skipped_twts_counter +=1
    
    df = pd.DataFrame(dict_tweets)
    return df


def get_combined_tweets(main_handles,other_handles,keywords,endDate,pages):
    print("end_date",endDate)
    endDate = datetime.datetime.strptime(endDate, '%Y-%m-%d')
    start_year = endDate.year - 1
    startDate = datetime.datetime(year=start_year,month=endDate.month,day=endDate.day)
    print("startDate",startDate)
    print("endDate",endDate)

    tweets_df_owner = get_owner_tweets(main_handles,startDate,endDate,pages)
    tweets_df_related = get_related_tweets(other_handles,keywords,startDate,endDate,pages)
    df_combined = pd.concat([tweets_df_owner, tweets_df_related])

    return df_combined

