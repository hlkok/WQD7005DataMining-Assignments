#!/usr/bin/env python
# coding: utf-8

# In[9]:


#------------------------------------------------
# CRAWL FOR COVID-19 RELATED TWEETS FROM TWITTER:
#------------------------------------------------
# Course: WQD7005 Data Mining | Master of Data Science @ University of Malaya
# Group Members: Azwa Kamaruddin (WQD170089), Kok Hon Loong (WQD170086)
# Assignment: Milestone 3 - Accessing and Processing of Data from Hadoop Data Warehouse or Data Lake using Python
# This code is to crawl for Covid-19 related tweets from Twitter.
# The raw data will be STORED into Google Cloud Storage data lake in JSON format.


# In[10]:


# Import required libraries:
from google.cloud import storage
from io import BytesIO, StringIO
import os
import sys
import json
import tweepy as tw
import pandas as pd
import re


# In[11]:


# Use Twitter API to fetch tweets from the platform.
# Twitter API credentials are stored in a GCP bucket to prevent unauthorised used from our public GitHub repo.
client = storage.Client()
bucket = client.get_bucket('wqd7005dm-authorization')

blob_authentication = bucket.get_blob('authentication.csv')
key = blob_authentication.download_as_string()

s=str(key,'utf-8')
df = StringIO(s)
key_df = pd.read_csv(df,header=None)

consumer_key = key_df.iloc[0][0]
consumer_secret = key_df.iloc[0][1]
access_token = key_df.iloc[0][2]
access_token_secret = key_df.iloc[0][3]


# In[12]:


# Use Tweepy library to read twitter data:
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, parser=tw.parsers.JSONParser())


# In[13]:


# -----------------------------
# CRAWLING tweets from Twitter:
# -----------------------------
# Each similar terms are seperated by the OR operator
# We exclude any retweets and tweets with links
q = "'corona virus' OR 'coronavirus' OR 'covid19' OR 'covid-19' -filter:retweets -filter:links"
results = api.search(
            q,
            lang = 'en',
            count = 100
        )


# In[14]:


# Test to check the first crawled tweet in JSON format:
results["statuses"][0]


# In[7]:


# --------------------------------------
# STORING Twitter data to the data lake:
# --------------------------------------
with open('twitter_data.json', 'w') as outfile:
    json.dump(results, outfile)


# In[8]:


get_ipython().system("gsutil cp 'twitter_data.json' 'gs://wqd7005dm-covid19-ds/'")


# In[ ]:




