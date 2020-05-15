#!/usr/bin/env python
# coding: utf-8

# In[6]:


#---------------------------------------------------------------
# ACCESS TWEETS FROM GOOGLE CLOUD STORAGE (DATA LAKE):
#---------------------------------------------------------------
# Course: WQD7005 Data Mining | Master of Data Science @ University of Malaya
# Group Members: Azwa Kamaruddin (WQD170089), Kok Hon Loong (WQD170086)
# Assignment: Milestone 3 - Accessing and Processing of Data from Hadoop Data Warehouse or Data Lake using Python
# This code is to ACCESS the stored crawled tweets in Google Cloud Storage data lake.


# In[7]:


import pandas as pd
import json
from google.cloud import storage
from io import BytesIO, StringIO


# In[8]:


client = storage.Client()
bucket = client.get_bucket('wqd7005dm-covid19-ds')
blob = bucket.get_blob('twitter_data.json')
data = blob.download_as_string()


# In[9]:


# Read the JSON file from the data lake:
s=str(data,'utf-8')
df = StringIO(s)


# In[10]:


# Load the JSON file here:
json.load(df)


# In[ ]:




