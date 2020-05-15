#!/usr/bin/env python
# coding: utf-8

# In[17]:


#---------------------------------------------------------------
# ACCESS TIME-SERIES DATA FROM GOOGLE CLOUD STORAGE (DATA LAKE):
#---------------------------------------------------------------
# Course: WQD7005 Data Mining | Master of Data Science @ University of Malaya
# Group Members: Azwa Kamaruddin (WQD170089), Kok Hon Loong (WQD170086)
# Assignment: Milestone 3 - Accessing and Processing of Data from Hadoop Data Warehouse or Data Lake using Python
# This code is to ACCESS the stored crawled data from Google Cloud Storage data lake.
# Data crawled are global data but only ASEAN countries and the consolidated regions in China will be extracted for analysis.


# In[18]:


import pandas as pd
from google.cloud import storage
from io import BytesIO, StringIO


# In[19]:


client = storage.Client()
bucket = client.get_bucket('wqd7005dm-covid19-ds')


# In[20]:


blob_confirmed = bucket.get_blob('confirmed_cases.csv')
blob_deaths = bucket.get_blob('death_cases.csv')
blob_recovered = bucket.get_blob('recovered_cases.csv')


# In[21]:


data_confirmed = blob_confirmed.download_as_string()
data_deaths = blob_deaths.download_as_string()
data_recovered = blob_recovered.download_as_string()


# In[22]:


# Read confirmed cases csv file from the data lake:
s_confirmed=str(data_confirmed,'utf-8')
df_confirmed = StringIO(s_confirmed)
confirmed_cases = pd.read_csv(df_confirmed)


# In[23]:


# Read death cases csv file from the data lake:
s_deaths=str(data_deaths,'utf-8')
df_deaths = StringIO(s_deaths)
death_cases = pd.read_csv(df_deaths)


# In[24]:


# Read recovered cases csv file from the data lake:
s_recovered=str(data_recovered,'utf-8')
df_recovered = StringIO(s_recovered)
recovered_cases = pd.read_csv(df_recovered)


# In[25]:


# * run this code only once*

# Drop unneeded columns from the dataframe:
confirmed_cases = confirmed_cases.drop('Unnamed: 0', axis=1)
confirmed_cases = confirmed_cases.drop(['Lat','Long'], axis=1)
confirmed_cases = confirmed_cases.drop('Province/State', axis=1)

death_cases = death_cases.drop('Unnamed: 0', axis=1)
death_cases = death_cases.drop(['Lat','Long'], axis=1)
death_cases = death_cases.drop('Province/State', axis=1)

recovered_cases = recovered_cases.drop('Unnamed: 0', axis=1)
recovered_cases = recovered_cases.drop(['Lat','Long'], axis=1)
recovered_cases = recovered_cases.drop('Province/State', axis=1)

# Rename column header to simply 'Country' and set it as the index:
confirmed_cases = confirmed_cases.rename(columns={'Country/Region':'Country'})
confirmed_cases = confirmed_cases.set_index('Country')

death_cases = death_cases.rename(columns={'Country/Region':'Country'})
death_cases = death_cases.set_index('Country')

recovered_cases = recovered_cases.rename(columns={'Country/Region':'Country'})
recovered_cases = recovered_cases.set_index('Country')


# In[26]:


# Obtain only the confirmed cases for the 10 ASEAN countries + China

# Get data for ASEAN regions:
asean = ['Malaysia', 'Singapore', 'Thailand', 'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Myanmar', 'Philippines', 'Vietnam']

confirmed_cases_asean = confirmed_cases[confirmed_cases.index.isin(asean)]
death_cases_asean = death_cases[death_cases.index.isin(asean)]
recovered_cases_asean = recovered_cases[recovered_cases.index.isin(asean)]


# In[28]:


confirmed_cases_asean


# In[29]:


death_cases_asean


# In[30]:


recovered_cases_asean


# In[31]:


# China:
confirmed_cases_china = confirmed_cases[confirmed_cases.index == 'China']
confirmed_cases_china_combined = confirmed_cases_china.groupby('Country').sum()

death_cases_china = death_cases[death_cases.index == 'China']
death_cases_china_combined = death_cases_china.groupby('Country').sum()

recovered_cases_china = recovered_cases[recovered_cases.index == 'China']
recovered_cases_china_combined = recovered_cases_china.groupby('Country').sum()


# In[32]:


# Transpose table so that we have a more logical structure for time series data:
# ASEAN:
t_confirmed_cases_asean = confirmed_cases_asean.T
t_death_cases_asean = death_cases_asean.T
t_recovered_cases_asean = recovered_cases_asean.T

# China:
t_confirmed_cases_china_combined = confirmed_cases_china_combined.T
t_death_cases_china_combined = death_cases_china_combined.T
t_recovered_cases_china_combined = recovered_cases_china_combined.T


# In[33]:


t_confirmed_cases_asean


# In[34]:


# The above transposed table of ASEAN and China covid-19 data are now ready for analysis in Milestone 4.


# In[18]:


# # Convert dataframe to csv file:
# t_confirmed_cases_asean.to_csv('t_confirmed_cases_asean.csv', index=True, header=True)
# t_death_cases_asean.to_csv('t_death_cases_asean.csv', index=True, header=True)
# t_recovered_cases_asean.to_csv('t_recovered_cases_asean.csv', index=True, header=True)

# t_confirmed_cases_china_combined.to_csv('t_confirmed_cases_china_combined.csv', index=True, header=True)
# t_death_cases_china_combined.to_csv('t_death_cases_china_combined.csv', index=True, header=True)
# t_recovered_cases_china_combined.to_csv('t_recovered_cases_china_combined.csv', index=True, header=True)


# In[19]:


# # Save csv file into Google Cloud Storage datalake:

# # ASEAN time series data:
# !gsutil cp 't_confirmed_cases_asean.csv' 'gs://wqd7005-covid19-data'
# !gsutil cp 't_death_cases_asean.csv' 'gs://wqd7005-covid19-data'
# !gsutil cp 't_recovered_cases_asean.csv' 'gs://wqd7005-covid19-data'

# # China time series data:
# !gsutil cp 't_confirmed_cases_china_combined.csv' 'gs://wqd7005-covid19-data'
# !gsutil cp 't_death_cases_china_combined.csv' 'gs://wqd7005-covid19-data'
# !gsutil cp 't_recovered_cases_china_combined.csv' 'gs://wqd7005-covid19-data'


# In[ ]:




