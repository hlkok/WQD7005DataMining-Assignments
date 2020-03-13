#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# WDQ7005 - Data Mining
# Master of Data Science | University of Malaya
# Assignment Part A: Web Crawling of Real-time Data

# Group Members:
# Azwa Kamaruddin (WQD170089)
# Kok Hon Loong (WQD170086)


# In[1]:


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
from bs4 import BeautifulSoup


# In[ ]:


### =============================== ###
### CODE STRUCTURE
### =============================== ###
#
# We partitioned the code into 3 sections:
# 1. Number of CONFIRMED cases.
# 2. Number of DEATH cases.
# 3. Number of RECOVERED cases.
# For each section, we display the time series trend for ASEAN countries and China and compare between them.


# In[2]:


### =============================== ###
### NO. OF CONFIRMED CASES
### =============================== ###
url_confirmed_cases = "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"

try:
    page = requests.get(url_confirmed_cases, timeout=5)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content,'html.parser')
        table = soup.find("table", {"class": "js-csv-data csv-data js-file-line-container"})
        df = pd.read_html(str(table))
    else: 
        print(str(page.status_code) + " - Error, page not found.")
except requests.ConnectionError as e:
    print('Connection error')
    print(str(e))


# In[3]:


# Put the tabulated data into a dataframe and display the first 5 results:
confirmed_cases = df[0]
confirmed_cases.head()


# In[4]:


# Drop unneeded columns from the dataframe:
confirmed_cases = confirmed_cases.drop('Unnamed: 0', axis=1)
confirmed_cases = confirmed_cases.drop(['Lat','Long'], axis=1)
confirmed_cases = confirmed_cases.drop('Province/State', axis=1)

# Rename column header to simply 'Country' and set it as the index:
confirmed_cases = confirmed_cases.rename(columns={'Country/Region':'Country'})
confirmed_cases = confirmed_cases.set_index('Country')

# Display first 5 results:
confirmed_cases.head()


# In[5]:


# COMMENT: From the above table we can see that the number of confirmed cases are increasing for all listed countries.


# In[6]:


# Let's transpose the table and describe the data for each of the different countries:
confirmed_cases.transpose().describe()


# In[7]:


# COMMENT:
# count - shows how many days of data cases are being tracked. There are 51 days since the first tracking.
# mean - shows the average number of confirmed cases during the 51 days.
# max - the total number of confirmed cases to date.


# In[8]:


# Obtain only the confirmed cases for the 10 ASEAN countries + China

# Get data for ASEAN regions:
asean = ['Malaysia', 'Singapore', 'Thailand', 'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Myanmar', 'Philippines', 'Vietnam']
confirmed_cases_asean = confirmed_cases[confirmed_cases.index.isin(asean)]
confirmed_cases_asean


# In[9]:


# China:
confirmed_cases_china = confirmed_cases[confirmed_cases.index == 'China']
confirmed_cases_china.head()


# In[10]:


# For China, we combined all the different regions into a single row representing the entire China:
confirmed_cases_china_combined = confirmed_cases_china.groupby('Country').sum()
confirmed_cases_china_combined


# In[11]:


# Plot the number of confirmed cases over time for all ASEAN countries:

confirmed_cases_asean.T.plot()
plt.ylabel('No. of confirmed cases')
plt.xlabel('Days')
plt.show()


# In[12]:


# COMMENT: Singapore has the most number of recorded confirmed cases and are increasing exponentially.
# Malaysia is second followed by Thailand as the 3rd most recorded confirmed cases in ASEAN.


# In[13]:


# Plot the number of confirmed cases over time for China:

confirmed_cases_china_combined.T.plot()
plt.ylabel('No. of confirmed cases')
plt.xlabel('Days')
plt.show()


# In[14]:


# COMMENT: Confirmed cases in China continues to rise. More than 80k cases confirmed to date.


# In[15]:


# Plot the number of confirmed cases over time for China vs ASEAN countries:

ax = confirmed_cases_china_combined.T.plot()
confirmed_cases_asean.T.plot(ax=ax)


# In[16]:


# COMMENT: China numbers are too huge to compare with the number of cases in ASEAN.


# In[17]:


### =============================== ###
### NO. OF DEATH CASES
### =============================== ###
url_death_cases = "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"

try:
    page = requests.get(url_death_cases, timeout=5)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content,'html.parser')
        table = soup.find("table", {"class": "js-csv-data csv-data js-file-line-container"})
        df = pd.read_html(str(table))
    else: 
        print(str(page.status_code) + " - Error, page not found.")
except requests.ConnectionError as e:
    print('Connection error')
    print(str(e))


# In[18]:


# Put the tabulated data into a dataframe and display the first 5 results:
death_cases = df[0]
death_cases.head()


# In[19]:


# Drop unneeded columns from the dataframe:
death_cases = death_cases.drop('Unnamed: 0', axis=1)
death_cases = death_cases.drop(['Lat','Long'], axis=1)
death_cases = death_cases.drop('Province/State', axis=1)

# Rename column header to simply 'Country' and set it as the index:
death_cases = death_cases.rename(columns={'Country/Region':'Country'})
death_cases = death_cases.set_index('Country')

# Display first 5 results:
death_cases.head()


# In[20]:


# Transpose the table and describe the data for each of the different countries:
death_cases.transpose().describe()


# In[21]:


# COMMENT:
# count - shows how many days of data cases are being tracked. There are 51 days since the first tracking.
# mean - shows the average number of death cases during the 51 days.
# max - the total number of death cases to date.


# In[22]:


# Obtain only the death cases for the 10 ASEAN countries + China

# Get data for ASEAN regions:
asean = ['Malaysia', 'Singapore', 'Thailand', 'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Myanmar', 'Philippines', 'Vietnam']
death_cases_asean = death_cases[death_cases.index.isin(asean)]
death_cases_asean


# In[23]:


# China:
death_cases_china = death_cases[death_cases.index == 'China']
death_cases_china.head()


# In[24]:


# For China, we combined all the different regions into a single row representing the entire China:
death_cases_china_combined = death_cases_china.groupby('Country').sum()
death_cases_china_combined


# In[25]:


# Plot the number of death cases over time for all ASEAN countries:

death_cases_asean.T.plot()
plt.ylabel('No. of death cases')
plt.xlabel('Days')
plt.show()


# In[26]:


# COMMENT: The number of deaths from the covid-19 is rare/low in the ASEAN region.
# Only 4 patients have died: 2 from Phillipines, 1 from Thailand, and 1 from Indonesia.


# In[27]:


# Plot the number of death cases over time for China:

death_cases_china_combined.T.plot()
plt.ylabel('No. of confirmed cases')
plt.xlabel('Days')
plt.show()


# In[28]:


# COMMENT: The number of death cases in China has risen to more than 3k cases.


# In[29]:


# Plot the number of death cases over time for China vs ASEAN countries:

ax = death_cases_china_combined.T.plot()
death_cases_asean.T.plot(ax=ax)


# In[30]:


# COMMENT: Deaths in ASEAN countries due to covid-19 is rare or low compared to China.


# In[31]:


### =============================== ###
### NO. OF RECOVERED CASES
### =============================== ###
url_recovered_cases = "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"

try:
    page = requests.get(url_recovered_cases, timeout=5)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content,'html.parser')
        table = soup.find("table", {"class": "js-csv-data csv-data js-file-line-container"})
        df = pd.read_html(str(table))
    else: 
        print(str(page.status_code) + " - Error, page not found.")
except requests.ConnectionError as e:
    print('Connection error')
    print(str(e))


# In[32]:


# Put the tabulated data into a dataframe and display the first 5 results:
recovered_cases = df[0]
recovered_cases.head()


# In[33]:


# Drop unneeded columns from the dataframe:
recovered_cases = recovered_cases.drop('Unnamed: 0', axis=1)
recovered_cases = recovered_cases.drop(['Lat','Long'], axis=1)
recovered_cases = recovered_cases.drop('Province/State', axis=1)


# In[34]:


# Rename column header to simply 'Country' and set it as the index:
recovered_cases = recovered_cases.rename(columns={'Country/Region':'Country'})
recovered_cases = recovered_cases.set_index('Country')


# In[35]:


# Display first 5 results:
recovered_cases.head()


# In[36]:


# Transpose the table and describe the data for each of the different countries:
recovered_cases.transpose().describe()


# In[37]:


# COMMENT:
# count - shows how many days of data cases are being tracked. There are 51 days since the first tracking.
# mean - shows the average number of recovered cases during the 51 days.
# max - the total number of recovered cases to date.


# In[38]:


# Obtain only the recovered cases for the 10 ASEAN countries + China

# Get data for ASEAN regions:
asean = ['Malaysia', 'Singapore', 'Thailand', 'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Myanmar', 'Philippines', 'Vietnam']
recovered_cases_asean = recovered_cases[recovered_cases.index.isin(asean)]
recovered_cases_asean


# In[39]:


# China:
recovered_cases_china = recovered_cases[recovered_cases.index == 'China']
recovered_cases_china.head()


# In[40]:


# For China, we combined all the different regions into a single row representing the entire China:
recovered_cases_china_combined = recovered_cases_china.groupby('Country').sum()
recovered_cases_china_combined


# In[41]:


# Plot the number of recovered cases over time for all ASEAN countries:

recovered_cases_asean.T.plot()
plt.ylabel('No. of recovered cases')
plt.xlabel('Days')
plt.show()


# In[42]:


# COMMENT: Singapore continues to lead in terms of the number of patients who recovered from covid-19.
# Thailand is second in recovery, while Malaysia is in 3rd place.


# In[43]:


# Plot the number of recovered cases over time for China:

recovered_cases_china_combined.T.plot()
plt.ylabel('No. of confirmed cases')
plt.xlabel('Days')
plt.show()


# In[44]:


# COMMENT: Despite the huge number of confirmed cases at 80k, the number of patients who recovered is also increasing rapidly at 60k people.


# In[45]:


# Plot the number of recovered cases over time for China vs ASEAN countries:

ax = recovered_cases_china_combined.T.plot()
recovered_cases_asean.T.plot(ax=ax)


# In[46]:


# COMMENT: China recovery is increasing since they have the most case, while ASEAN countries are seen recovering as well but ASEAN cases are lower compared to China.


# In[47]:


### =============================== ###
### COMPARISON OF DEATH vs RECOVERED
### =============================== ###

# For ASEAN countries:

ax = death_cases_asean.plot(kind='hist')
recovered_cases_asean.plot(kind='hist', ax=ax)


# In[48]:


recovered_cases_asean


# In[49]:


death_cases_asean.transpose()[-1:].plot(kind='hist')


# In[50]:


recovered_cases_asean.transpose()[-1:].unstack().plot(kind='hist')


# In[ ]:




