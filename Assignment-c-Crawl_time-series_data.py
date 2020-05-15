#!/usr/bin/env python
# coding: utf-8

# In[7]:


#--------------------------------------------
# CRAWL FOR COVID-19 TIME SERIES DATA ONLINE:
#--------------------------------------------
# Course: WQD7005 Data Mining | Master of Data Science @ University of Malaya
# Group Members: Azwa Kamaruddin (WQD170089), Kok Hon Loong (WQD170086)
# Assignment: Milestone 3 - Accessing and Processing of Data from Hadoop Data Warehouse or Data Lake using Python
# This code is to crawl for Covid-19 time series data on the number of confirmed cases, death cases and recovered cases globally.
# The raw data will be STORED into Google Cloud Storage data lake in CSV format.


# In[8]:


import requests
import pandas as pd
from bs4 import BeautifulSoup


# In[9]:


### =============================== ###
### NO. OF CONFIRMED CASES
### =============================== ###
def switcher(x):
    return {
        0: "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
        1: "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
        2: "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    }[x]

try:
    for i in range(3):
        page = requests.get(switcher(i), timeout=5)
        if page.status_code == 200:
            soup = BeautifulSoup(page.content,'html.parser')
            table = soup.find("table", {"class": "js-csv-data csv-data js-file-line-container"})
            if i == 0:
                df_confirmed = pd.read_html(str(table))
            elif i == 1:
                df_deaths = pd.read_html(str(table))
            else:
                df_recovered = pd.read_html(str(table))
        else:
            print(str(page.status_code) + " - Error, page not found.")
except requests.ConnectionError as e:
    print('Connection error')
    print(str(e))


# In[10]:


# Put the tabulated data into a dataframe :
confirmed_cases = df_confirmed[0]
death_cases = df_deaths[0]
recovered_cases = df_recovered[0]


# In[11]:


# Convert dataframe to csv file:
confirmed_cases.to_csv('confirmed_cases.csv', index=False, header=True)
death_cases.to_csv('death_cases.csv', index=False, header=True)
recovered_cases.to_csv('recovered_cases.csv', index=False, header=True)


# In[12]:


# Save csv file into Google Cloud Storage datalake:
get_ipython().system("gsutil cp 'confirmed_cases.csv' 'gs://wqd7005dm-covid19-ds'")
get_ipython().system("gsutil cp 'death_cases.csv' 'gs://wqd7005dm-covid19-ds'")
get_ipython().system("gsutil cp 'recovered_cases.csv' 'gs://wqd7005dm-covid19-ds'")

