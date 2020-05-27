#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#---
# WDQ7005 - Data Mining
# Master of Data Science | University of Malaya
# Assignemnt: Part D (Milestone 4) - Interpretation and Communicating Data Insights 
#             - leveraging on the Group Assignment COVID-19 datasets
#
# Student Name: Kok Hon Loong (WQD170086)
# Date: 19th May 2020
#---


# In[1]:


# Importing relevant library packages usage for the assignment

# --- setup ---
import requests
import gc
import os
import random
import sys

import pandas as pd
pd.set_option('max_columns', 50)

from tqdm.notebook import tqdm
import numpy as np
import scipy as sp
import math
import time
from datetime import datetime
import operator

from IPython.core.display import display, HTML

# Importing the libraries packages for plotting graph and charts
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.colors as mcolors

import seaborn as sns
plt.style.use('seaborn')

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Importing library packages for web scrapping and Machine Learning

# Web crawling library - BeautifulSoup
from bs4 import BeautifulSoup

# Defining the Text style to be printed
class txtStyle:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# --- models ---
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold

import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.simplefilter('ignore')

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# In[2]:


# Crawl for near real-time datasets to be use for this assignment and store it into dataframe respectively

# Configuring the website to acquire the datasets
def switcher(x):
    return {
        0: "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
        1: "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
        2: "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",
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

# Put the tabulated data into a dataframe :
confirmed_cases = df_confirmed[0]
death_cases = df_deaths[0]
recovered_cases = df_recovered[0]


# In[3]:


# Display Confirmed Cases dataset acquired
confirmed_cases


# In[4]:


# Display Death Cases dataset acquired
death_cases


# In[5]:


# Display Recovered Cases dataset acquired
recovered_cases


# In[6]:


# Converting the date type to string for all datasets
def _convert_date_str(df):
    try:
        df.columns = list(df.columns[:5]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in df.columns[5:]]
    except:
        print('_convert_date_str failed with %y, try %Y')
        df.columns = list(df.columns[:5]) + [datetime.strptime(d, "%m/%d/%Y").date().strftime("%Y-%m-%d") for d in df.columns[5:]]

_convert_date_str(confirmed_cases)
_convert_date_str(death_cases)
_convert_date_str(recovered_cases)


# In[7]:


# Rename column name and filter out problematic data points (i.e. NULL value)
# removed_states = "Recovered|Grand Princess|Diamond Princess"
# removed_countries = "US|The West Bank and Gaza"

confirmed_cases.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
death_cases.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
recovered_cases.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

# confirmed_cases = confirmed_cases[~confirmed_cases["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
# death_cases    = death_cases[~death_cases["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
# recovered_cases = recovered_cases[~recovered_cases["Province_State"].replace(np.nan, "nan").str.match(removed_states)]

# confirmed_cases = confirmed_cases[~confirmed_cases["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]
# death_cases    = death_cases[~death_cases["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]
# recovered_cases = recovered_cases[~recovered_cases["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]


# In[8]:


# Clean the datasets by melting all the dates data to sum it into total cases with a new field at the end of the column by country and then store it in the respective melt dataframe
confirmed_globalMelt = confirmed_cases.melt(
    id_vars=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_cases.columns[5:], var_name='Date', value_name='ConfirmedCases')
death_globalMelt = death_cases.melt(
    id_vars=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=death_cases.columns[5:], var_name='Date', value_name='DeathCases')
recovered_globalMelt = recovered_cases.melt(
    id_vars=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=recovered_cases.columns[5:], var_name='Date', value_name='RecoveredCases')


# In[ ]:


#---
#  Business Intelligence (BI): to describe and diagnose the current COVID-19 situation worldwide
#---


# In[9]:


# Combined the Confirmed, Death and Recovered cases dataframe into into one 'train' dataframe
train = confirmed_globalMelt.merge(death_globalMelt, on=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])
train = train.merge(recovered_globalMelt, on=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])
train


# In[10]:


# Rename the column name to lowercases
train.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'DeathCases': 'fatalities', 'RecoveredCases': 'recovered'}, axis=1, inplace=True)
train['country_province'] = train['country'].fillna('') + '/' + train['province'].fillna('')

# Group the records by date for confirmed and fatalities cases
ww_df = train.groupby('date')[['confirmed', 'fatalities']].sum().reset_index()

# Add a new column for new_case by having confirmed cases minus yesterday confirmed cases daily
ww_df['newCase'] = ww_df['confirmed'] - ww_df['confirmed'].shift(1)
ww_df.head()


# In[11]:


# Pivoting the dataframe by categorizing the variable type
ww_melt_df = pd.melt(ww_df, id_vars=['date'], value_vars=['confirmed', 'fatalities', 'newCase'])
ww_melt_df


# In[12]:


# Plotting a graph to illustrate the Confirmed, Death and New Daily reported cases over time 
fig = px.line(ww_melt_df, x="date", y="value", color='variable', 
              title="Worldwide COVID-19 Confirmed, Fatalities and New Daily Cases Over Time")
fig.show()


# In[13]:


# Plotting a graph to illustrating the rate of the growth reported cases using the log-scale
fig = px.line(ww_melt_df, x="date", y="value", color='variable',
              title="Worldwide COVID-19 Confirmed, Fatalities and New Daily Cases Over Time (Log scale)",
             log_y=True)
fig.show()


# In[14]:


# Plotting a graph to view on the Mortality rate over time
ww_df['mortality'] = ww_df['fatalities'] / ww_df['confirmed']

fig = px.line(ww_df, x="date", y="mortality", 
              title="Worldwide COVID-19 Mortality Rate Over Time (Death Cases /Confirmed Cases )")
fig.show()


# In[15]:


# To analyze on the County COVID-19 growth Confirmed and Fatalities cases
country_df = train.groupby(['date', 'country'])[['confirmed', 'fatalities', 'recovered']].sum().reset_index()
country_df.tail(20)


# In[16]:


# To list down the countries in the dataset
countries = country_df['country'].unique()
print(f'{len(countries)} countries are in dataset:\n{countries}')


# In[17]:


# To set a dataframe comprises of the ASEAN countries in the dataset

# Get data for ASEAN regions:
asean = ['Malaysia', 'Singapore', 'Thailand', 'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Myanmar', 'Philippines', 'Vietnam']

countriesSEA = asean
print(f'{len(countriesSEA)} countries are in dataset:\n{countriesSEA}')


# In[18]:


# Identify the last date capture in the dataset
target_date = country_df['date'].max()
print('Date: ', target_date)

# Count the number of countries with cases falling into the 5 categories
for i in [1, 10, 100, 1000, 10000]:
    n_countries = len(country_df.query('(date == @target_date) & confirmed > @i'))
    print(f'{n_countries} countries have more than {i} confirmed COVID-19 cases')


# In[19]:


print('Date: ', target_date)
# Count the number of ASEAN countries with cases falling into the 5 categories
for i in [1, 10, 100, 1000, 10000]:
    n_countriesSEA = len(country_df.query('(date == @target_date) & (country == @countriesSEA) & confirmed > @i'))
    print(f'{n_countriesSEA} ASEAN countries have more than {i} confirmed COVID-19 cases')


# In[20]:


# Count the number of countries with cases falling into the 5 categories and illustrate with a graph representation
ax = sns.distplot(np.log10(country_df.query('date == @target_date')['confirmed'] + 1))
ax.set_xlim([0, 6])
ax.set_xticks(np.arange(7))
_ = ax.set_xticklabels(['0', '10', '100', '1k', '10k', '100k'])


# In[21]:


# To view in a chart which are the top countries of confirmed and fatalities COVID-19 cases
top_country_df = country_df.query('(date == @target_date) & (confirmed > 1000)').sort_values('confirmed', ascending=False)
top_country_melt_df = pd.melt(top_country_df, id_vars='country', value_vars=['confirmed', 'fatalities'])

fig = px.bar(top_country_melt_df.iloc[::-1],
             x='value', y='country', color='variable', barmode='group',
             title=f'Top Countries with accumulated Confirmed and Fatalities COVID-19 cases reported on {target_date}', text='value', height=2000, orientation='h')
fig.show()


# In[22]:


# To view in a chart which are the ASEAN countries of confirmed and fatalities COVID-19 cases
top_countrySEA_df = country_df.query('(date == @target_date) & (country == @countriesSEA) ').sort_values('confirmed', ascending=False)
top_countrySEA_melt_df = pd.melt(top_countrySEA_df, id_vars='country', value_vars=['confirmed', 'fatalities'])

fig = px.bar(top_countrySEA_melt_df.iloc[::-1],
             x='value', y='country', color='variable', barmode='group',
             title=f'ASEAN Countries with accumulated Confirmed and Fatalities COVID-19 cases reported on {target_date}', text='value', height=650, orientation='h')
fig.show()


# In[23]:


# Plotting the graph for Top 30 countries with COVID-19 Confirmed cases
top30_countries = top_country_df.sort_values('confirmed', ascending=False).iloc[:30]['country'].unique()
top30_countries_df = country_df[country_df['country'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='date', y='confirmed', color='country',
              title=f'Confirmed COVID-19 Cases for Top 30 countries as of {target_date}')
fig.show()


# In[24]:


# Plotting the graph for Top 30 countries with COVID-19 Fatilities cases
top30_countries = top_country_df.sort_values('fatalities', ascending=False).iloc[:30]['country'].unique()
top30_countries_df = country_df[country_df['country'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='date', y='fatalities', color='country',
              title=f'COVID-19 Fatalities Cases for Top 30 countries as of {target_date}')
fig.show()


# In[25]:


# Plotting a graph to illustrate the High Mortality rate by Country
top_country_df = country_df.query('(date == @target_date) & (confirmed > 100)')
top_country_df['mortality_rate'] = top_country_df['fatalities'] / top_country_df['confirmed']
top_country_df = top_country_df.sort_values('mortality_rate', ascending=False)

fig = px.bar(top_country_df[:30].iloc[::-1],
             x='mortality_rate', y='country',
             title=f'High Mortality Rate on COVID-19: Top 30 countries on {target_date} (Fatalities / Confirmed)', text='mortality_rate', height=800, orientation='h')
fig.show()


# In[26]:


# Plotting a graph to illustrate the Low Mortality rate by Country
fig = px.bar(top_country_df[-30:],
             x='mortality_rate', y='country',
             title=f'Low Mortality Rate on COVID-19: Top 30 countries on {target_date} (Fatalities / Confirmed)', text='mortality_rate', height=800, orientation='h')
fig.show()


# In[27]:


# Plotting the graph for ASEAN countries with COVID-19 Confirmed cases
topSEAc_countries = top_countrySEA_df.sort_values('confirmed', ascending=False).iloc[:]['country'].unique()
topSEAc_countries_df = country_df[country_df['country'].isin(topSEAc_countries)]
fig = px.line(topSEAc_countries_df,
              x='date', y='confirmed', color='country',
              title=f'Confirmed COVID-19 Cases for ASEAN countries as of {target_date}')
fig.show()


# In[28]:


# Plotting the graph for ASEAN countries with COVID-19 Recovered cases
topSEAc_countries = top_countrySEA_df.sort_values('confirmed', ascending=False).iloc[:]['country'].unique()
topSEAc_countries_df = country_df[country_df['country'].isin(topSEAc_countries)]
fig = px.line(topSEAc_countries_df,
              x='date', y='recovered', color='country',
              title=f'Recovered COVID-19 Cases for ASEAN countries as of {target_date}')
fig.show()


# In[29]:


# Plotting the graph for ASEAN countries with COVID-19 Fatilities cases
topSEAf_countries = top_countrySEA_df.sort_values('fatalities', ascending=False).iloc[:]['country'].unique()
topSEAf_countries_df = country_df[country_df['country'].isin(topSEAf_countries)]
fig = px.line(topSEAf_countries_df,
              x='date', y='fatalities', color='country',
              title=f'COVID-19 Fatalities Cases for ASEAN countries as of {target_date}')
fig.show()


# In[30]:


# Plotting a graph to illustrate the Mortality rate for ASEAN Country
countrySEA_df = country_df.query('(date == @target_date) & (country == @countriesSEA)')
countrySEA_df['mortality_rate'] = countrySEA_df['fatalities'] / countrySEA_df['confirmed']
countrySEA_df = countrySEA_df.sort_values('mortality_rate', ascending=False)

fig = px.bar(countrySEA_df[:].iloc[::-1],
             x='mortality_rate', y='country',
             title=f'Mortality Rate on COVID-19: ASEAN countries on {target_date} (Fatalities /  Confirmed)', text='mortality_rate', height=500, orientation='h')
fig.show()


# In[31]:


# To view the Worldwide Confirmed Cases on map
all_country_df = country_df.query('date == @target_date')
all_country_df['confirmed_log1p'] = np.log10(all_country_df['confirmed'] + 1)
all_country_df['fatalities_log1p'] = np.log10(all_country_df['fatalities'] + 1)
all_country_df['mortality_rate'] = all_country_df['fatalities'] / all_country_df['confirmed']

fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="confirmed_log1p", 
                    hover_name="country", hover_data=["confirmed", 'fatalities', 'mortality_rate'],
                    range_color=[all_country_df['confirmed_log1p'].min(), all_country_df['confirmed_log1p'].max()], 
                    color_continuous_scale="peach", 
                    title=f'Worldwide Countries with COVID-19 Confirmed Cases as of {target_date}')

# To update colorbar to show raw values, but this does not work somehow...
trace1 = list(fig.select_traces())[0]
trace1.colorbar = go.choropleth.ColorBar(
    tickvals=[0, 1, 2, 3, 4, 5],
    ticktext=['1', '10', '100', '1000','10000', '10000'])
fig.show()


# In[32]:


# To view the Worldwide Fatalities on map
fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="fatalities_log1p", 
                    hover_name="country", range_color=[0, 4],
                    hover_data=['confirmed', 'fatalities', 'mortality_rate'],
                    color_continuous_scale="peach", 
                    title=f'Worldwide Countries with COVID-19 Fatalities Cases as of {target_date}')
fig.show()


# In[33]:


# To view the Worldwide Mortality Rate on map
fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="mortality_rate", 
                    hover_name="country", range_color=[0, 0.12], 
                    color_continuous_scale="peach", 
                    title=f'Worldwide Countries with COVID-19 Mortality Rate as of {target_date}')
fig.show()


# In[34]:


# To plot the fatility growth for countries since 10 deaths
n_countries = 80
n_start_death = 10
fatality_top_countires = top_country_df.sort_values('fatalities', ascending=False).iloc[:n_countries]['country'].values
country_df['date'] = pd.to_datetime(country_df['date'])


df_list = []
for country in fatality_top_countires:
    this_country_df = country_df.query('country == @country')
    start_date = this_country_df.query('fatalities > @n_start_death')['date'].min()
    this_country_df = this_country_df.query('date >= @start_date')
    this_country_df['date_since'] = this_country_df['date'] - start_date
    this_country_df['fatalities_log1p'] = np.log10(this_country_df['fatalities'] + 1)
    this_country_df['fatalities_log1p'] -= this_country_df['fatalities_log1p'].values[0]
    df_list.append(this_country_df)

tmpdf = pd.concat(df_list)
tmpdf['date_since_days'] = tmpdf['date_since'] / pd.Timedelta('1 days')

fig = px.line(tmpdf,
              x='date_since_days', y='fatalities_log1p', color='country',
              title=f'COVID-19 Fatalities by Country since 10 Deaths, as of {target_date}')
fig.add_trace(go.Scatter(x=[0, 28], y=[0, 4], name='Double by 7 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[0, 56], y=[0, 4], name='Double by 14 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[0, 84], y=[0, 4], name='Double by 21 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.show()

# Observation: Sudden increase in China on the days 85, was due to the updated Death cases reported for Wuhan (Ref: https://www.bbc.com/news/world-asia-china-52321529)


# In[35]:


# Daily newly Confirmed Cases trend reported
country_df['prev_confirmed'] = country_df.groupby('country')['confirmed'].shift(1)
country_df['new_case'] = country_df['confirmed'] - country_df['prev_confirmed']
country_df['new_case'].fillna(0, inplace=True)
top30_country_df = country_df[country_df['country'].isin(top30_countries)]

fig = px.line(top30_country_df,
              x='date', y='new_case', color='country',
              title=f'Daily New Confirmed COVID-19 Cases for Top 30 Countries Worldwide as of {target_date}')
fig.show()


# In[36]:


# To illustrate Geographical Animation - Confirmed Cases spread over time
country_df['date'] = country_df['date'].apply(str)
country_df['confirmed_log1p'] = np.log1p(country_df['confirmed'])
country_df['fatalities_log1p'] = np.log1p(country_df['fatalities'])

fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="confirmed", size='confirmed', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['confirmed'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title=f'COVID-19: Worldwide Confirmed Cases Spread Over Time as of {target_date}', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# In[37]:


# To illustrate Geographical Animation - Fatalities Cases spread over time
fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="fatalities", size='fatalities', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['fatalities'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title=f'COVID-19: Worldwide Fatalities Growth Over Time as of {target_date}', color_continuous_scale="portland")
fig.show()


# In[38]:


# To illustrate Geographical Animation - Daily New Cases Trend
country_df.loc[country_df['new_case'] < 0, 'new_case'] = 0.
fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="new_case", size='new_case', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['new_case'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title=f'COVID-19: Daily New Cases Reported Over Time in Worldwide as of {target_date}', color_continuous_scale="portland")
fig.show()


# In[39]:


# To illustrate the condition of COVID-19 in Asia Pacific
top_asian_country_df = country_df[country_df['country'].isin(['China', 'Japan', 'Korea, South', 'Malaysia', 'Thailand', 'Philippines', 'Indonesia', 'Singapore', 'Vietnam', 'Australia', 'New Zealand', 'Taiwan*'])]

fig = px.line(top_asian_country_df,
              x='date', y='new_case', color='country',
              title=f'Daily New COVID-19 Confirmed Cases in Selected Pan-Asia Pacific Countries as of {target_date}')
fig.show()


# In[40]:


# To analyze COVID-19 pandemic situation in China - Daily New Confirmed Cases
china_df = train.query('country == "China"')
china_df['prev_confirmed'] = china_df.groupby('province')['confirmed'].shift(1)
china_df['new_case'] = china_df['confirmed'] - china_df['prev_confirmed']
china_df.loc[china_df['new_case'] < 0, 'new_case'] = 0.

fig = px.line(china_df,
              x='date', y='new_case', color='province',
              title=f'Daily New COVID-19 Confirmed Cases in China by Province as of {target_date}')
fig.show()


# In[ ]:


#---
#  Modelling: to forecast/predict COVID-19
#---


# In[41]:


# Preparing to split the dataset
covid19_ds = train
covid19_ds


# In[42]:


# NOTE: Must run the above data preparation above before running this cell each time.

# # Dropping the unwanted features and leave 1 unwanted feature to drop the data in x.
# covid19_ds = covid19_ds.drop('Unnamed: 0', axis=1)
# covid19_ds = covid19_ds.drop(['Lat','Long'], axis=1)
# covid19_ds = covid19_ds.drop('country_province', axis=1)

# # province is a label to predict province in y. Using the drop() function to take all other data in x.
# y=covid19_ds.province
# x=covid19_ds.drop('province',axis=1)

# # Split the acquired dataset with 80% for trainSet and 20% for testSet
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# covid19_trainSet = x_train
# covid19_testSet = x_test

covid19_trainSet = covid19_ds
covid19_testSet = covid19_ds


# In[ ]:


# # To view the new covid19_trainSet with 80% of the near real-time data acquired from web crawling
# covid19_trainSet


# In[ ]:


# # To view the new covid19_testSet with 20% of the near real-time data acquired from web crawling
# covid19_testSet


# In[ ]:


# # To sort and group the trainCC_df by country for confirmed cases
# trainCC_df = covid19_trainSet.fillna('NA').groupby(['country','date'])['confirmed'].sum() \
#                           .groupby(['country']).max().sort_values() \
#                           .groupby(['country']).sum().sort_values(ascending = False)

# top10_countries = pd.DataFrame(trainCC_df).head(10)
# top10_countries


# In[43]:


# To sort and group the trainCC_df by country for confirmed cases
trainCC_df = covid19_ds.fillna('NA').groupby(['country','date'])['confirmed'].sum()                           .groupby(['country']).max().sort_values()                           .groupby(['country']).sum().sort_values(ascending = False)

top10_countries = pd.DataFrame(trainCC_df).head(10)
top10_countries


# In[ ]:


# # To group the trainTS_df by date and country
# trainTS_df=covid19_trainSet.groupby(['date','country']).agg('sum').reset_index()
# trainTS_df


# In[45]:


# To group the trainTS_df by date and country
trainTS_df=covid19_ds.groupby(['date','country']).agg('sum').reset_index()
trainTS_df.tail(5)


# In[46]:


# Time evaluation - Confirmed Cases
def pltCountry_cases(confirmed,*argv):
    f, ax=plt.subplots(figsize=(16,5))
    labels=argv
    for a in argv: 
        countryTS=trainTS_df.loc[(trainTS_df['country']==a)]
        plt.plot(countryTS['date'],countryTS['confirmed'],linewidth=3)
        plt.xticks(rotation=40)
        plt.legend(labels)
        ax.set(title=f'Time evaluation of the number of COVID-19 confirmed cases as of {target_date}...' )
  


# In[47]:


pltCountry_cases('confirmed','Malaysia')


# In[48]:


# Time evaluation - Fatalities Cases
def pltCountry_fatalities(fatalities,*argv):
    f, ax=plt.subplots(figsize=(16,5))
    labels=argv
    for a in argv: 
        countryTS=trainTS_df.loc[(trainTS_df['country']==a)]
        plt.plot(countryTS['date'],countryTS['fatalities'],linewidth=3)
        plt.xticks(rotation=40)
        plt.legend(labels)
        ax.set(title=f'Time evalution of the number of COVID-19 fatalities cases as of {target_date}...' )


# In[49]:


pltCountry_fatalities('fatalities','Malaysia')


# In[50]:


# Comparing a group of countries with similar evolution: Brazil, Russia, India, China + Malaysia
pltCountry_cases('confirmed', 'Brazil','Russia','India','China', 'Malaysia')
pltCountry_fatalities('fatilities','Brazil','Russia','India','China', 'Malaysia')


# In[51]:


covid19_testSet['date'] = pd.to_datetime(covid19_testSet['date'])
covid19_trainSet['date'] = pd.to_datetime(covid19_trainSet['date'])


# In[52]:


# Countries comparison for Confirmed COVID-19 cases
case='confirmed'
def timeCompare(time,*argv):
    Coun1=argv[0]
    Coun2=argv[1]
    f,ax=plt.subplots(figsize=(16,5))
    labels=argv  
    countryTS=trainTS_df.loc[(trainTS_df['country']==Coun1)]
    plt.plot(countryTS['date'],countryTS[case],linewidth=2)
    plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Time Evaluation of actual COVID-19 cases',ylabel='Number of cases' )

    countryTS2=trainTS_df.loc[trainTS_df['country']==Coun2]
    #country2['Date']=country2['Date']-datetime.timedelta(days=time)
    plt.plot(countryTS2['date'],countryTS2[case],linewidth=2)
    # plt.xticks([])
    plt.legend(labels)
    ax.set(title=f' Time Evaluation of COVID-19 confirmed cases in %d days difference as of {target_date} '%time ,ylabel='Number of %s cases'%case )


# In[53]:


timeCompare(8,'Malaysia','Singapore')
timeCompare(6,'Malaysia','Indonesia')
timeCompare(7,'Malaysia','Philippines')
timeCompare(7,'Malaysia','US')


# In[54]:


timeCompare(8,'US','China')
timeCompare(6,'US','Russia')
timeCompare(7,'US','United Kingdom')
timeCompare(7,'US','India')


# In[55]:


# Auto-Regressive Integrated Moving Average (ARIMA) Model - is a class of statistical models for analyzing 
# and forecasting time series data. It explicitly caters to a suite of standard structures in time series data, 
# and as such provides a simple yet powerful method for making skillful time series forecasts.

# Cases distribution with rolling mean and standard deviation

sns.set(palette = 'Set1',style='darkgrid')

#Function for making a time serie on a designated country and plotting the rolled mean and standard 
def roll(country,case='confirmed'):
    ts=trainTS_df.loc[(trainTS_df['country']==country)]  
    ts=ts[['date',case]]
    ts=ts.set_index('date')
    ts.astype('int64')
    a=len(ts.loc[(ts['confirmed']>=10)])
    ts=ts[-a:]
    return (ts.rolling(window=4,center=False).mean().dropna())


def rollPlot(country, case='confirmed'):
    ts=trainTS_df.loc[(trainTS_df['country']==country)]  
    ts=ts[['date',case]]
    ts=ts.set_index('date')
    ts.astype('int64')
    a=len(ts.loc[(ts['confirmed']>=10)])
    ts=ts[-a:]
    plt.figure(figsize=(16,6))
    plt.plot(ts.rolling(window=7,center=False).mean().dropna(),label='Rolling Mean')
    plt.plot(ts[case])
    plt.plot(ts.rolling(window=7,center=False).std(),label='Rolling Standard')
    plt.legend()
    plt.title(f'COVID-19 confirmed cases distribution in %s with rolling mean and rolling standard as of {target_date}' %country)
    plt.xticks([])


# In[56]:


tsC1=roll('Malaysia')
rollPlot('Malaysia')


# In[57]:


tsC2=roll('Singapore')
rollPlot('Singapore')


# In[58]:


tsC3=roll('China')
rollPlot('China')


# In[59]:


tsC4=roll('Vietnam')
rollPlot('Vietnam')


# In[60]:


tsC5=roll('US')
rollPlot('US')


# In[61]:


tsC6=roll('United Kingdom')
rollPlot('United Kingdom')


# In[62]:


#Decomposing the ts to find its properties - Malaysia
print(txtStyle.BOLD + f'        TIME SERIES DECOMPOSITION RESULT FOR MALAYSIA AS OF {target_date}' + txtStyle.END)
fig=sm.tsa.seasonal_decompose(tsC1.values,freq=7).plot()


# In[63]:


#Decomposing the ts to find its properties - Singapore
print(txtStyle.BOLD + f'        TIME SERIES DECOMPOSITION RESULT FOR SINGAPORE AS OF {target_date}' + txtStyle.END)
fig=sm.tsa.seasonal_decompose(tsC2.values,freq=7).plot()


# In[64]:


#Decomposing the ts to find its properties - China
print(txtStyle.BOLD + f'        TIME SERIES DECOMPOSITION RESULT FOR CHINA AS OF {target_date}' + txtStyle.END)
fig=sm.tsa.seasonal_decompose(tsC3.values,freq=7).plot()


# In[65]:


#Decomposing the ts to find its properties - Vietnam
print(txtStyle.BOLD + f'        TIME SERIES DECOMPOSITION RESULT FOR VIETNAM AS OF {target_date}' + txtStyle.END)
fig=sm.tsa.seasonal_decompose(tsC4.values,freq=7).plot()


# In[66]:


#Decomposing the ts to find its properties - US
print(txtStyle.BOLD + f'          TIME SERIES DECOMPOSITION RESULT FOR US AS OF {target_date}' + txtStyle.END)
fig=sm.tsa.seasonal_decompose(tsC5.values,freq=7).plot()


# In[67]:


#Decomposing the ts to find its properties - United Kingdom
print(txtStyle.BOLD + f'        TIME SERIES DECOMPOSITION RESULT FOR UNITED KINGDOM AS OF {target_date}' + txtStyle.END)
fig=sm.tsa.seasonal_decompose(tsC6.values,freq=7).plot()


# In[68]:


# Function to check the stationarity of the time serie using Dickey fuller test
# Here the p-value helps us to reject the null hypothesis of the non-stationarity of the data with confidence, but
# that assumption is sometimes not enough, we should also consider the time series might not be an AR (auto-regression).
# This supposedly means that the parameter i(d) will be 0 and the model would be an ARMA model.

def stationarity(ts):
    print('Results of Dickey-Fuller Test:')
    test = adfuller(ts, autolag='AIC')
    results = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for i,val in test[4].items():
        results['Critical Value (%s)'%i] = val
    print (results)


# In[69]:


#For Malaysia
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'TIME SERIES STATIONARITY TEST FOR MALAYSIA AS OF {target_date}' + txtStyle.END)
tsC=tsC1['confirmed'].values
stationarity(tsC)


# In[70]:


#For Singapore
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'TIME SERIES STATIONARITY TEST FOR SINGAPORE AS OF {target_date}' + txtStyle.END)
tsC=tsC2['confirmed'].values
stationarity(tsC)


# In[71]:


#For China
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'TIME SERIES STATIONARITY TEST FOR CHINA AS OF {target_date}' + txtStyle.END)
tsC=tsC3['confirmed'].values
stationarity(tsC)


# In[72]:


#For Vietnam
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'TIME SERIES STATIONARITY TEST FOR VIETNAM AS OF {target_date}' + txtStyle.END)
tsC=tsC4['confirmed'].values
stationarity(tsC)


# In[73]:


#For US
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'TIME SERIES STATIONARITY TEST FOR US AS OF {target_date}' + txtStyle.END)
tsC=tsC5['confirmed'].values
stationarity(tsC)


# In[74]:


#For UK
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'TIME SERIES STATIONARITY TEST FOR UNITED KINGDOM AS OF {target_date}' + txtStyle.END)
tsC=tsC6['confirmed'].values
stationarity(tsC)


# In[75]:


# Auto Correlation Function (ACF) and Partial Auto Correclation Function (PACF)
# As the p-value helps us to reject the null hypothesis of the non-stationarity of the data with confidence, 
# but that assumption is sometimes not enough, we should also consider the time series might not be an AR (auto-regression).
# This supposedly means that the parameter i(d) will be 0 and the model would be an ARMA model.

def corr(ts):
    plot_acf(ts,lags=12,title="Auto Correlation Function (ACF)")
    plot_pacf(ts,lags=12,title="Partial Auto Correclation Function (PACF)")
    


# In[76]:


# For Malaysia
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'AUTO CORRELATION FUNCTION FOR MALAYSIA AS OF {target_date}' + txtStyle.END)
corr(tsC1)


# In[77]:


# For Singapore
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'AUTO CORRELATION FUNCTION FOR SINGAPORE AS OF {target_date}' + txtStyle.END)
corr(tsC2)


# In[78]:


# For China
print(txtStyle.BOLD + txtStyle.UNDERLINE + f'AUTO CORRELATION FUNCTION FOR CHINA AS OF {target_date}' + txtStyle.END)
corr(tsC3)


# In[79]:


#---
#  Interpretation and Evaluation: based on the factuals and diagnostics to forecast/predict COVID-19 confirmed cases 
#                                 for following week
#---


# In[80]:


# Building the model
covid19_trainSet = covid19_trainSet.set_index(['date'])
covid19_testSet = covid19_testSet.set_index(['date'])


# In[81]:


def create_features(trainTS_df,label=None):
    """
    Creates time series features from datetime index.
    """
    trainTS_df = trainTS_df.copy()
    trainTS_df['date'] = trainTS_df.index
#     trainTS_df['hour'] = trainTS_df['date'].dt.hour
#     trainTS_df['dayofweek'] = trainTS_df['date'].dt.dayofweek
#     trainTS_df['quarter'] = trainTS_df['date'].dt.quarter
#     trainTS_df['month'] = trainTS_df['date'].dt.month
#     trainTS_df['year'] = trainTS_df['date'].dt.year
#     trainTS_df['dayofyear'] = trainTS_df['date'].dt.dayofyear
#     trainTS_df['dayofmonth'] = trainTS_df['date'].dt.day
#     trainTS_df['weekofyear'] = trainTS_df['date'].dt.weekofyear
    
#     X = trainTS_df[['hour','dayofweek','quarter','month','year',
#            'dayofyear','dayofmonth','weekofyear']]
    X = trainTS_df[['date']]
   
    return X


# In[82]:


train_features=pd.DataFrame(create_features(covid19_trainSet))
test_features=pd.DataFrame(create_features(covid19_testSet))
features_and_target_train = pd.concat([covid19_trainSet,train_features], axis=1)
features_and_target_test = pd.concat([covid19_testSet,test_features], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def FunLabelEncoder(df):
    for c in trainTS_df.columns:
        if trainTS_df.dtypes[c] == object:
            le.fit(trainTS_df[c].astype(str))
            trainTS_df[c] = le.transform(trainTS_df[c].astype(str))
    return trainTS_df
features_and_target_train= FunLabelEncoder(features_and_target_train)


# In[83]:


# x_train= features_and_target_train[['country','month', 'dayofyear', 'dayofmonth' , 'weekofyear']]
x_train= features_and_target_train[['country','date']]
y1 = features_and_target_train[['confirmed']]
y2 =features_and_target_train[['fatalities']]
# x_test = features_and_target_test[['country', 'month', 'dayofyear', 'dayofmonth' , 'weekofyear']]
x_test = features_and_target_test[['country', 'date']]


# In[84]:


# Mean absolute percentage error for Confirmed Cases in Malaysia
def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100

def split(ts):
    #splitting 80%/20% because of little amount of data
    size = int(len(ts) * 0.80)
    covid19_trainSet = ts[:size]
    covid19_testSet = ts[size:]
    return(covid19_trainSet,covid19_testSet)


#Arima modeling for ts
def arima(ts,covid19_testSet):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(covid19_testSet))[0]
    
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(covid19_testSet, c='red',label='real values')
    plt.legend()
    plt.title('True vs Predicted Confirmed COVID-19 Cases in Malaysia for the coming week')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(covid19_testSet,pred))
    return (pred)



covid19_trainSet,covid19_testSet=split(tsC1)
pred=arima(covid19_trainSet,covid19_testSet)


# In[85]:


# Mean absolute percentage error for Confirmed Cases in Singapore
def mape(y2, y_pred): 
    y2, y_pred = np.array(y2), np.array(y_pred)
    return np.mean(np.abs((y2 - y_pred) / y2)) * 100

def split(ts):
    #splitting 80%/20% because of little amount of data
    size = int(len(ts) * 0.80)
    covid19_trainSet= ts[:size]
    covid19_testSet = ts[size:]
    return(covid19_trainSet,covid19_testSet)


#Arima modeling for ts
def arima(ts,covid19_testSet):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(covid19_testSet))[0]
    
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(covid19_testSet, c='red',label='real values')
    plt.legend()
    plt.title('True vs Predicted COVID-19 Confirmed Cases in Singapore for the coming week')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(covid19_testSet,pred))
    return (pred)



covid19_trainSet,covid19_testSet=split(tsC2)
pred=arima(covid19_trainSet,covid19_testSet)


# In[86]:


# Mean absolute percentage error for Confirmed Cases in China
def mape(y2, y_pred): 
    y2, y_pred = np.array(y2), np.array(y_pred)
    return np.mean(np.abs((y2 - y_pred) / y2)) * 100

def split(ts):
    #splitting 80%/20% because of little amount of data
    size = int(len(ts) * 0.80)
    covid19_trainSet= ts[:size]
    covid19_testSet = ts[size:]
    return(covid19_trainSet,covid19_testSet)


#Arima modeling for ts
def arima(ts,covid19_testSet):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(covid19_testSet))[0]
    
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(covid19_testSet, c='red',label='real values')
    plt.legend()
    plt.title('True vs Predicted COVID-19 Confirmed Cases in China for the coming week')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(covid19_testSet,pred))
    return (pred)



covid19_trainSet,covid19_testSet=split(tsC3)
pred=arima(covid19_trainSet,covid19_testSet)


# In[87]:


# Mean absolute percentage error for Confirmed Cases in Vietnam
def mape(y2, y_pred): 
    y2, y_pred = np.array(y2), np.array(y_pred)
    return np.mean(np.abs((y2 - y_pred) / y2)) * 100

def split(ts):
    #splitting 80%/20% because of little amount of data
    size = int(len(ts) * 0.80)
    covid19_trainSet= ts[:size]
    covid19_testSet = ts[size:]
    return(covid19_trainSet,covid19_testSet)


#Arima modeling for ts
def arima(ts,covid19_testSet):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(covid19_testSet))[0]
    
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(covid19_testSet, c='red',label='real values')
    plt.legend()
    plt.title('True vs Predicted COVID-19 Confirmed Cases in Vietnam for the coming week')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(covid19_testSet,pred))
    return (pred)



covid19_trainSet,covid19_testSet=split(tsC4)
pred=arima(covid19_trainSet,covid19_testSet)


# In[88]:


# Mean absolute percentage error for Confirmed Cases in US
def mape(y2, y_pred): 
    y2, y_pred = np.array(y2), np.array(y_pred)
    return np.mean(np.abs((y2 - y_pred) / y2)) * 100

def split(ts):
    #splitting 80%/20% because of little amount of data
    size = int(len(ts) * 0.80)
    covid19_trainSet= ts[:size]
    covid19_testSet = ts[size:]
    return(covid19_trainSet,covid19_testSet)


#Arima modeling for ts
def arima(ts,covid19_testSet):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(covid19_testSet))[0]
    
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(covid19_testSet, c='red',label='real values')
    plt.legend()
    plt.title('True vs Predicted COVID-19 Confirmed Cases in US for the coming week')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(covid19_testSet,pred))
    return (pred)



covid19_trainSet,covid19_testSet=split(tsC5)
pred=arima(covid19_trainSet,covid19_testSet)


# In[89]:


# Mean absolute percentage error for Confirmed Cases in United Kingdom
def mape(y2, y_pred): 
    y2, y_pred = np.array(y2), np.array(y_pred)
    return np.mean(np.abs((y2 - y_pred) / y2)) * 100

def split(ts):
    #splitting 80%/20% because of little amount of data
    size = int(len(ts) * 0.80)
    covid19_trainSet= ts[:size]
    covid19_testSet = ts[size:]
    return(covid19_trainSet,covid19_testSet)


#Arima modeling for ts
def arima(ts,covid19_testSet):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(covid19_testSet))[0]
    
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(covid19_testSet, c='red',label='real values')
    plt.legend()
    plt.title('True vs Predicted COVID-19 Confirmed Cases in United Kingdom for the coming week')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(covid19_testSet,pred))
    return (pred)



covid19_trainSet,covid19_testSet=split(tsC6)
pred=arima(covid19_trainSet,covid19_testSet)


# In[ ]:




