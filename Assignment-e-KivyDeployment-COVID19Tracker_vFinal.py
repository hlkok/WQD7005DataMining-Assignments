#!/usr/bin/env python
# coding: utf-8

# In[1]:


#---
# Name: Kok Hon Loong
# Matric ID: WQD170086
# Course: WQD7005 Data Mining
# Assignment: Part E (Milestone 5) - Deployment mobile application using Kivy on COVID-19 Tracker
#
# Acknowledgement: The Kivy code is guided to me by Mr. N. Sivaram (a YouTuber)
#---


# In[2]:


#---
# Importing required libraries
#---

#--->>>
# importing standard libraries for Python
import requests
import json
import gc
import os
import random
import sys

import pandas as pd
pd.set_option('max_columns', 50)

import numpy as np
import scipy as sp
import math
import time
from datetime import datetime, date
import operator

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

# Importing library packages for web scrapping and Machine Learning
# Web crawling library - BeautifulSoup
from bs4 import BeautifulSoup

#--->>>
# config - important to set the config to run Kivy not in full screen
from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')

# importing kivy libraries
import kivy
from kivy.app import App
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivymd.uix.snackbar import Snackbar
from kivy.core.window import Window

# This is to import libraries for user interfaces (uix) layouts design
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget

# This is to import libraries to be use with .kv widgets and design text file
from kivy.properties import ObjectProperty, NumericProperty, StringProperty

# This is to import the builder for the .kv file to load the correct function
from kivy.lang import Builder


# In[3]:


#--->>>
# Crawling for near real-time datasets to be use for this assignment and store it into dataframe respectively

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


# In[4]:


#--->>>
# To print the datasets descriptions acquired fresh from web crawling it
print(txtStyle.BOLD + f'Confirmed Cases Dataset:'+ txtStyle.END) 
print(confirmed_cases.info())
print('\n')

print(txtStyle.BOLD + f'Death Cases Dataset:'+ txtStyle.END) 
print(death_cases.info())
print('\n')

print(txtStyle.BOLD + f'Recovered Cases Dataset:'+ txtStyle.END) 
print(recovered_cases.info())
    


# In[5]:


#--->>>
# Data Preprocessing and Transformation


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

# Renaming the column names
confirmed_cases.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
death_cases.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
recovered_cases.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

# Clean the datasets by melting all the dates data to sum it into total cases with a new field at the end of the column by country and then store it in the respective melt dataframe
confirmed_globalMelt = confirmed_cases.melt(
    id_vars=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_cases.columns[5:], var_name='Date', value_name='ConfirmedCases')
death_globalMelt = death_cases.melt(
    id_vars=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=death_cases.columns[5:], var_name='Date', value_name='DeathCases')
recovered_globalMelt = recovered_cases.melt(
    id_vars=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=recovered_cases.columns[5:], var_name='Date', value_name='RecoveredCases')

# Combined the Confirmed, Death and Recovered cases dataframe into into one 'train' dataframe
combined_df = confirmed_globalMelt.merge(death_globalMelt, on=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])
combined_df = combined_df.merge(recovered_globalMelt, on=['Unnamed: 0', 'Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])

# Rename the column name to lowercases
combined_df.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'DeathCases': 'fatalities', 'RecoveredCases': 'recovered'}, axis=1, inplace=True)

# Drop unneeded columns from the dataframe for better interpretability:
combined_df = combined_df.drop('Unnamed: 0', axis=1)
combined_df = combined_df.drop(['Lat','Long'], axis=1)

print(txtStyle.BOLD + f'COVID-19 Cleaned Dataset:'+ txtStyle.END)
combined_df.info()


# In[6]:


#--->>>
# New cleaned and transformed dataframe - covid19_df
covid19_df = combined_df
covid19_df = covid19_df.groupby(['date', 'country'])[['confirmed', 'fatalities', 'recovered']].sum().reset_index()

covid19_df


# In[7]:


# To list down the countries in the dataset
countries = covid19_df['country'].unique()
print(f'{len(countries)} countries are in dataset:\n{countries}')


# In[8]:


# Identify the first date capture in the dataset
first_date = covid19_df['date'].min()
print('Date: ', first_date)


# In[9]:


# Identify the last date capture in the dataset
last_date = covid19_df['date'].max()
print('Date: ', last_date)


# In[10]:


country_SQ = ['China']
country_MQ = ['Malaysia', 'Singapore', 'Thailand', 'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Myanmar', 'Philippines', 'Vietnam']

countryQuery_df = covid19_df.query('(date == @last_date) & (country == @country_SQ) ').sort_values('confirmed', ascending=False)
countryQuery_df


# In[11]:



countryQuery_df = covid19_df.query('(date == @last_date) & (country == @country_MQ) ').sort_values('confirmed', ascending=False)
countryQuery_df


# In[ ]:





# In[12]:


#--->>> This section onwards are meant for Kivy application...


# In[13]:


class AboutUs(FloatLayout):
    pass


# In[14]:


class LiveCases(FloatLayout):
    search_input = ObjectProperty()
    location = StringProperty()
    confirmed = NumericProperty()
    deaths = NumericProperty()
    recovered = NumericProperty()
    today = date.today()
    
    def search(self):
        try:
            url = 'https://covid2019-api.herokuapp.com/v2/country/{}'
            result = requests.get(url=url.format(self.search_input.text)).json()
            self.location = result['data']['location']
            self.confirmed = result['data']['confirmed']
            self.deaths = result['data']['deaths']
            self.recovered = result['data']['recovered']
        except:
            Snackbar(text="Invalid country entered. Please try again...",font_size=15).show()


# In[15]:


class Covid19TrackerApp(MDApp):
    def __init__(self,**kwargs):
        self.theme_cls = ThemeManager()
        self.theme_cls.primary_palette = 'BlueGray'
        self.theme_cls.accent_palette = 'Green'
        self.theme_cls.theme_style = 'Light'
        Window.size = (400,600)
        super().__init__(**kwargs)


# In[ ]:


#---
# Run this application
#---

if __name__ == "__main__":
    Covid19TrackerApp().run()
    

