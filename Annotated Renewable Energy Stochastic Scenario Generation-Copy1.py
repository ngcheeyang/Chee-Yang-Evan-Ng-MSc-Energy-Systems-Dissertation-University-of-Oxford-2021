#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all the packages needed 
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d
#import the package to generate a random number, note that seed is use to set a series of random number that can be reused
#ie for rooftop solar and utility solar to use the same sequenece of random number
#because if the sun is shinny, and if we have good rooftop solar production, utility solar production should also be good 
from random import random, seed 
#this is the package needed to generate an empirical distribution function
from statsmodels.distributions.empirical_distribution import ECDF

#pd.set_option('display.max_rows', None)


# In[2]:


#here we set the numbers of scenario that will be generated for winter, summer, and other (autumn and spring) 
#nfor example, the number 247 is obtained by deviding 90 days of winter by 365 days in a year; multiply by 1000 scenario
#this ensure the stochastic scenarios we generated follow the proportion of seasonal distribution of supply/demand
winter_probabilities = 247
summer_probabilities = 252
other_probabilities  = 501


# In[1]:


#this section set the if statement use to classify each day of raw data obtained from renewable.ninja (NASA's data) into seasons

def season_setting(row):
    day = row.dayofyear
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    
    if day in spring:
        season = 'other'
    elif day in summer:
        season = 'summer'
    elif day in fall:
        season = 'other'
    else:
        season = 'winter'
    return season
hours = range(0,24)


# In[24]:


#seed 1, is the sequence of random number used for all solar random number generation
seed(1)

pathName = os.getcwd()
fileNames = os.listdir(pathName)
numFiles = []
#this is to import/read all 29 files with the ending RS.csv; for each node
for fileNames in fileNames:
    if fileNames.endswith("RS.csv"):
        numFiles.append(fileNames)
print(numFiles)

#a stochastic number is generated for each node, at each hour, of each scenario
hours = range(0,24)

RS_CF = pd.DataFrame()

for f, file in enumerate(numFiles):
    #this is just to clean the raw file gotten from renewable ninja; and only consider the column we care about

    file_name = 'Node ' + str(f)
    
    t = pd.read_csv(file, skiprows = 3, usecols = ['local_time','electricity'],                    parse_dates = ['local_time'],                    infer_datetime_format = True, dayfirst = True)
    #assigning a season to each day (each with 24 hours) within the raw data obtained
    t['season'] = t.apply(season_setting, axis = 1)
    winter_days = t[t['season'] == 'winter']
    summer_days = t[t['season'] == 'summer']
    other_days  = t[t['season'] == 'other']
    seasons = [winter_days, summer_days, other_days]


    for n, season in enumerate(seasons):
        if(n == 0):
            season_name = 'winter'
        elif(n == 1):
            season_name = 'summer'
        else:
            season_name = 'other'
            
            #create a random number for each hour, at each node, within a scenario
        for hour in hours:
            winter_random_probabilities = [random() for _ in range(winter_probabilities)]
            summer_random_probabilities = [random() for _ in range(summer_probabilities)]
            other_random_probabilities  = [random() for _ in range(other_probabilities) ]
            season_probabilities = [winter_random_probabilities, summer_random_probabilities, other_random_probabilities]
            
            #creating temperory data frame for each output, so that it can all be put together 
            temp_df = pd.DataFrame()
            #use the data from the 'electricity' col in each raw data file (how much electricity is generated for a 1kW plant)
            #to create an empirical cummulative distribution function - for each location in that season, at that hour
            hourly_data = season[season.local_time.dt.time == dt.time(hour)]['electricity']
            ecdf = ECDF(hourly_data)
            x = np.linspace(min(hourly_data), max(hourly_data), 1000)
            slope_changes = sorted(set(x))
            #0 slope change means all values are 0 for solar irradiance (because it is night time)
            #temp_df = pd.DataFrame()
            if(len(slope_changes) == 1):
                for k, random_value in enumerate(season_probabilities[n]):
                    capacity_factor = 0
                    row = {'RS CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                    temp_df = temp_df.append(row, ignore_index = True)
                    
            #create a random number (between 0 and 1); this number is ten correspnded to the probability value that
            #is plugged in to find the corresponding value of solar energy production at that instance; creating
            #stochastic number of solar energy production at that hour that internalize the distribution of solar raw data
            else:
                temp_df = pd.DataFrame()
                sample_edf_values_at_slope_changes = [ ecdf(item) for item in slope_changes]
                inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes, fill_value='extrapolate')
                for k, random_value in enumerate(season_probabilities[n]):
                    capacity_factor = float(inverted_edf(random_value))
                    if(capacity_factor < 0):
                        capacity_factor = 0
                    row = {'RS CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                    temp_df = temp_df.append(row, ignore_index = True)
            
            RS_CF = pd.concat([RS_CF,temp_df], axis = 1).sort_index()
RS_CF


# In[25]:


RS_CF.to_excel('Rooftop Solar Stochastic Output.xlsx')


# In[26]:


seed(1)

US_CF = pd.DataFrame()

pathName = os.getcwd()
fileNames = os.listdir(pathName)
numFiles = []
for fileNames in fileNames:
    if fileNames.endswith("US.csv"):
        numFiles.append(fileNames)
print(numFiles)

for f, file in enumerate(numFiles):
    
    file_name = 'Node ' + str(f)
    
    t = pd.read_csv(file, skiprows = 3, usecols = ['local_time','electricity'],                    parse_dates = ['local_time'],                    infer_datetime_format = True, dayfirst = True)
    
    t['season'] = t.apply(season_setting, axis = 1)
    winter_days = t[t['season'] == 'winter']
    summer_days = t[t['season'] == 'summer']
    other_days  = t[t['season'] == 'other']
    seasons = [winter_days, summer_days, other_days]

    for n, season in enumerate(seasons):
        if(n == 0):
            season_name = 'winter'
        elif(n == 1):
            season_name = 'summer'
        else:
            season_name = 'other'
        
        for hour in hours:
            winter_random_probabilities = [random() for _ in range(winter_probabilities)]
            summer_random_probabilities = [random() for _ in range(summer_probabilities)]
            other_random_probabilities  = [random() for _ in range(other_probabilities) ]
            season_probabilities = [winter_random_probabilities, summer_random_probabilities, other_random_probabilities]
            
          
            hourly_data = season[season.local_time.dt.time == dt.time(hour)]['electricity']
            ecdf = ECDF(hourly_data)
            x = np.linspace(min(hourly_data), max(hourly_data), 1000)
            slope_changes = sorted(set(x))
            #0 slope change means all values are 0 for solar irradiance (because it is night time)
            temp_df = pd.DataFrame()
            if(len(slope_changes) == 1):
                for k, random_value in enumerate(season_probabilities[n]):
                    capacity_factor = 0
                    row = {'US CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                    temp_df = temp_df.append(row, ignore_index = True)
            
            else:
                sample_edf_values_at_slope_changes = [ ecdf(item) for item in slope_changes]
                inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes, fill_value='extrapolate')
                for k, random_value in enumerate(season_probabilities[n]):
                    capacity_factor = float(inverted_edf(random_value))
                    if(capacity_factor < 0):
                        capacity_factor = 0
                    row = {'US CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                    temp_df = temp_df.append(row, ignore_index = True)
            US_CF = pd.concat([US_CF,temp_df], axis = 1).sort_index()
US_CF


# In[27]:


US_CF.to_excel('Utility Scale Solar Stochastic Output.xlsx')


# In[12]:


seed(2)

ON_CF = pd.DataFrame()

#pathName = os.getcwd()
#fileNames = os.listdir(pathName)
#numFiles = []
#for fileNames in fileNames:
#    if fileNames.endswith("ON.csv"):
#        numFiles.append(fileNames)
#print(numFiles)

numFiles = ['Node1ON.csv']
for f, file in enumerate(numFiles):
    
    file_name = 'Node ' + str(f)
    print('File ' + file_name + 'started')
    t = pd.read_csv(file, skiprows = 3, usecols = ['local_time','electricity'],                    parse_dates = ['local_time'],                    infer_datetime_format = True, dayfirst = True)
    
    t['season'] = t.apply(season_setting, axis = 1)
    winter_days = t[t['season'] == 'winter']
    summer_days = t[t['season'] == 'summer']
    other_days  = t[t['season'] == 'other']
    seasons = [winter_days, summer_days, other_days]
    
    num = 0

    for n, season in enumerate(seasons):
        if(n == 0):
            season_name = 'winter'
        elif(n == 1):
            season_name = 'summer'
        else:
            season_name = 'other'
        

        for hour in [0]:
            
            winter_random_probabilities = [random() for _ in range(winter_probabilities)]
            summer_random_probabilities = [random() for _ in range(summer_probabilities)]
            other_random_probabilities  = [random() for _ in range(other_probabilities) ]
            season_probabilities = [winter_random_probabilities, summer_random_probabilities, other_random_probabilities]
            
            hourly_data = season[season.local_time.dt.time == dt.time(hour)]['electricity']
            try:
                ecdf = ECDF(hourly_data)
                x = np.linspace(min(hourly_data), max(hourly_data), 1000)
                y = ecdf(x)
                df_to_export = pd.DataFrame({'x' : x, 'y' : y})
                df_to_export.to_csv(str(num) + 'node ecdf.csv')
                num += 1
                print(num)
                plt.plot(x, y)
                slope_changes = sorted(set(x))
                #0 slope change means all values are 0 for solar irradiance (because it is night time)
                temp_df = pd.DataFrame()
                if(len(slope_changes) == 1):
                    for k, random_value in enumerate(season_probabilities[n]):
                        capacity_factor = 0
                        row = {'ON CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                        temp_df = temp_df.append(row, ignore_index = True)
                
                else:
                    sample_edf_values_at_slope_changes = [ ecdf(item) for item in slope_changes]
                    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes, fill_value='extrapolate')
                    for k, random_value in enumerate(season_probabilities[n]):
                        capacity_factor = float(inverted_edf(random_value))
                        if(capacity_factor < 0):
                            capacity_factor = 0
                        row = {'ON CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                        temp_df = temp_df.append(row, ignore_index = True)
                ON_CF = pd.concat([ON_CF,temp_df], axis = 1).sort_index()
            except:
                print(hourly_data)
#ON_CF


# In[33]:


ON_CF.to_excel('Onshore Wind Stochastic Output New N2.xlsx')


# In[34]:


seed(3)

pathName = os.getcwd()
fileNames = os.listdir(pathName)
numFiles = []
for fileNames in fileNames:
    if fileNames.endswith("OF.csv"):
        numFiles.append(fileNames)
print(numFiles)

hours = range(0,24)

OF_CF = pd.DataFrame()

for f, file in enumerate(numFiles):

    file_name = 'Node ' + str(f)
    
    t = pd.read_csv(file, skiprows = 3, usecols = ['local_time','electricity'],                    parse_dates = ['local_time'],                    infer_datetime_format = True, dayfirst = True)
    
    t['season'] = t.apply(season_setting, axis = 1)
    winter_days = t[t['season'] == 'winter']
    summer_days = t[t['season'] == 'summer']
    other_days  = t[t['season'] == 'other']
    seasons = [winter_days, summer_days, other_days]
    


    for n, season in enumerate(seasons):
        if(n == 0):
            season_name = 'winter'
        elif(n == 1):
            season_name = 'summer'
        else:
            season_name = 'other'
        for hour in hours:
            winter_random_probabilities = [random() for _ in range(winter_probabilities)]
            summer_random_probabilities = [random() for _ in range(summer_probabilities)]
            other_random_probabilities  = [random() for _ in range(other_probabilities) ]
            season_probabilities = [winter_random_probabilities, summer_random_probabilities, other_random_probabilities]
            
            temp_df = pd.DataFrame()
            hourly_data = season[season.local_time.dt.time == dt.time(hour)]['electricity']
            ecdf = ECDF(hourly_data)
            x = np.linspace(min(hourly_data), max(hourly_data), 1000)
            slope_changes = sorted(set(x))
            #0 slope change means all values are 0 for solar irradiance (because it is night time)
            #temp_df = pd.DataFrame()
            if(len(slope_changes) == 1):
                for k, random_value in enumerate(season_probabilities[n]):
                    capacity_factor = 0
                    row = {'OF CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                    temp_df = temp_df.append(row, ignore_index = True)
            
            else:
                temp_df = pd.DataFrame()
                sample_edf_values_at_slope_changes = [ ecdf(item) for item in slope_changes]
                inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes, fill_value='extrapolate')
                for k, random_value in enumerate(season_probabilities[n]):
                    capacity_factor = float(inverted_edf(random_value))
                    if(capacity_factor < 0):
                        capacity_factor = 0
                    row = {'OF CF %s Hour %s Season %s' %(file_name, hour, season_name) : capacity_factor}
                    temp_df = temp_df.append(row, ignore_index = True)
            
            OF_CF = pd.concat([OF_CF,temp_df], axis = 1).sort_index()
OF_CF


# In[35]:


OF_CF.to_excel('Offshore Wind Stochastic Output new sto seed.xlsx')


# In[ ]:




