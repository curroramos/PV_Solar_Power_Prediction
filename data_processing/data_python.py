#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:46:28 2022

@author: curro
"""

import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd

mat = loadmat('/Users/curro/PV_prediction/Matlab_data/solarRad.mat')  # load mat-file
mdata = mat['vd']



initial_date = datetime(2014, 1, 1, 0, 0, 0) # ([2014 7 19  7  0 0])
final_date = datetime(2014, 12, 29, 23, 59, 55) #([2014 7 25 21 0 0])

# Create a datetimeindex
date_series = pd.date_range(start=initial_date,end=final_date, freq = '5min')
# # Delete last time
# date_series = date_series[:-1]

# Create a dataframe
columns = ['Irradiancia solar.  NIP pyrheliometer (W/m2)',
    'Temperatura (ºC)',
    'Presión barométrica (MBar)',
    'Humedad Relativa (%)',
    'Irradiancia Teórica (W/m2)',
    'Desviación Radiación directa (W/m2)']

df = pd.DataFrame(mdata, index = date_series, columns = columns)

# Crop the data
# a = 80000
# df = df[a + 19000: a + 20000]

# Plot the data
df.plot(subplots=True, figsize=(12, 12));

# Box plot
df.plot.box(subplots=True, figsize=(20, 10))

# Filter invalid data
#Como saber qué datos no sirven?

# Calculate the mean of the times
df = df.resample('1h').mean()

# # Calculate the daily mean temperature
# df['Temperatura (ºC)'] = df['Temperatura (ºC)'].resample('1D').mean()
# # Fill Nan with previous value
# df['Temperatura (ºC)'] = df['Temperatura (ºC)'].fillna(method='ffill')

# # Calculate the daily mean pressure
# df['Presión barométrica (MBar)'] = df['Presión barométrica (MBar)'].resample('1D').mean()
# # Fill Nan with previous value
# df['Presión barométrica (MBar)'] = df['Presión barométrica (MBar)'].fillna(method='ffill')

# Plot the resampled data
df.plot(subplots=True, figsize=(12, 12));

# Correlation between different variables
for column in columns[1:]:
    col1 = columns[0]
    col2 = column
    corr = df[col1].corr(df[col2])
    print(f"Correlation between {col1} and {col2} is {str(round(corr,2))}")

# Filter outliers
x = df['Temperatura (ºC)']
df['Temperatura (ºC)'] = x[x.between(x.quantile(.15), x.quantile(.85))] 

x = df['Presión barométrica (MBar)']
df['Presión barométrica (MBar)'] = x[x.between(x.quantile(.15), x.quantile(.85))]

x = df['Humedad Relativa (%)']
df['Humedad Relativa (%)'] = x[x.between(x.quantile(.15), x.quantile(.85))]

# Fill empty values
df = df.fillna(method = 'bfill')
df = df.fillna(method = 'ffill')

# Correlation after filtering
print('\n Correlation after filtering and filling values:')
for column in columns[1:]:
    col1 = columns[0]
    col2 = column
    corr = df[col1].corr(df[col2])
    print(f"Correlation between {col1} and {col2} is {str(round(corr,2))}")


# Plot the filtered data
df.plot(subplots=True, figsize=(12, 12));

# Box plot filtered data
df.plot.box(subplots=True, figsize=(20, 10))

# Export data
df.to_json('processed_data/multivariate_data_filtered.json')













