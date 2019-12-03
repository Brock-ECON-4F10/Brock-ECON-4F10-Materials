# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:47:38 2019

@author: Ivan
"""

import pandas as pd
import statsmodels.api as sm

delta = pd.read_csv('DAL.csv')
aa = pd.read_csv('AAL.csv')

delta['Date'] = pd.to_datetime(delta['Date'])
aa['Date'] = pd.to_datetime(aa['Date'])

aa.set_index('Date',inplace=True) # replace dataset index with "Date" column
delta.set_index('Date',inplace=True) 

merged_data = pd.merge(aa,delta,left_index=True,right_index=True)
merged_data['Adj Close_y'].plot()
merged_data['Adj Close_x'].plot()

merged_data.iloc[(merged_data.index > '2018-08-01') & (merged_data.index < '2019-01-01')]['Adj Close_y'].plot()
merged_data.iloc[(merged_data.index > '2018-08-01') & (merged_data.index < '2019-01-01')]['Adj Close_x'].plot()

merged_data = sm.add_constant(merged_data)
full_sample_model = sm.OLS(merged_data['Adj Close_y'],merged_data[['const','Adj Close_x']]).fit()
full_sample_model.summary()

full_sample_model.resid.plot()

sub_data = merged_data.iloc[(merged_data.index > '2018-08-01') & (merged_data.index < '2019-01-01')]

restricted_model = sm.OLS(sub_data['Adj Close_y'],sub_data[['const','Adj Close_x']]).fit()
restricted_model.resid.plot()

adf_test_result = sm.tsa.stattools.adfuller(restricted_model.resid,maxlag=1)


