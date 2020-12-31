# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:04:13 2020

@author: grbi
"""

#%%

import matplotlib as plt 
import statsmodels.api as sm

import os 
os.chdir('C:\\Users\\bood\\PycharmProjects\\cftc')
from cfunctions import *
#%%
#----------------- Define which models you'd like to check:------------------------------------

def binarity(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:  
        return 0


model_types = pd.read_sql_query(str("SELECT * from cftc.model_type_desc"), engine1)

ongoingQuery = pd.read_sql_query(str(" Select distinct(bb_tkr) from cftc.model_desc;"), engine1)
#%%

df_result =pd.DataFrame(index  = model_types.model_type_id.drop_duplicates(), columns = pd.read_sql_query(str(" Select distinct(bb_tkr) from cftc.model_desc;"), engine1).bb_tkr)
df_MincerZarnowitz = pd.DataFrame(index  = model_types.model_type_id.drop_duplicates(), columns = pd.read_sql_query(str(" Select distinct(bb_tkr) from cftc.model_desc;"), engine1).bb_tkr)
df_direction = pd.DataFrame(index  = model_types.model_type_id.drop_duplicates(), columns = pd.read_sql_query(str(" Select distinct(bb_tkr) from cftc.model_desc;"), engine1).bb_tkr)
for idx in model_types.index:
    print(idx)
    temp = model_types.loc[idx,:].T

    ongoingQuery = pd.read_sql_query(str(" Select * from cftc.model_desc where model_type_id ="+ str(int(temp.loc['model_type_id']))), engine1)
    for i in ongoingQuery.index:
        print(i)
        
        forecast = pd.read_sql_query(str('select * from cftc.forecast where model_id =' +str(ongoingQuery.loc[i,'model_id'])), engine1,index_col = 'px_date')
        exposure = getexposure(type_of_trader = model_types.loc[idx,'cot_type'], norm = model_types.loc[idx,'cot_norm'], bb_tkr = ongoingQuery.loc[i,'bb_tkr'])
        exposure.columns = exposure.columns.droplevel(0)
        exposure['diff'] = exposure.net_specs.diff()
        
        
        df_sample = pd.merge(left = forecast[['qty']], right = exposure[['diff']] , left_index = True, right_index = True, how = 'left')
        df_sample.columns = ['forecast','cftc']
        # plt.pyplot.scatter(df_sample.forecast, df_sample.cftc)
        
        
        #Calc OOS R2:
        e_diff = df_sample['cftc'] - df_sample['forecast']
        mspe_diff = (e_diff**2).sum(axis = 0)
        var_diff = ((df_sample['cftc']-df_sample['cftc'].mean(axis=0))**2).sum(axis = 0)
        oosR2_diff = 1 - mspe_diff/var_diff
        df_result.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = oosR2_diff
        
        # Mincer Zarnowitz Regression:
        mod_fit = sm.OLS(df_sample.cftc,df_sample.forecast).fit()
        df_MincerZarnowitz.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = mod_fit.rsquared
        
        # Direction right:
        df_sample['cftc_binary'] = df_sample.cftc.apply(binarity)
        df_sample['cftc_forecast'] = df_sample.forecast.apply(binarity)
        try:
            df_direction.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = df_sample[df_sample.cftc_forecast == df_sample.cftc_binary].cftc.count() / df_sample.shape[0]
        except:
            print('something went wrong')
        df_result.loc[int(temp.loc['model_type_id']), ongoingQuery.loc[i,'bb_tkr']] = oosR2_diff

df_result.to_excel('C:\\Users\\bood\\switchdrive\\Tracking Traders\\R2.xlsx')
