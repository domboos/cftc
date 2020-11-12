# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:04:13 2020

@author: grbi
"""


import os 
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu')
from cfunctions import *

#----------------- Define which models you'd like to check:------------------------------------
cot_type = 'net_managed_money'  # or net_non_commercials
cot_norm = 'exposure' #TODO: include models with net/oi
gamma_type = 'linear1' #OR: 'dom', 'sqrt' , 'arctan', 'flat', 'log
alpha = 1e-06 # OR: 

    
#get Models
q = str("SELECT * FROM cftc.vw_model_desc" + " where 1=1 and max_date is not null " + "and cot_type = '" + str(cot_type) + "' and cot_norm ='"  + cot_norm + "' and gamma_type = '" + gamma_type+ "' and alpha = "+str(1e-06))
df_models = pd.read_sql_query(q, engine1)


r2_results = pd.DataFrame(index = df_models.bb_tkr, columns =['r2'])
for idx in df_models.index:
    modelid = df_models.loc[idx,'model_id']
    bb_tkr =  df_models.loc[idx,'bb_tkr']
    df_models.columns
    forecast = pd.read_sql_query(str('select * from cftc.forecast where model_id =' +str(modelid)), engine1,index_col = 'px_date')

    exposure = getexposure(type_of_trader = cot_type, norm = cot_norm, bb_tkr = bb_tkr)
    exposure.columns = exposure.columns.droplevel(0)
    exposure['diff'] = exposure.net_specs.diff()
    
    
    df_sample = pd.merge(left = forecast[['qty']], right = exposure[['diff']] , left_index = True, right_index = True, how = 'left')
    df_sample.columns = ['forecast','cftc']
    df_sample.head()

    #Calc OOS R2:
    e_diff = df_sample['cftc'] - df_sample['forecast']
    mspe_diff = (e_diff**2).sum(axis = 0)
    var_diff = ((df_sample['cftc']-df_sample['cftc'].mean(axis=0))**2).sum(axis = 0)
    oosR2_diff = 1 - mspe_diff/var_diff
    
    r2_results.loc[bb_tkr,'r2'] = oosR2_diff
    # mod_lvl = smf.ols('cftc ~ forecast',df_sample).fit()
    # df_sample.plot(kind = 'scatter',x = 'forecast', y = 'cftc')     
    # fig =  sns.lmplot(x='forecast', y='cftc', data=df_sample)
