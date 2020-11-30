# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:04:13 2020

@author: grbi
"""

import matplotlib as plt 
import os 
os.chdir('C:\\Users\\bood\\PycharmProjects\\cftc')
from cfunctions import *

#----------------- Define which models you'd like to check:------------------------------------




model_types = pd.read_sql_query(str("SELECT * from cftc.model_type_desc"), engine1)

ongoingQuery = pd.read_sql_query(str(" Select distinct(bb_tkr) from cftc.model_desc;"), engine1)

df_result =pd.DataFrame(index  = model_types.model_type_id.drop_duplicates(), columns = pd.read_sql_query(str(" Select distinct(bb_tkr) from cftc.model_desc;"), engine1).bb_tkr)
for idx in model_types.index:
    print(idx)
    temp = model_types.loc[idx,:].T

    ongoingQuery = pd.read_sql_query(str(" Select * from cftc.model_desc where model_type_id ="+ str(int(temp.loc['model_type_id']))), engine1)
    for i in ongoingQuery.index:
        # print(i)
        
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
        df_result.loc[int(temp.loc['model_type_id']), ongoingQuery.loc[i,'bb_tkr']] = oosR2_diff

df_result.to_excel('C:\\Users\\bood\\switchdrive\\Tracking Traders\\02_Daten\\R2.xlsx')

    
    # cot_type = model_types.loc[idx,'cot_type']  # or net_non_commercials
    # cot_norm = model_types.loc[idx,'cot_norm']
    # est_window = model_types.loc[idx,'est_window']
    # lookback = model_types.loc[idx,'lookback']
    # gamma_type = model_types.loc[idx,'gamma_type']  #OR: 'dom', 'sqrt' , 'arctan', 'flat', 'log
    # gamma_para = model_types.loc[idx,'gamma_para']
    # naildown_value0 = model_types.loc[idx,'naildown_value0']
    # naildown_value1 = model_types.loc[idx,'naildown_value1']
    # alpha = 1e-06 # OR: 





# #get Models
# q = str("SELECT * FROM cftc.vw_model_desc" + " where 1=1 and max_date is not null " + "and cot_type = '" + str(cot_type) + "' and cot_norm ='"  + cot_norm + "' and gamma_type = '" + gamma_type+ "' and alpha = "+str(1e-06))
# df_models = pd.read_sql_query(q, engine1)


# r2_results = pd.DataFrame(index = df_models.bb_tkr, columns =['r2'])
# for idx in df_models.index:
#     modelid = df_models.loc[idx,'model_id']
#     bb_tkr =  df_models.loc[idx,'bb_tkr']
#     df_models.columns
#     forecast = pd.read_sql_query(str('select * from cftc.forecast where model_id =' +str(modelid)), engine1,index_col = 'px_date')

#     exposure = getexposure(type_of_trader = cot_type, norm = cot_norm, bb_tkr = bb_tkr)
#     exposure.columns = exposure.columns.droplevel(0)
#     exposure['diff'] = exposure.net_specs.diff()
    
    
#     df_sample = pd.merge(left = forecast[['qty']], right = exposure[['diff']] , left_index = True, right_index = True, how = 'left')
#     df_sample.columns = ['forecast','cftc']
#     df_sample.head()

#     #Calc OOS R2:
#     e_diff = df_sample['cftc'] - df_sample['forecast']
#     mspe_diff = (e_diff**2).sum(axis = 0)
#     var_diff = ((df_sample['cftc']-df_sample['cftc'].mean(axis=0))**2).sum(axis = 0)
#     oosR2_diff = 1 - mspe_diff/var_diff
    
#     r2_results.loc[bb_tkr,'r2'] = oosR2_diff
#     # mod_lvl = smf.ols('cftc ~ forecast',df_sample).fit()
#     # df_sample.plot(kind = 'scatter',x = 'forecast', y = 'cftc')     
#     # fig =  sns.lmplot(x='forecast', y='cftc', data=df_sample)
