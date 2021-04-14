#%%
import pandas as pd
import numpy as np 
import statsmodels.api as sm
from datetime import datetime
import os 
os.chdir('/home/jovyan/work/')
from cfunctions import *
import sys
#%%
#For overview
model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc where model_type_id IN (82,76,95,100)", engine1)
model_types.head()
#%% #* Get Data:
def getForecast(model_type,bb_tkr):
    models = pd.read_sql_query(f"SELECT * FROM cftc.model_desc where model_type_id ={model_type}",engine1)
    model_id = models[models.bb_tkr == bb_tkr].model_id.values[0]
    fcast = pd.read_sql_query(f"SELECT * from cftc.forecast where model_id = {model_id}",engine1)
    return fcast

def empirical_vals2(model1_type_id,model2_type_id,bb_tkr):
    #* for Zarnowitz2Models
    model_types = pd.read_sql_query(f"SELECT * from cftc.model_type_desc where model_type_id IN ({model1_type_id},{model2_type_id})", engine1).set_index('model_type_id')
    
    #Sanity Check if both models have same dependent variable 
    if list(model_types.cot_type)[0] != list(model_types.cot_type)[1]:
        print('wrong choice of models')
    else:
        empirical_vals = getexposure(type_of_trader = model_types.loc[model1_type_id,'cot_type'], norm = model_types.loc[model1_type_id,'cot_norm'], bb_tkr = bb_tkr)
        empirical_vals.columns = empirical_vals.columns.droplevel(0)
        empirical_vals['diff'] = empirical_vals.net_specs.diff()
        del empirical_vals['net_specs']
    return empirical_vals


def empirical_vals1(model_type_id,bb_tkr):
    #* for Zarnowitz1Models
    #* Already returns diff() values for empirical values
    model_types = pd.read_sql_query(f"SELECT * from cftc.model_type_desc where model_type_id IN ({model_type_id})", engine1).set_index('model_type_id')

    empirical_vals = getexposure(type_of_trader = model_types.loc[model_type_id,'cot_type'], norm = model_types.loc[model_type_id,'cot_norm'], bb_tkr = bb_tkr)
    empirical_vals.columns = empirical_vals.columns.droplevel(0)
    empirical_vals['diff'] = empirical_vals.net_specs.diff()
    
    del empirical_vals['net_specs']
    return empirical_vals
#%%% #* Mincer-Zarnowitz Regression for 1 Model and Combined
def Zarnowitz1Model(model_type_id):
    model_type_id = 93
    oot = pd.read_sql_query("SELECT * FROM cftc.order_of_things",engine1)
    bb_tkrs = list(oot.bb_tkr)

    result = pd.DataFrame(index = bb_tkrs, columns = ['model_type_id','intercept','tstat_intercept','pval_intercept','beta','tstat_beta','pval_beta','rsquared'])
    result['model_type_id'] = model_type_id

    for bb_tkr in bb_tkrs:
        print(bb_tkr)
        fcast = getForecast(model_type=model_type_id,bb_tkr= bb_tkr)
        emp_vals = empirical_vals1(model_type_id=model_type_id, bb_tkr= bb_tkr)
        df = pd.merge(fcast,emp_vals,how = 'left', on = 'px_date')
        df = df.dropna()
        
        #* Mincer Zarnowitz Regression:y-x = a + (b-1)x
        y = (df['diff'] - df['qty']).values
        x = sm.add_constant(df['qty']).values
        mod_MZ = sm.OLS(y,x).fit()
        # print(mod_MZ.summary())

        #wirte to results:
        result.loc[bb_tkr,'intercept'] =  mod_MZ.params[0] 
        result.loc[bb_tkr,'tstat_intercept'] = mod_MZ.tvalues[0]
        result.loc[bb_tkr,'pval_intercept'] =  mod_MZ.pvalues[0] 
        result.loc[bb_tkr,'beta'] =  mod_MZ.params[1] 
        result.loc[bb_tkr,'tstat_beta'] = mod_MZ.tvalues[1]
        result.loc[bb_tkr,'pval_beta'] =  mod_MZ.pvalues[1] 
        result.loc[bb_tkr,'rsquared'] = mod_MZ.rsquared 
        
    return result


def Zarnowitz2Models(model1_type_id,model2_type_id):
    oot = pd.read_sql_query("SELECT * FROM cftc.order_of_things",engine1)
    bb_tkrs = list(oot.bb_tkr)

    result = pd.DataFrame(index = bb_tkrs, columns = ['m1','m2','beta_m1','beta_m2','tstat_m1','pval_m1','tstat_m2','pval_m2','rsquared'])

    for bb_tkr in bb_tkrs:
        print(bb_tkr)
        fcast_m1 = getForecast(model_type=model1_type_id,bb_tkr= bb_tkr)
        fcast_m2 = getForecast(model_type=model2_type_id,bb_tkr= bb_tkr)
        emp_vals = empirical_vals2(model1_type_id=model1_type_id,model2_type_id= model2_type_id, bb_tkr= bb_tkr)
        df = pd.merge(fcast_m1[['px_date','qty']],fcast_m2[['px_date','qty']], how = 'inner', on = 'px_date', suffixes=('_m1', '_m2'))
        df = pd.merge(df,emp_vals,how = 'left', on = 'px_date')
        
        #* Mincer Regression:  Mincer Zarnowitz Regression: empirc_vals = a + b_1*f_m1 + b_2*f_m2
        y = df['diff']
        x = df[['qty_m1','qty_m2']]
        x = sm.add_constant(x)
        mod_MZ = sm.OLS(y,x).fit()


        result.loc[bb_tkr,'beta_m1'] =  mod_MZ.params[1] # beta M1
        result.loc[bb_tkr,'beta_m2'] =  mod_MZ.params[2] # beta M2
        result.loc[bb_tkr,'tstat_m1'] = mod_MZ.tvalues[1] #M1
        result.loc[bb_tkr,'pval_m1'] =  mod_MZ.pvalues[1] #pval M1
        result.loc[bb_tkr,'tstat_m2'] = mod_MZ.tvalues[2] #M2
        result.loc[bb_tkr,'pval_m2'] =  mod_MZ.pvalues[2] #pval M2
        result.loc[bb_tkr,'rsquared'] = mod_MZ.rsquared #r-squared
    
    result['m1'] = model1_type_id
    result['m2'] = model2_type_id
    return result 

#%% #* Exposure:
res1 = Zarnowitz2Models(model1_type_id = 76,model2_type_id= 82)
res2 = Zarnowitz2Models(model1_type_id = 95,model2_type_id= 100)
df = res1.append(res2)
df.to_excel('forecast_comparison_v2.xlsx')

#%% #* Openinterest:
resboth = Zarnowitz2Models(model1_type_id = 93,model2_type_id= 137)
res93 = Zarnowitz1Model(model_type_id= 93)
res137 = Zarnowitz1Model(model_type_id= 137)

os.chdir('/home/jovyan/work/reports/results')
writer = pd.ExcelWriter('ZarnowitzRegressions_93_137.xlsx', engine='xlsxwriter')
resboth.to_excel(writer, sheet_name= 'MZ_137_and_93')
res93.to_excel(writer, sheet_name = 'MZ_93') 
res137.to_excel(writer, sheet_name='MZ_137')
writer.save()
os.chdir('/home/jovyan/work/')