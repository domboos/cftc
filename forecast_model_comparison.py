#%%
import matplotlib.pyplot as plt
# import matplotlib as plt 
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
#* for other calcs:
from cfunctions import *

#For overview
model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc where model_type_id IN (82,76,95,100)", engine1)
model_types.head()
#%%

def getForecast(model_type,bb_tkr):
    models = pd.read_sql_query(f"SELECT * FROM cftc.model_desc where model_type_id ={model_type}",engine1)
    model_id = models[models.bb_tkr == bb_tkr].model_id.values[0]
    fcast = pd.read_sql_query(f"SELECT * from cftc.forecast where model_id = {model_id}",engine1)
    return fcast

def empirical_vals(model1_type_id,model2_type_id,bb_tkr):
    model_types = pd.read_sql_query(f"SELECT * from cftc.model_type_desc where model_type_id IN ({model1_type_id},{model2_type_id})", engine1).set_index('model_type_id')
    
    if list(model_types.cot_type)[0] != list(model_types.cot_type)[1]:
        sys.exit('wrong choice of models')
    else:
        empirical_vals = getexposure(type_of_trader = model_types.loc[model1_type_id,'cot_type'], norm = model_types.loc[model1_type_id,'cot_norm'], bb_tkr = bb_tkr)
        empirical_vals.columns = empirical_vals.columns.droplevel(0)
        empirical_vals['diff'] = empirical_vals.net_specs.diff()
        del empirical_vals['net_specs']
    return empirical_vals


def compare2ModelsForecast(model1_type_id,model2_type_id):

    oot = pd.read_sql_query("SELECT * FROM cftc.order_of_things",engine1)
    bb_tkrs = list(oot.bb_tkr)

    result = pd.DataFrame(index = bb_tkrs, columns = ['m1','m2','beta_m1','beta_m2','tstat_m1','pval_m1','tstat_m2','pval_m2','rsquared'])


    for bb_tkr in bb_tkrs:
        print(bb_tkr)
        fcast_m1 = getForecast(model_type=model1_type_id,bb_tkr= bb_tkr)
        fcast_m2 = getForecast(model_type=model2_type_id,bb_tkr= bb_tkr)
        emp_vals = empirical_vals(model1_type_id=model1_type_id, model2_type_id=model2_type_id, bb_tkr= bb_tkr)
        df = pd.merge(fcast_m1[['px_date','qty']],fcast_m2[['px_date','qty']], how = 'inner', on = 'px_date', suffixes=('_m1', '_m2'))
        df = pd.merge(df,emp_vals,how = 'left', on = 'px_date')
        
        #* Mincer Regression:  Mincer Zarnowitz Regression: empirc_vals = a + b_1*f_m1 + b_2*f_m2
        y = df['diff']
        x = df[['qty_m1','qty_m2']]
        x = sm.add_constant(x)
        mod_MZ = sm.OLS(y,x).fit()


        result.loc[bb_tkr,'beta_m1'] =    mod_MZ.params[1] # beta M1
        result.loc[bb_tkr,'beta_m2'] =  mod_MZ.params[2] # beta M2
        result.loc[bb_tkr,'tstat_m1'] =    mod_MZ.tvalues[1] #M1
        result.loc[bb_tkr,'pval_m1'] =    mod_MZ.pvalues[1] #pval M1
        result.loc[bb_tkr,'tstat_m2'] =    mod_MZ.tvalues[2] #M2
        result.loc[bb_tkr,'pval_m2'] =    mod_MZ.pvalues[2] #pval M2
        result.loc[bb_tkr,'rsquared'] =    mod_MZ.rsquared #r-squared
    
    result['m1'] = model1_type_id
    result['m2'] = model2_type_id
    return result 

#%% 

res1 = compare2ModelsForecast(model1_type_id = 76,model2_type_id= 82)
res2 = compare2ModelsForecast(model1_type_id = 95,model2_type_id= 100)

df = res1.append(res2)

df.to_excel('forecast_comparison.xlsx')
