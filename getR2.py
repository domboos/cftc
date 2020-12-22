#%%
import matplotlib.pyplot as plt
# import matplotlib as plt 
import statsmodels.api as sm

#* for other calcs:
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



#%%

bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things",engine1).bb_tkr
model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)
model_ids = pd.read_sql_query("SELECT model_id from cftc.vw_model_desc",engine1)


df_r2_MincerZarnowitz = pd.DataFrame(index  = model_types.model_type_id.drop_duplicates(), columns = bb_tkrs)
df_r2_fromHand = pd.DataFrame(index  = model_types.model_type_id.drop_duplicates(), columns = bb_tkrs)

df_direction = pd.DataFrame(index  = model_types.model_type_id.drop_duplicates(), columns = bb_tkrs)
df_coefs = pd.DataFrame( index =model_ids.model_id, columns =['const','beta0','tstat','975Int_const','025Int_const','975Int_beta','025Int_const'])

#%%
for idx in model_types.index: #iterates through model_type_ids
    print(f"model_type : {idx}")
    
    temp = model_types.loc[idx,:].T

    ongoingQuery = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {int(temp.loc['model_type_id'])}", engine1).set_index('model_id')
    
    
    for i in ongoingQuery.index: #iterates through model_id
        
        try:
            forecast = pd.read_sql_query(f"SELECT * FROM cftc.forecast WHERE model_id = {i}",engine1,index_col = 'px_date')
            exposure = getexposure(type_of_trader = model_types.loc[idx,'cot_type'], norm = model_types.loc[idx,'cot_norm'], bb_tkr = ongoingQuery.loc[i,'bb_tkr'])
            exposure.columns = exposure.columns.droplevel(0)
            exposure['diff'] = exposure.net_specs.diff()
            
            
            df_sample = pd.merge(left = forecast[['qty']], right = exposure[['diff']] , left_index = True, right_index = True, how = 'left')
            df_sample.columns = ['forecast','cftc']

            #get OpenInterst
            oi = gets(engine1, type='agg_open_interest', data_tab='vw_data', desc_tab='cot_desc', bb_tkr=ongoingQuery.loc[i,'bb_tkr'])
            oi.columns =[ 'oi']
            oi['OIma52'] = oi.rolling(52).mean()
            
            #merge with df_sample
            df_sample = pd.merge(df_sample,oi, right_index=True,left_index = True, how = 'left')
            df_sample['cftc_adj'] = df_sample.cftc / df_sample.OIma52
            df_sample['forecast_adj'] = df_sample.forecast / df_sample.OIma52
            
            #Calc OOS R2:
            e_diff = df_sample['cftc_adj'] - df_sample['forecast_adj']
            mspe_diff = (e_diff**2).sum(axis = 0)
            var_diff = ((df_sample['cftc_adj']-df_sample['cftc_adj'].mean(axis=0))**2).sum(axis = 0)
            try:
                oosR2_diff = 1 - mspe_diff/var_diff
                df_r2_fromHand.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = np.nan
            except ValueError:
                print("by calculating the r2 by hand something went wrong look at the following values:")
                print(mspe_diff)
                print(var_diff)
                df_r2_fromHand.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = np.nan
            
            #! Mincer Zarnowitz Regression:
            #* y-x = a + (b-1)x
            y = df_sample.cftc_adj.values - df_sample.forecast_adj
            x = sm.add_constant(df_sample.forecast_adj).values
            
            mod_fit0 = sm.OLS(y,x).fit()
            del y

            #* y = ax + b
            y = df_sample.cftc_adj.values
            mod_fit = sm.OLS(y,x).fit()
            # print(mod_fit.summary())
                    
            df_r2_MincerZarnowitz.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = mod_fit.rsquared
            
            df_coefs.loc[i,'tstat'] = mod_fit0.tvalues[1] #* test if beta1 is significantly diffent from 1
            df_coefs.loc[i,'const'] = mod_fit.params[0]
            df_coefs.loc[i,'beta0'] = mod_fit.params[1]
            df_coefs.loc[i,'975Int_const'] = mod_fit.conf_int(alpha = 0.05)[0][1]
            df_coefs.loc[i,'025Int_const'] = mod_fit.conf_int(alpha = 0.05)[0][0]
            df_coefs.loc[i,'975Int_beta'] = mod_fit.conf_int(alpha = 0.05)[1][1]
            df_coefs.loc[i,'025Int_const'] = mod_fit.conf_int(alpha = 0.05)[1][0]
            
            #* Some Plots
            # plt.plot(oi)
            # plt.pyplot.scatter(df_sample.forecast, df_sample.cftc)
            # sm.qqplot(mod_fit.resid, fit = True, line='45')
            
            # Direction right:
            df_sample['cftc_binary'] = df_sample.cftc.apply(binarity)
            df_sample['cftc_forecast'] = df_sample.forecast.apply(binarity)
            try:
                df_direction.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = df_sample[df_sample.cftc_forecast == df_sample.cftc_binary].cftc.count() / df_sample.shape[0]
            except:
                df_direction.loc[int(temp.loc['model_type_id']),ongoingQuery.loc[i,'bb_tkr']] = np.nan
                print('something went wrong')
        except:
            print(f"{ongoingQuery.loc[i,'bb_tkr']}: check regression")
            continue 

# %%
