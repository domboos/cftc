# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:25:44 2019

@author: grbi
"""

import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

#from datetime import datetime, timedelta

os.getcwd()
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc\\data')

#---------------------------------------------------------------------   
# for return signals:
maxlag = 250


# import datafiles:
nonC = pd.read_excel('Net_NonC.xlsx', index_col=0)
fut = pd.read_excel('Prices.xlsx', index_col=0)
carry_all = pd.read_excel('carry.xlsx',index_col = 0)
vix = pd.read_excel('vix.xlsx', index_col=0)


#set market you want to research:
ind = 'CL1'


#calculate log-returns:
ret = pd.DataFrame({ind : np.log(fut[ind]/fut[ind].shift(1))}, index = fut.index).fillna(0)
midx = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0],[0]])
ret.columns = midx



#get cftc data: create variable:
c = pd.DataFrame({'pos': nonC[ind]},index = nonC.index)
midx = pd.MultiIndex(levels=[['cftc'], ['net_specs']], codes=[[0], [0]])
c.columns = midx




#calculate Volatility
vol = 100 * ret.ewm(span = 150).std().diff(5) #meaning of span == n: alpha = 2/(n + 1)
midx = pd.MultiIndex(levels=[['ov'], ['vol']], codes=[[0],[0]])
vol.columns = midx



#process VIX data: 
vix = np.log(vix/vix.shift(1))
midx = pd.MultiIndex(levels=[['ov'], ['log_vix']], codes=[[0], [0]])
vix.columns = midx

#try: 'CT1', 'KC1', 'SB1',
carry = carry_all[ind].fillna(0, axis = 0)
#carry = carry_all[ind].fillna(method = 'bfill', axis = 0)
#carry = carry_all[ind].dropna()
# ----------------------------------------------------------------------------------------------------------------------
# create gamma the regularization matrix for the linreg
# todo: check if there is a bias (theoretical and empirical)
# how good are the oos results, test forcast vs. realized. Bias? or slope not equal one?
# how good is the weighing function (i + pp) / (maxlag + pp), what is the impact of rr
pp = 0.25
rr = 1/4


# number of additional variables
gamma = np.zeros((maxlag + 1, maxlag + 1))
#gamma1 = np.zeros((maxlag + 1, maxlag + 1))

# loop creates gamma and lagged returns in ret
# check this out: https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model

for i in range(0, maxlag+1):
    ret['ret', str(i+1).zfill(3)] = ret['ret', '000'].shift(i+1)
    
    # what happens to the first element, is there a downward bias? If pp is small that's probably irrelevant but with
    # significant contribution of pp that might become an issue.
    # Tikhonov regularization parameter
#    gamma[i, i] = 1 - 1 / (i+1)**rr
    
    gamma[i, i] = 1 - 1 / (i+1)**rr
#    gamma[i, i] = 1
    if i < maxlag:
        gamma[i, i + 1] = - gamma[i, i]



ret = ret.iloc[maxlag:,:] #delete the rows with nan due to its shift.

# Nail down the long end
gamma[maxlag, maxlag] = 1
#gamma[maxlag, maxlag] = 0


#plt.figure('gamma')
#plt.plot(np.diagonal(gamma))
#plt.title('gamma Coefficients')

# ----------------------------------------------------------------------------------------------------------------------
# evaluation of potential variables
# ----------------------------------------------------------------------------------------------------------------------

# inner join is to merge weekly data (cftc) and daily data (returns) / diff data 
c_ret = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).fillna(0)
c_ret_diff = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().fillna(0)

#todo: Variable selection: try and ( vix and vola single tryout)

#carry -diff: 
cf = pd.DataFrame({'carry': carry.diff(5)})
midx = pd.MultiIndex(levels=[['ov'], ['carry_diff']], codes=[[0],[0]])
cf.columns = midx

#carry abs: 
#cf = pd.DataFrame({'carry': carry})
#midx = pd.MultiIndex(levels=[['ov'], ['carry']], codes=[[0],[0]])
#cf.columns = midx


#vola:
#cf_temp = c.shift(1).copy()
#midx = pd.MultiIndex(levels=[['ov'], ['net_lag']], codes=[[0],[0]])
#cf_temp.columns = midx
#cf_temp = pd.merge(cf_temp, vol, how = 'inner', left_index =True, right_index = True)
#cf_temp['ov','vol_pos'] = cf_temp['ov']['net_lag']*cf_temp['ov']['vol']
#
#cf = pd.DataFrame(cf_temp['ov','vol_pos'])
#del cf_temp


#vix: 
#cf_temp = c.shift(1).copy()
#midx = pd.MultiIndex(levels=[['ov'], ['net_lag']], codes=[[0],[0]])
#cf_temp.columns = midx
#cf_temp = pd.merge(cf_temp, vix, how = 'inner', left_index =True, right_index = True)
#cf_temp['ov','vix_pos'] = cf_temp['ov']['net_lag']*cf_temp['ov']['log_vix']

#cf = cf_temp['ov','vix_pos']
#del cf_temp


#cc_ret: level incl. ov
cc_ret = pd.merge(c_ret, cf, how='inner', left_index=True, right_index=True)
#cc_ret_diff: delta incl diff of c: 
cc_ret_diff = pd.merge(c_ret_diff, cf, how='inner', left_index=True, right_index=True)

gamma_ov = np.concatenate((gamma, np.zeros((maxlag+1, cf.shape[1]))), axis=1)




# ----------------------------------------------------------------------------------------------------------------------
# Estimation:
# ----------------------------------------------------------------------------------------------------------------------
#set window
window = 52 *5

#smoothing parameter:
alpha = [2.0]

model_idx = cc_ret.index[:]
model_clm = pd.MultiIndex.from_product([alpha, ['level', 'diff_'], ['ov', 'dod']])
models = pd.DataFrame(index=model_idx, columns=model_clm)
scores = pd.DataFrame(index=model_idx, columns=model_clm)
prediction = pd.DataFrame(index=model_idx, columns=model_clm)


for idx,day in enumerate(cc_ret.index[0:-(window+1)]):
    
##  rolling window parameters:
    w_start = cc_ret.index[idx]
    w_end = cc_ret.index[idx + window]
    forecast_period = cc_ret.index[idx+window+1] # includes the day x in [:x]
    
##  variable for level
#    y = np.concatenate((cc_ret['cftc'].loc[w_start:w_end,:].values, np.zeros((maxlag + 1, 1))))
#    X_dod = np.concatenate((cc_ret['ret'].loc[w_start:w_end,:],gamma * alpha), axis=0)
    
##  variable for diffs:
    y_diff = np.concatenate((cc_ret_diff['cftc'].loc[w_start:w_end,:].values, np.zeros((maxlag + 1, 1))))
    X_dod_diff = np.concatenate((cc_ret_diff['ret'].loc[w_start:w_end,:],gamma * alpha), axis=0)
   
##  instruments for ov (other variables)
#    X_ov = np.concatenate((cc_ret[['ret', 'ov']].loc[w_start:w_end,:], gamma_ov * alpha),axis=0)
    X_ov_diff = np.concatenate((cc_ret_diff[['ret', 'ov']].loc[w_start:w_end,:], gamma_ov * alpha), axis=0)
    
##  fit the models
#    models.loc[w_end, (alpha, 'level', 'dod')] = sm.OLS(y,X_dod).fit() #sm.add_constant(X_dod)
#    models.loc[w_end, (alpha, 'level', 'ov')] = sm.OLS(y,X_ov).fit()
    models.loc[w_end, (alpha, 'diff_', 'dod')] = sm.OLS(y_diff,X_dod_diff).fit()
    models.loc[w_end, (alpha, 'diff_', 'ov')] = sm.OLS(y_diff,X_ov_diff).fit()
    
##  Rsquared - insample:
#    scores.loc[w_end, (alpha, 'level', 'dod')] = models.loc[w_end, (alpha, 'level', 'dod')].get_values()[0].rsquared
#    scores.loc[w_end, (alpha, 'level', 'ov')] = models.loc[w_end, (alpha, 'level', 'ov')].get_values()[0].rsquared        
    scores.loc[w_end, (alpha, 'diff_', 'dod')] = models.loc[w_end, (alpha, 'diff_', 'dod')].get_values()[0].rsquared
    scores.loc[w_end, (alpha, 'diff_', 'ov')] = models.loc[w_end, (alpha, 'diff_', 'ov')].get_values()[0].rsquared
    
##  Predictions    
    prediction.loc[forecast_period, (alpha, 'diff_', 'dod')] = \
    sum(models.loc[w_end, (alpha, 'diff_', 'dod')].get_values()[0].params * cc_ret_diff['ret'].loc[forecast_period,:])
    
    prediction.loc[forecast_period, (alpha, 'diff_', 'ov')] = \
    sum(models.loc[w_end, (alpha, 'diff_', 'ov')].get_values()[0].params * cc_ret_diff[['ret','ov']].loc[forecast_period,:])
    
#    prediction.loc[forecast_period, (alpha, 'level', 'dod')] = \
#    sum(models.loc[w_end, (alpha, 'level', 'dod')].get_values()[0].params * cc_ret['ret'].loc[forecast_period,:])
#    
#    prediction.loc[forecast_period, (alpha, 'level', 'ov')] = \
#    sum(models.loc[w_end, (alpha, 'level', 'ov')].get_values()[0].params * cc_ret[['ret','ov']].loc[forecast_period,:])
    
    print(w_end)

 

#------------------------------------------------------------------------------
#-------------------------------- diff: ---------------------------------------
pred = prediction.loc[:,(alpha, ['diff_'], ['ov','dod'])].astype(float)   
emp = cc_ret_diff['cftc','net_specs']


results_diff = pd.merge(emp,pred, how='inner', left_index=True, right_index=True).dropna()
results_diff.columns = ['cftc', 'forecast_ov', 'forecast_dod']



print(smf.ols('cftc ~ forecast_dod',results_diff).fit().summary())
print(smf.ols('cftc ~ forecast_ov',results_diff).fit().summary())

plt.scatter(results_diff['forecast_ov'],results_diff['cftc'])
plt.scatter(results_diff['forecast_dod'],results_diff['cftc'])




plt.figure()
sns.regplot(x="forecast_ov", y="cftc", data=results_diff)
sns.regplot(x="forecast_dod", y="cftc", data=results_diff)
#------------------------------------------------------------------------------
#-------------------------------- level: ---------------------------------------
pred = prediction.loc[:,(alpha, ['level'], ['ov','dod'])].astype(float)   
emp = cc_ret['cftc','net_specs']


results_lvl = pd.merge(emp,pred, how='inner', left_index=True, right_index=True).dropna()
results_lvl.columns = ['cftc', 'forecast_ov', 'forecast_dod']

plt.figure()
plt.scatter(results_lvl['forecast_ov'],results_lvl['cftc'])
plt.scatter(results_lvl['forecast_dod'],results_lvl['cftc'])

print(smf.ols('cftc ~ forecast_ov',results_lvl).fit().summary())
print(smf.ols('cftc ~ forecast_dod',results_lvl).fit().summary())


plt.figure()
sns.regplot(x="forecast_ov", y="cftc", data=results_lvl) #2009-06-23 00:00:00
sns.regplot(x="forecast_dod", y="cftc", data=results_lvl)

plt.plot(results_lvl.index,results_lvl['cftc'] - results_lvl['forecast_ov'])


# ----------------------------------------------------------------------------------------------------------------------
# Appendix:
# ----------------------------------------------------------------------------------------------------------------------

# checking if no look ahead bias: first diff, then lvl - model   
#cc_ret_diff['ret'].loc[w_start:w_end,:].tail() #for estimation
#cc_ret_diff['ret'].loc[forecast_period,:].head() #out-of sample rets

#cc_ret['ret'].loc[w_start:w_end,:].tail() #for estimation
#cc_ret['ret'].loc[forecast_period,:].head() #out-of sample rets


#ewm:  halflife
#result_t = []
#for i in np.arange(0.01,10,0.01):
#    t = 1 - np.exp(np.log(0.5)/10)
#    result_t.append(t)


