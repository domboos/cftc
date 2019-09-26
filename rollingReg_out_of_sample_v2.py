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

from datetime import datetime, timedelta

os.getcwd()
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc\\data')

#---------------------------------------------------------------------   
# for return signals:
maxlag = 250


# import and create returns
fut = pd.read_excel('Prices.xlsx', index_col=0)


#set market you want to research:
ind = 'EC1'


#calculate log-returns:
ret = pd.DataFrame({ind : np.log(fut[ind]/fut[ind].shift(1))}, index = fut.index).fillna(0)
midx = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0],[0]])
ret.columns = midx



#get cftc data: create variable:
nonC = pd.read_excel('Net_NonC.xlsx', index_col=0)
c = pd.DataFrame({'pos': nonC[ind]},index = nonC.index)
midx = pd.MultiIndex(levels=[['cftc'], ['net_specs']], codes=[[0], [0]])
c.columns = midx




#calculate Volatility
vol = 100 * ret.ewm(span = 10).std()
vol1 = vol.diff(5)
midx = pd.MultiIndex(levels=[['ov'], ['vol']], codes=[[0],[0]])
vol.columns = midx



#get VIX data: 
vix = pd.read_excel('vix.xlsx', index_col=0)
vix = np.log(vix/vix.shift(1))
midx = pd.MultiIndex(levels=[['ov'], ['log_vix']], codes=[[0], [0]])
vix.columns = midx


# ----------------------------------------------------------------------------------------------------------------------
# create gamma the regularization matrix for the linreg
# todo: check if there is a bias (theoretical and empirical)
# how good are the oos results, test forcast vs. realized. Bias? or slope not equal one?
# how good is the weighing function (i + pp) / (maxlag + pp), what is the impact of rr
pp = 0.25
rr = 1/4


# number of additional variables
gamma = np.zeros((maxlag + 1, maxlag + 1))
gamma1 = np.zeros((maxlag + 1, maxlag + 1))

# loop creates gamma and lagged returns in ret
# check this out: https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model

for i in range(0, maxlag+1):
    ret['ret', str(i+1).zfill(3)] = ret['ret', '000'].shift(i+1)
    
    # what happens to the first element, is there a downward bias? If pp is small that's probably irrelevant but with
    # significant contribution of pp that might become an issue.
    # Tikhonov regularization parameter
#    gamma[i, i] = 1 - 1 / (i+1)**rr
    
    gamma[i, i] = 1 - 1 / (i+1)**(rr*1)
    gamma1[i, i] = 1 - 1 / (i+1)**(rr*2)
    if i < maxlag:
        gamma[i, i + 1] = - gamma[i, i]



ret = ret.iloc[maxlag:,:]

# Nail down the long end
#gamma[maxlag, maxlag] = 1
gamma[maxlag, maxlag] = 0



plt.figure('gamma')
plt.plot(np.diagonal(gamma))
plt.plot(np.diagonal(gamma1))
plt.title('gamma Coefficients')


# ----------------------------------------------------------------------------------------------------------------------
# evaluation of potential variables
# ----------------------------------------------------------------------------------------------------------------------

# inner join is to merge weekly data (cftc) and daily data (returns) / diff data 
c_ret = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).fillna(0)
c_ret_diff = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().fillna(0)


#x1 = ret.iloc[:,0].shift(4)


#Merge other variables to cf:
cf = c.shift(1) #

midx = pd.MultiIndex(levels=[['ov'], ['cftc_lag1']], codes=[[0],[0]])
cf.columns = midx
#Todo: multiply vola with the posititon on the day before:
cf = pd.merge(cf, vol, how='inner', left_index=True, right_index=True).fillna(0)
cf = pd.merge(cf, vix, how='inner', left_index=True, right_index=True).dropna(0)
cf.head()

# !!!
#todo: 
cf['ov','vix*lagpos'] = cf['ov','log_vix']*cf['ov','cftc_lag1']
cf['ov','vol*lagpos'] = cf['ov','log_vix']*cf['ov','cftc_lag1']
cf = cf.iloc[:,[0,1,3]] #drop vix
#cf.head()



#cc_ret: level incl. ov
cc_ret = pd.merge(c_ret, cf, how='inner', left_index=True, right_index=True)
#cc_ret_diff: delta incl diff of c: 
cc_ret_diff = pd.merge(c_ret_diff, cf, how='inner', left_index=True, right_index=True)

gamma_ov = np.concatenate((gamma, np.zeros((maxlag+1, cf.shape[1]))), axis=1)


#del cf, midx, c_ret, c_ret_diff,pp,rr, c


#set window
window = 52 *5

#smoothing parameter:
alpha = [2.0]

model_idx = cc_ret.index[:]
model_clm = pd.MultiIndex.from_product([A, ['level', 'diff_'], ['ov', 'dod']])
models = pd.DataFrame(index=model_idx, columns=model_clm)
scores = pd.DataFrame(index=model_idx, columns=model_clm)
prediction = pd.DataFrame(index=model_idx, columns=model_clm)

tryme = pd.DataFrame(index=model_idx, columns=model_clm)


# try and error: 
#day = '2005-01-04 00:00:00'
#print(cc_ret.index[0])
#print((cc_ret.index[0]+ timedelta(days=window)))
#cc_ret.index[0]


for idx,day in enumerate(cc_ret.index[window:-1]):
    
    temp_window_start = day - timedelta(weeks= window)
    forecast_period = day + timedelta(weeks = 1) # includes the day x in [:x]
    
    #variable for level
#    y = np.concatenate((['cftc'].loc[temp_window_start:day,:].values, np.zeros((maxlag + 1, 1))))
#    X_dod = np.concatenate((cc_ret['ret'].loc[temp_window_start:day,:],gamma * alpha), axis=0)
    
##  variable for diffs
    y_diff = np.concatenate((cc_ret_diff['cftc'].loc[temp_window_start:day,:].values, np.zeros((maxlag + 1, 1))))
    X_dod_diff = np.concatenate((cc_ret_diff['ret'].loc[temp_window_start:day,:],gamma * alpha), axis=0)
    
       
    # instruments for ov (other variables)
    X_ov = np.concatenate((cc_ret[['ret', 'ov']].loc[temp_window_start:day,:], gamma_ov * alpha),axis=0)
    X_ov_diff = np.concatenate((cc_ret_diff[['ret', 'ov']].loc[temp_window_start:day,:], gamma_ov * alpha), axis=0)
 
    
#    #fit the models
#    models.loc[day, (alpha, 'level', 'dod')] = sm.OLS(y,sm.add_constant(X_dod)).fit()
#    models.loc[day, (alpha, 'level', 'ov')] = sm.OLS(y,X_ov).fit()
    models.loc[day, (alpha, 'diff_', 'dod')] = sm.OLS(y_diff,X_dod_diff).fit()
    models.loc[day, (alpha, 'diff_', 'ov')] = sm.OLS(y_diff,X_ov_diff).fit()
    
#    print(sm.OLS(y_diff,X_ov_diff).fit().summary())
    
#    scores.loc[i, (alpha, 'level', 'dod')] = models.loc[day, (alpha, 'level', 'dod')].get_values()[0].rsquared
#    scores.loc[i, (alpha, 'level', 'ov')] = models.loc[day, (alpha, 'level', 'ov')].get_values()[0].rsquared        
    scores.loc[day, (alpha, 'diff_', 'dod')] = models.loc[day, (alpha, 'diff_', 'dod')].rsquared
    scores.loc[day, (alpha, 'diff_', 'ov')] = models.loc[day, (alpha, 'diff_', 'ov')].rsquared
    
    prediction.loc[forecast_period, (alpha, 'diff_', 'dod')] = \
    sum(models.loc[day, (alpha, 'diff_', 'dod')].params * cc_ret_diff['ret'].loc[forecast_period,:])
    
    prediction.loc[forecast_period, (alpha, 'diff_', 'ov')] = \
    sum(models.loc[day, (alpha, 'diff_', 'ov')].params * cc_ret_diff[['ret','ov']].loc[forecast_period,:])
    
    
    print(day) 
    
# checking if no look ahead bias:     
#cc_ret_diff['ret'].loc[temp_window_start:day,:].tail()
#cc_ret_diff['ret'].loc[forecast_period,:].head()
#cc_ret_diff['ret'].loc[day:(day+ timedelta(weeks = 4)),:]

  

#------------------------------------------------------------------------------
# Without multiindexes: 
pred = prediction.loc[:,(alpha, 'diff_', ['ov','dod'])].astype(float)   
emp = cc_ret_diff['cftc','net_specs']



x = pd.merge(emp,pred, how='inner', left_index=True, right_index=True).dropna()
x.columns = ['cftc', 'forecast_ov', 'forecast_dod']
    

x.head()
plt.figure()
plt.scatter(x['forecast_ov'],x['cftc'])
plt.scatter(x['forecast_dod'],x['cftc'])

mod5 = smf.ols('cftc ~ forecast_ov',x).fit()
print(mod5.summary())

print(smf.ols('cftc ~ forecast_dod',x).fit().summary())

plt.figure()
sns.regplot(x="forecast_ov", y="cftc", data=x)
sns.regplot(x="forecast_dod", y="cftc", data=x)



#------------------------------------------------------------------------------



plt.plot(x)

print(models.loc[i, (alpha, 'diff_', 'dod')].get_values()[0].summary())

scores = scores.set_index(scores['Dates'],drop = True)
scores = scores.dropna()
del scores['Dates']

models = models.set_index(models['Dates'],drop = True)
models = models.dropna()
del models['Dates']

print(models.loc[i, (alpha, 'diff_', 'ov')].get_values()[0].summary())


plt.figure(str(ind)+ '_scores')
plt.plot(scores)
plt.title('R-squared, market:' + str(ind))
plt.legend(scores.columns, loc='best')

#---------------------------------------------------------------------

