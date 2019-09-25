# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:25:44 2019

@author: grbi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:29:57 2019

@author: grbi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:14:06 2019

@author: grbi
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os

os.getcwd()
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc\\data')

#---------------------------------------------------------------------   
# for return signals:
maxlag = 250


# import and create returns
fut = pd.read_excel('Prices.xlsx', index_col=0)


#set market you want to research:
ind = 'KC1'


#calculate log-returns:
ret = pd.DataFrame({ind : np.log(fut[ind]/fut[ind].shift(1))})
ret = ret.dropna()
midx = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0],[0]])
ret.columns = midx

#get cftc data: create variable:
nonC = pd.read_excel('Net_NonC.xlsx', index_col=0)
c = pd.DataFrame({'pos': nonC[ind]})
midx = pd.MultiIndex(levels=[['cftc'], ['net_specs']], codes=[[0], [0]])
c.columns = midx


#calculate Volatility
vol = 100 * ret.ewm(span = 60).std()
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
# loop creates gamma and lagged returns in ret
# check this out: https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model

for i in range(0, maxlag+1):
    ret['ret', str(i+1).zfill(3)] = ret['ret', '000'].shift(i+1)
    
    # what happens to the first element, is there a downward bias? If pp is small that's probably irrelevant but with
    # significant contribution of pp that might become an issue.
    # Tikhonov regularization parameter
    gamma[i, i] = 1 - 1 / (i+1)**rr
    if i < maxlag:
        gamma[i, i + 1] = - gamma[i, i]

ret = ret.dropna()

# Nail down the long end
gamma[maxlag, maxlag] = 1
#gamma[maxlag, maxlag] = 0


#plt.figure('gamma')
#plt.plot(np.diagonal(gamma))
#plt.title('gamma Coefficients')


# ----------------------------------------------------------------------------------------------------------------------
# evaluation of potential variables
# ----------------------------------------------------------------------------------------------------------------------
# c_ret consists of the (net/oi) positions and same day return as well as all lagged returns
# current version then takes diff -> this yields the same coefficients
# to do column name
# with diff and holidays there is some date missmatch

# inner join is to merge weekly data (cftc) and daily data (returns) / diff data 
c_ret = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).dropna()
c_ret_diff = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().dropna()


#x1 = ret.iloc[:,0].shift(4)

#todo: naming!!!
#Merge other variables to cf:
cf = c.shift(1) #


midx = pd.MultiIndex(levels=[['ov'], ['cftc_lag1']], codes=[[0],[0]])
cf.columns = midx
 
cf = pd.merge(cf, vol, how='inner', left_index=True, right_index=True).dropna()
cf = pd.merge(cf, vix, how='inner', left_index=True, right_index=True).dropna()
cf.head()


#Todo: multiply vola with the posititon on the day before:
# !!! from here on is copy paste original:
#cf = c.shift(2)
#midx = pd.MultiIndex(levels=[['ov'], ['cftc_lag1']], codes=[[0],[0]])
#cf.columns = midx
#cf = pd.merge(cf, vol, how='inner', left_index=True, right_index=True)
#cf['ov', 'vol'] = cf['ov', 'cftc_lag1'] / cf['ov', 'vol']


#midx = pd.MultiIndex(levels=[['ov'], ['cftc_lag1','vol','log_vix' ]], codes=[[0,0,0],[0,1,2]])
#cf.columns = midx



#cf = cf.iloc[:,1]



#cf['ov', 'cftc_lag1_diff'] = (c.values - c.shift(1).values)[:-1]
#cc_ret: level incl. shifted c
cc_ret = pd.merge(c_ret, cf, how='inner', left_index=True, right_index=True).dropna()
#cc_ret_diff: delta incl diff of c: 
cc_ret_diff = pd.merge(c_ret_diff, cf, how='inner', left_index=True, right_index=True).dropna()
add_var = cf.shape[1]
gamma_ov = np.concatenate((gamma, np.zeros((maxlag+1, add_var))), axis=1)


#set window
window = 52 *5

#smoothing parameter:
alpha = [2.5]


model_idx = c_ret.index[:]
model_clm = pd.MultiIndex.from_product([alpha, ['level', 'diff_'], ['ov', 'dod']])
models = pd.DataFrame(index=model_idx, columns=model_clm)
scores = pd.DataFrame(index=model_idx, columns=model_clm)

#reset index to iter with with index:
# todo: is there a easier way with Dates as index?
models = models.reset_index()
scores = scores.reset_index()
prediction = pd.DataFrame(index=model_idx, columns=model_clm)
prediction = prediction.reset_index()

#lag_idx = c_ret.index[P - 1]


i= window
#Todo: 
for i in range(window,len(c_ret.index[:])):
    print()
#    y = np.concatenate((cc_ret['cftc'].iloc[i-window:i,:].values, np.zeros((maxlag + 1, 1))))
#    X_dod = np.concatenate((cc_ret['ret'].iloc[i-window:i,:], gamma * alpha), axis=0)
    
    # variable for diffs
    y_diff = np.concatenate((cc_ret_diff['cftc'].iloc[i-window:i,:].values, np.zeros((maxlag + 1, 1))))
    X_dod_diff = np.concatenate((cc_ret_diff['ret'].iloc[i-window:i,:],gamma * alpha), axis=0)

    # instruments for ov
#    X_ov = np.concatenate((cc_ret[['ret', 'ov']].iloc[i-window:i,:], gamma_ov * alpha),axis=0)
#    X_ov_diff = np.concatenate((cc_ret_diff[['ret', 'ov']].iloc[i-window:i,:], gamma_ov * alpha), axis=0)
 
    
    #fit the models
#    models.loc[i, (alpha, 'level', 'dod')] = sm.OLS(y,sm.add_constant(X_dod)).fit()
    models.loc[i, (alpha, 'diff_', 'dod')] = sm.OLS(y_diff,X_dod_diff).fit()
#    models.loc[i, (alpha, 'level', 'ov')] = sm.OLS(y,X_ov).fit()
#    models.loc[i, (alpha, 'diff_', 'ov')] = sm.OLS(y_diff,X_ov_diff).fit()
    
#    scores.loc[i, (alpha, 'level', 'dod')] = models.loc[i, (alpha, 'level', 'dod')].get_values()[0].rsquared
    scores.loc[i, (alpha, 'diff_', 'dod')] = models.loc[i, (alpha, 'diff_', 'dod')].get_values()[0].rsquared
#    scores.loc[i, (alpha, 'level', 'ov')] = models.loc[i, (alpha, 'level', 'ov')].get_values()[0].rsquared        
#    scores.loc[i, (alpha, 'diff_', 'ov')] = models.loc[i, (alpha, 'diff_', 'ov')].get_values()[0].rsquared
    
    prediction.loc[i+1, (alpha, 'diff_', 'dod')] = \
    sum(models.loc[i, (alpha, 'diff_', 'dod')].get_values()[0].params * cc_ret_diff['ret'].iloc[i+1,:])

    
    print(str(i)) # + ' number of obs(diff/level): ' + str(models.loc[i, (alpha, 'diff_', 'ov')].get_values()[0].nobs) +\
#          '/' + str(models.loc[i, (alpha, 'level', 'dod')].get_values()[0].nobs))





prediction = prediction.set_index(prediction['Dates'],drop = True)

x = pd.DataFrame({'forecast': (prediction.iloc[:,4]),
                  'emp': cc_ret_diff['cftc','net_specs']}).dropna()
x['forecast'] = x.forecast.astype(float)

plt.figure()
plt.scatter(x['forecast'],x['emp'])

mod5 = smf.ols('emp ~ forecast',x).fit()
print(mod5.summary())


import seaborn as sns; sns.set(color_codes=True)

sns.regplot(x="forecast", y="emp", data=x)

plt.plot(x['emp'])

#returns lbp: 250, fittingperiod 5y
#mod4 = smf.ols('emp ~ forecast',x).fit()
#print(mod2.summary())

#daily returns: mod1 R2: 0.016
#weekly returns: mod2: R2: 0.03

#weekly returns, shift(1): mod3: R2: 0.046
#daily returns, shift(1): mod4:  R2: 0.027
    




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

