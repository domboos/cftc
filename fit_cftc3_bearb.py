

import pandas as pd
#import mwh_functions as mwh
import numpy as np
import math
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import mwh_functions
#from av import gets
from itertools import islice
import os

os.getcwd()
os.chdir('C:\\Users\\grbi\\Documents\\Paper')

# from pyglmnet todo: check this out

# COMMENT ON THE MODEL
# in the weekly regression the following variables are useful
# return x volume (or x OI) and its first lag, delta volatility x current position (not for net), actual position,
# lagged position change, contemporaneous change in OI (not for net)
## ----------------------------------------------------------------------------------------------------------------------


# todo:
# separate long / short positions and buy sells,
# add volume OI
# clean daily forecast, add to weekly
# normalize daily return forecst with volume or OI
# out of sample evaluation
# for the return the reversal has to be included

# create db engine


# ----------------------------------------------------------------------------------------------------------------------
# set parameters

maxlag = 250
# decay = .9975 is pretty good and pretty reasonable
decay = 0.9975 # gewichtung beobachtung über Zeit

# import and create returns
fut = pd.read_excel('Daten_v1\\Prices.xlsx', index_col=0)

tickers = list(fut)
ind = 2


ret = pd.DataFrame({tickers[ind] : np.log(fut[tickers[ind]]/fut[tickers[ind]].shift(1))})
ret = ret.dropna()

midx = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0],[0]])
ret.columns = midx

# nonC: (l-s)/oi 
nonC = pd.read_excel('Daten_v1\\Net_NonC.xlsx', index_col=0)
c = pd.DataFrame({'pos': nonC[tickers[ind]]})

c.head()

# change in open interest 
ov = pd.DataFrame({'net': c['pos']})
ov['doi'] = ov.net - ov.net.shift(1)

vol = 100 * ret.ewm(alpha=0.1).std()
midx = pd.MultiIndex(levels=[['ov'], ['vol']], codes=[[0],[0]])
vol.columns = midx

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
#gamma[maxlag, maxlag] = 1
gamma[maxlag, maxlag] = 0


#plt.figure('gamma')
#plt.plot(np.diagonal(gamma))
#plt.title('gamma Coefficients')

# querying cftc data
midx = pd.MultiIndex(levels=[['cftc'], ['net_specs']], codes=[[0], [0]])
c.columns = midx


# ----------------------------------------------------------------------------------------------------------------------
# evaluation of potential variables
# ----------------------------------------------------------------------------------------------------------------------
# c_ret consists of the (net/oi) positions and same day return as well as all lagged returns
# current version then takes diff -> this yields the same coefficients
# to do column name
# with diff and holidays there is some date missmatch

# inner join is to merge weekly data (cftc) and daily data (returns) 

c_ret = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).dropna()

#
#shift price data from monday to monday (Traders can't use the information from tuesday yet.)
#c_ret_diff = pd.merge(c.diff(), ret.iloc[:, :-1].diff(5).shift(1), how='inner', left_index=True, right_index=True).dropna()
c_ret_diff = pd.merge(c, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().dropna()




cf = c.shift(1) #

midx = pd.MultiIndex(levels=[['ov'], ['cftc_lag1']], codes=[[0],[0]])
cf.columns = midx

cf.head()
cf = pd.merge(cf, vol, how='inner', left_index=True, right_index=True)





#cf = cf.iloc[:,1]



#cf['ov', 'cftc_lag1_diff'] = (c.values - c.shift(1).values)[:-1]
#cc_ret: level incl. shifted c
cc_ret = pd.merge(c_ret, cf, how='inner', left_index=True, right_index=True).dropna()
#cc_ret_diff: delta incl diff of c: 
cc_ret_diff = pd.merge(c_ret_diff, cf, how='inner', left_index=True, right_index=True).dropna()
add_var = 2
gamma_ov = np.concatenate((gamma, np.zeros((maxlag+1, add_var))), axis=1)


# ov
retFac = np.fromfunction(lambda i, j: decay ** i, cc_ret.values.shape)[::-1]
cc_ret_decay = cc_ret*retFac

retFac2 = np.fromfunction(lambda i, j: decay ** i, cc_ret_diff.values.shape)[::-1]
cc_ret_diff_decay = cc_ret_diff*retFac2


# A is the weigth given to parameter smoothing, this value has to be fitted and it interacts with the decay. The decay
# reduces the average weight.
# mean absolute error regression might be better, maybe we get a stepfunction for the parameters!!!
#A = [0.8 ,1, 2.5, 5]
A = [2.5]
diff_adj = 0.2

# define the first fitting period and lagged index
P = 270
model_idx = c_ret.index[P:]
model_clm = pd.MultiIndex.from_product([A, ['level', 'diff_'], ['ov', 'dod']])
models = pd.DataFrame(index=model_idx, columns=model_clm)
scores = pd.DataFrame(index=model_idx, columns=model_clm)
models_ = pd.DataFrame(index=model_idx, columns=['lag_idx'])
models_['time_weight'] = retFac[P:, 1]
lag_idx = c_ret.index[P - 1]

for idx, row in islice(c_ret.iterrows(), P, None):
    for alpha in A:
        # if model_type == 'dod':
        y = np.concatenate((cc_ret_decay['cftc'].loc[:lag_idx, :].values, np.zeros((maxlag + 1, 1))))
        X_dod = np.concatenate((cc_ret_decay['ret'].loc[:lag_idx],
                            gamma * alpha * models_.loc[idx, 'time_weight']), axis=0)
        # variable for diffs
        y_diff = np.concatenate((cc_ret_diff_decay['cftc'].loc[:lag_idx, :].values, np.zeros((maxlag + 1, 1))))
        X_dod_diff = np.concatenate((cc_ret_diff_decay['ret'].loc[:lag_idx],
                                 gamma * alpha * diff_adj * models_.loc[idx, 'time_weight']), axis=0)

        # instruments for ov
        X_ov = np.concatenate((cc_ret_decay[['ret', 'ov']].loc[:lag_idx],
                            gamma_ov * alpha * models_.loc[idx, 'time_weight']),axis=0)
        X_ov_diff = np.concatenate((cc_ret_diff_decay[['ret', 'ov']].loc[:lag_idx],
                                 gamma_ov * alpha * diff_adj * models_.loc[idx, 'time_weight']), axis=0)

        models.loc[idx, (alpha, 'level', 'dod')] = linear_model.LinearRegression().fit(X_dod, y)
        models.loc[idx, (alpha, 'diff_', 'dod')] = linear_model.LinearRegression().fit(X_dod_diff, y_diff)
        models.loc[idx, (alpha, 'level', 'ov')] = linear_model.LinearRegression().fit(X_ov, y)
        models.loc[idx, (alpha, 'diff_', 'ov')] = linear_model.LinearRegression().fit(X_ov_diff, y_diff)

        scores.loc[idx, (alpha, 'level', 'dod')] \
            = models.loc[idx, (alpha, 'level', 'dod')].score(cc_ret_decay[['ret']].loc[:lag_idx],
                                                      cc_ret_decay['cftc'].loc[:lag_idx, :].values)
        scores.loc[idx, (alpha, 'diff_', 'dod')] \
            = models.loc[idx, (alpha, 'diff_', 'dod')].score(cc_ret_diff_decay[['ret']].loc[:lag_idx],
                                                      cc_ret_diff_decay['cftc'].loc[:lag_idx, :].values)
        scores.loc[idx, (alpha, 'level', 'ov')] \
            = models.loc[idx, (alpha, 'level', 'ov')].score(cc_ret_decay[['ret', 'ov']].loc[:lag_idx],
                                                      cc_ret_decay['cftc'].loc[:lag_idx, :].values)
        scores.loc[idx, (alpha, 'diff_', 'ov')] \
            = models.loc[idx, (alpha, 'diff_', 'ov')].score(cc_ret_diff_decay[['ret', 'ov']].loc[:lag_idx],
                                                      cc_ret_diff_decay['cftc'].loc[:lag_idx, :].values)

    print(str(idx) + ' calculated')
    lag_idx = idx
    models_.loc[idx, 'lag_idx'] = lag_idx

# ----------------------------------------------------------------------------------------------------------------------
col = [5, 10, 20, 35, 100, 200, 250]
alpha = A[0]
coefA = pd.DataFrame(index=model_idx, columns=col)
for idx, row in models.iterrows():
    coefA.loc[idx, :] = row[(alpha, 'diff_', 'dod')].coef_[0, col]

tickers[ind]

plt.figure('Coefficients')
plt.plot(coefA)
plt.title('some loadings for the dod diff model, market:' + str(tickers[ind]))
plt.legend(coefA.columns, loc='best')



plt.figure('score')
plt.plot(scores)
plt.title('R-squared, market:' + str(tickers[ind]))
plt.legend(scores.columns, loc='best')

# ----------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
fig.suptitle('Comparing Models')
mat1 = 20
mat2 = 150
col = ['level, alpha = ' + str(alpha), 'diff_, alpha = ' + str(alpha)]
coef1 = pd.DataFrame(index=model_idx, columns=col)
coef2 = pd.DataFrame(index=model_idx, columns=col)

for idx, row in models.iterrows():
    coef1.loc[idx, 'level'] = row[(alpha, 'level', 'dod')].coef_[0, mat1]
    coef1.loc[idx, 'diff_'] = row[(alpha, 'diff_', 'dod')].coef_[0, mat1]
    coef2.loc[idx, 'level'] = row[(alpha, 'level', 'dod')].coef_[0, mat2]
    coef2.loc[idx, 'diff_'] = row[(alpha, 'diff_', 'dod')].coef_[0, mat2]

axs[0].plot(coef1)
axs[1].plot(coef2)
axs[0].legend(model_clm)
axs[1].legend(model_clm)
axs[0].set_title(str(mat1) + ' days')
axs[1].set_title(str(mat2) + ' days')


# ----------------------------------------------------------------------------------------------------------------------
plt.figure('hallo ')
# plt.plot(models.loc[model_idx[-1], (alpha, 'level')].coef_[0, :])
# plt.plot(models.loc[model_idx[-1], (alpha, 'diff_')].coef_[0, :])
print(models.loc[model_idx[-1], (alpha, 'level', 'dod')])
print(models.loc[model_idx[-1], (alpha, 'level', 'dod')].coef_)
print(models.loc[model_idx[-1], (alpha, 'diff_', 'dod')].coef_)
plt.plot(models.loc[model_idx[0], (alpha, 'level', 'dod')].coef_[0])
plt.plot(models.loc[model_idx[-1], (alpha, 'diff_', 'dod')].coef_[0])
plt.title('Coeff for ' + str(model_idx[-1]))
plt.legend(col)


# ----------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharey=True)
fig.suptitle('Comparing different alphas')
for alpha in A:
    # axs[0].plot(models.loc[model_idx[-1], (alpha, 'level')].coef_[0, :])
    # axs[1].plot(models.loc[model_idx[-1], (alpha, 'diff_')].coef_[0, :])
    print(models.loc[model_idx[-1], (alpha, 'level', 'dod')].coef_[0])
    axs[0, 0].plot(models.loc[model_idx[-1], (alpha, 'level', 'dod')].coef_[0, :])
    axs[0, 1].plot(models.loc[model_idx[-1], (alpha, 'diff_', 'dod')].coef_[0, :])
    axs[1, 0].plot(models.loc[model_idx[-1], (alpha, 'level', 'ov')].coef_[0, :])
    axs[1, 1].plot(models.loc[model_idx[-1], (alpha, 'diff_', 'ov')].coef_[0, :])

axs[0, 0].legend(A)
axs[0, 1].legend(A)
axs[1, 0].legend(A)
axs[1, 1].legend(A)
axs[0, 0].set_title('Level Model - returns')
axs[0, 1].set_title('Diff Model - returns')
axs[1, 0].set_title('Level Model - all')
axs[1, 1].set_title('Diff Model - all')
#
#for idx, row in ret.iterrows():

tickers[ind]


plt.show()
plt.figure()
plt.plot(nonC['CC1'])
plt.plot(nonC['SB1'])
plt.plot(nonC['CL1'])
plt.plot(nonC['GC1'])
plt.legend(col)
