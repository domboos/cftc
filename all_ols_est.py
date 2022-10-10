# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:10:00 2020

@author: grbi, dominik
"""
import pickle
import datetime
import numpy as np
import statsmodels.api as sm
from cfunctions import engine1,gets, getexposure,getRetMat,merge_pos_ret


# crate engine


# speed up db
from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

import pandas as pd

# MAIN

bb = pd.read_sql_query("select bb_tkr from cftc.order_of_things", engine1)
print(bb)

for idx, tkr in bb.iterrows():
    # feching and structure returns
    print(datetime.now().strftime("%H:%M:%S"))
    bb_ykey = 'COMDTY'
    fut = gets(engine1, type='px_last', desc_tab='fut_desc', data_tab='data', bb_tkr=tkr.bb_tkr, adjustment='by_ratio')
    # calc rets:
    ret_series = pd.DataFrame(index=fut.index)
    ret_series.loc[:, 'ret'] = np.log(fut / fut.shift(1))
    ret_series = ret_series.dropna()  # deletes first value
    ret_series.columns = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0], [0]])

    # fecthing cot, crate lagged returns and merge
    pos = getexposure(engine1, type_of_trader='net_non_commercials', norm='percent_oi', bb_tkr=tkr.bb_tkr,
                      bb_ykey='COMDTY', start_dt='1998-01-19')
    ret = getRetMat(ret_series, 260)
    cr = merge_pos_ret(pos, ret, True)
    # hi

    if idx == 0:
        CRR = cr
    else:
        CRR = pd.concat([CRR, cr], ignore_index=False)

print(CRR)

x = CRR['ret'].values
y = CRR['cftc']

model_fit = sm.OLS(y, x).fit()

print(model_fit.summary())

res = {'data_all':CRR,
       'x':x,
       'y':y,
       'ols_model':model_fit}

with open('pooled_ols_result.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)