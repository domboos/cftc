# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:10:00 2020

@author: grbi
"""

import numpy as np
import sqlalchemy as sq
import statsmodels.api as sm
from cfunctions import *
from datetime import datetime

from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

import pandas as pd

engine1 = sq.create_engine(
    "postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")


# MAIN

# refreshing model view and fetching
conn = engine1.connect()
conn.execute('REFRESH MATERIALIZED VIEW cftc.vw_model_desc')
model_list = pd.read_sql_query("SELECT * FROM cftc.vw_model_desc WHERE max_date IS NULL ORDER BY bb_tkr, bb_ykey",
                               engine1)

for idx, model in model_list.iterrows():
    # feching and structure returns
    print(datetime.now().strftime("%H:%M:%S"))
    if idx == 0 or (bb_tkr != model.bb_tkr or bb_ykey != model.bb_ykey):
        bb_tkr = model.bb_tkr
        bb_ykey = model.bb_ykey
        fut = gets(engine1, type='px_last', desc_tab='fut_desc', data_tab='data', bb_tkr=bb_tkr, adjustment='by_ratio')
        # calc rets:
        ret_series = pd.DataFrame(index=fut.index)
        ret_series.loc[:, 'ret'] = np.log(fut / fut.shift(1))
        ret_series = ret_series.dropna()  # deletes first value
        ret_series.columns = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0], [0]])

    # decay
    window = model.est_window
    lags = np.arange(1, model.lookback+1)
    beta = pd.DataFrame(columns={'model_id', 'px_date', 'return_lag', 'qty'})
    beta['return_lag'] = lags
    beta['model_id'] = model.model_id
    fcast = pd.DataFrame(data=[model.model_id], columns={'model_id'})
    if model.decay is not None:
        retFac = np.fromfunction(lambda i, j: model.decay ** i, [window, model.lookback])[::-1]

    # gamma
    gamma = getGamma(maxlag=model.lookback, regularization=model.regularization, gammatype=model.gamma_type,
                     gammapara=model.gamma_para, naildownvalue0=model.naildown_value0,
                     naildownvalue1=model.naildown_value1)

    # fecthing cot, crate lagged returns and merge
    pos = getexposure(type_of_trader=model.cot_type, norm=model.cot_norm, bb_tkr=bb_tkr, bb_ykey=bb_ykey)
    ret = getRetMat(ret_series, model.lookback)
    cr = merge_pos_ret(pos, ret, model.diff)

    for idx2, day in enumerate(cr.index[0:-(window + model.fit_lag)]):

        # rolling window parameters:
        w_start = cr.index[idx2]
        w_end = cr.index[idx2 + window]
        # welcher wert????
        forecast_period = cr.index[idx2 + window + model.fit_lag]  # includes the day x in [:x]

        if model.decay is not None:
            x0 = cr['ret'].loc[w_start:w_end, :].values * retFac
            y0 = cr['cftc'].loc[w_start:w_end, :] * retFac[:, 1] # not tested
        else:
            x0 = cr['ret'].loc[w_start:w_end, :].values
            y0 = cr['cftc'].loc[w_start:w_end, :]

        alpha = getAlpha(alpha_type=model.alpha_type, y=y0) * model.alpha

        y = np.concatenate((y0, np.zeros((gamma.shape[0], 1))))
        x = np.concatenate((x0, gamma * alpha), axis=0)

        ##  fit the models
        model_fit = sm.OLS(y, x).fit()

        beta.qty = model_fit.params
        beta.px_date = forecast_period
        fcast['qty'] = model_fit.predict(cr['ret'].loc[forecast_period, :].values)
        fcast['px_date'] = forecast_period
        if idx2 == 0:
            beta_all = beta.copy()
            fcast_all = fcast.copy()
        else:
            beta_all = beta_all.append(beta, ignore_index=True)
            fcast_all = fcast_all.append(fcast, ignore_index=True)

    beta_all.to_sql('beta', engine1, schema='cftc', if_exists='append', index=False)
    fcast_all.to_sql('forecast', engine1, schema='cftc', if_exists='append', index=False)
    print('---')