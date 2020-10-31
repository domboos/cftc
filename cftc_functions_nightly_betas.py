# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:10:00 2020

@author: grbi
"""

import numpy as np
import sqlalchemy as sq
import statsmodels.api as sm
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


def gets(engine, type, data_tab='data', desc_tab='cot_desc', series_id=None, bb_tkr=None, bb_ykey='COMDTY',
         start_dt='1900-01-01', end_dt='2100-01-01', constr=None, adjustment=None):
    if constr is None:
        constr = ''
    else:
        constr = ' AND ' + constr

    if series_id is None:
        if desc_tab == 'cot_desc':
            series_id = pd.read_sql_query("SELECT cot_id FROM cftc.cot_desc WHERE bb_tkr = '" + bb_tkr +
                                          "' AND bb_ykey = '" + bb_ykey + "' AND cot_type = '" + type + "'", engine1)
        else:
            series_id = pd.read_sql_query("SELECT px_id FROM cftc.fut_desc WHERE bb_tkr = '" + bb_tkr +
                                          "' AND adjustment= '" + adjustment + "' AND bb_ykey = '" + bb_ykey +
                                          "' AND data_type = '" + type + "'", engine1)

        series_id = str(series_id.values[0][0])
    else:
        series_id = str(series_id)

    h_1 = " WHERE px_date >= '" + str(start_dt) + "' AND px_date <= '" + str(end_dt) + "' AND px_id = "
    h_2 = series_id + constr + " order by px_date"
    fut = pd.read_sql_query('SELECT px_date, qty FROM cftc.' + data_tab + h_1 + h_2, engine, index_col='px_date')
    return fut


def getexposure(type_of_trader, norm, bb_tkr, start_dt='1900-01-01', end_dt='2100-01-01', bb_ykey='COMDTY'):
    """
    Parameters
    ----------
    type_of_exposure : str()
        one of: 'net_managed_money','net_non_commercials','ratio_mm','ratio_nonc'
    bb_tkr : TYPE
        Ticker from the commofity; example 'KC'
    start_dt : str(), optional
        The default is '1900-01-01'.
    end_dt :  str(), optional
        The default is '2100-01-01'.
    bb_ykey :  str(), optional
        The default is 'COMDTY'.

    Returns
    -------
    exposure : pd.DataFrame() with Multiindex (cftc,net_specs)
        Returns the exposure of the underlying position in USD (net_pos * fut_price * (Multiplier(?)) )


    """
    # Note:
    # - Exposure = mult * fut_adj_none * net_pos
    # - contract_size =  mult * fut_adj_none

    pos = gets(engine1, type=type_of_trader, data_tab='vw_data', bb_tkr=bb_tkr, bb_ykey=bb_ykey,
               start_dt=start_dt, end_dt=end_dt, adjustment=None)  # constr=constr,

    if norm == 'percent_oi':
        oi = gets(engine1, type='agg_open_interest', data_tab='vw_data', desc_tab='cot_desc', bb_tkr=bb_tkr,
                  bb_ykey=bb_ykey, start_dt=start_dt, end_dt=end_dt)
        pos_temp = pd.merge(left=pos, right=oi, how='left', left_index=True, right_index=True,
                            suffixes=('_pos', '_oi'))
        exposure = pd.DataFrame(index=pos_temp.index, data=(pos_temp.qty_pos / pos_temp.qty_oi), columns=['qty'])

    elif norm == 'exposure':
        price_non_adj = gets(engine1, type='contract_size', desc_tab='fut_desc', data_tab='vw_data', bb_tkr=bb_tkr,
                             bb_ykey=bb_ykey, start_dt=start_dt, end_dt=end_dt, adjustment='none')
        df_merge = pd.merge(left=pos, right=price_non_adj, left_index=True, right_index=True, how='left')

        exposure = pd.DataFrame(index=df_merge.index)
        exposure['qty'] = (df_merge.qty_y * df_merge.qty_x).values

    else:
        print('wrong type_of_exposure')

    midx = pd.MultiIndex(levels=[['cftc'], ['net_specs']], codes=[[0], [0]])
    exposure.columns = midx

    return exposure


####------------------------------------------------------------------------------
####-------------------------Gamma and Returns------------------------------------
####------------------------------------------------------------------------------

def getGamma(maxlag, regularization='d1', gammatype='sqrt', gammapara=1, naildownvalue0=1, naildownvalue1=1):
    """
    Parameters
    ----------
    ret : pd.DataFrame()
        log return series
    gammatype :  (str):    'flat','linear','dom','arctan','log','sqrt'
        How to calc the gamma_function (one of:'flat','linear','dom','arctan','log')
    maxlag : int
        DESCRIPTION.

    """

    def kth_diag_indices(a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols

    gamma = np.zeros((maxlag, maxlag))

    t = maxlag
    if regularization == 'd2':
        t -= 1

    # loop creates gamma
    for i in range(0, t):

        if gammatype == 'dom':
            gamma[i, i] = 1 - 1 / (i + 1) ** gammapara

        elif gammatype == 'flat':
            gamma[i, i] = 1

        elif gammatype == 'linear1':
            if i>0:
                gamma[i, i] = (i + 1) / (maxlag + 1)

        elif gammatype == 'linear':
            gamma[i, i] = (i + 1) / (maxlag + 1)

        elif gammatype == 'arctan':
            gamma[i, i] = np.arctan(gammapara * i)

        elif gammatype == 'log':
            gamma[i, i] = np.log(1 + gammapara * i / maxlag)

        elif gammatype == 'sqrt':
            gamma[i, i] = np.sqrt(i + 1)

    # standardize sum of diagonal values to 1
    gsum = gamma.diagonal(0).sum()
    gamma[np.diag_indices_from(gamma)] /= gsum

    # default case
    rows, cols = kth_diag_indices(gamma, 1)
    gamma[rows, cols] = -gamma.diagonal()[:-1]
    # naildown

    if regularization == 'd2':

        gamma[np.diag_indices_from(gamma)] /= 2

        rowsm1, colsm1 = kth_diag_indices(gamma, 2)
        gamma[rowsm1, colsm1] = gamma.diagonal()[:-2]

        # fade out:
        gamma[maxlag - 1, maxlag - 1] = naildownvalue1
        gamma[maxlag - 1, maxlag] = -naildownvalue1

    # nail_down and delete zero rows
    gamma[gamma.shape[0] - 1, gamma.shape[1] - 1] = naildownvalue0
    gamma = np.delete(gamma, np.where(~gamma.any(axis=1))[0], axis=0)

    return gamma


def getRetMat(ret, maxlag):
    """
    Parameters
    ----------
    ret : pd.DataFrame()
        log return series
    maxlag : int
        DESCRIPTION.

    """

    # loop creates lagged returns in ret
    for i in range(0, maxlag):
        ret['ret', str(i + 1).zfill(3)] = ret['ret', '000'].shift(i + 1)

    ret = ret.iloc[maxlag:, :]  # delete the rows with nan due to its shift.
    return ret


def getAlpha(alpha_type, y):
    """
    Parameters
    ----------
    alpha_type : str()
        either 'stdev',var'
    y : np vector
        independent variable

    Returns
    -------
    alpha : value
        scaling factor for gamma matrix

    """
    if alpha_type == 'std':
        alpha = y.std()[0]

    elif alpha_type == 'var':
        alpha = y.var()[0]

    else:
        alpha = 1

    return alpha


def merge_pos_ret(pos, ret, diff):
    if diff:
        cr = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().dropna()
    else:   #level
        cr = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).dropna()
    return cr


# MAIN

model_list = loadedData = pd.read_sql_query("SELECT * FROM cftc.model_desc ORDER BY bb_tkr, bb_ykey", engine1)

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
                     naildownvalue1=model.naildown_value0)

    # fecthing cot, crate lagged returns and merge
    pos = getexposure(type_of_trader=model.cot_type, norm=model.cot_norm, bb_tkr=bb_tkr, bb_ykey=bb_ykey)
    ret = getRetMat(ret_series, model.lookback)
    cr = merge_pos_ret(pos, ret, model.diff)

    for idx2, day in enumerate(cr.index[0:-(window + 2)]):

        # rolling window parameters:
        w_start = cr.index[idx2]
        w_end = cr.index[idx2 + window]
        forecast_period = cr.index[idx2 + window + 2]  # includes the day x in [:x]

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

    print(beta_all)
    print(fcast_all)
    beta_all.to_sql('beta', engine1, schema='cftc', if_exists='append', index=False)
    fcast_all.to_sql('forecast', engine1, schema='cftc', if_exists='append', index=False)
    print('---')