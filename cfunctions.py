# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:10:00 2020

@author: grbi
"""
import os
import numpy as np
import pandas as pd
import sqlalchemy as sq
import statsmodels.api as sm
from datetime import datetime
import scipy.optimize as op
from dotenv import load_dotenv

# todo: if error is thrown do in Terminal: pip install python-dotenv
load_dotenv()

try:
    conn = os.getenv('DATABASE_URL')
    engine1 = sq.create_engine(conn)
except Exception as e:
    print('Connection to db could not be established, because DATABASE_URL is not defined')
    print(e)

# crate engine


# speed up db
from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

# functions
# -------------------------------------------------------------------
def gets(engine1, type, data_tab='data', desc_tab='cot_desc', series_id=None, bb_tkr=None, bb_ykey='COMDTY',
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
        print(desc_tab)
        print(type)
        series_id = str(series_id.values[0][0])
    else:
        series_id = str(series_id)

    h_1 = " WHERE px_date >= '" + str(start_dt) + "' AND px_date <= '" + str(end_dt) + "' AND px_id = "
    h_2 = series_id + constr + " order by px_date"
    fut = pd.read_sql_query('SELECT px_date, qty FROM cftc.' + data_tab + h_1 + h_2, engine1, index_col='px_date')
    return fut


def getexposure(engine, type_of_trader, norm, bb_tkr, start_dt='1900-01-01', end_dt='2100-01-01', bb_ykey='COMDTY'):
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

    pos = gets(engine, type=type_of_trader, data_tab='vw_data', bb_tkr=bb_tkr, bb_ykey=bb_ykey,
               start_dt=start_dt, end_dt=end_dt, adjustment=None)  # constr=constr,

    if norm == 'percent_oi':
        oi = gets(engine, type='agg_open_interest', data_tab='vw_data', desc_tab='cot_desc', bb_tkr=bb_tkr,
                  bb_ykey=bb_ykey, start_dt=start_dt, end_dt=end_dt)
        pos_temp = pd.merge(left=pos, right=oi, how='left', left_index=True, right_index=True,
                            suffixes=('_pos', '_oi'))
        exposure = pd.DataFrame(index=pos_temp.index, data=(pos_temp.qty_pos / pos_temp.qty_oi), columns=['qty'])

    elif norm == 'exposure':
        price_non_adj = gets(engine, type='contract_size', desc_tab='fut_desc', data_tab='vw_data', bb_tkr=bb_tkr,
                             bb_ykey=bb_ykey, start_dt=start_dt, end_dt=end_dt, adjustment='none')
        df_merge = pd.merge(left=pos, right=price_non_adj, left_index=True, right_index=True, how='left')

        exposure = pd.DataFrame(index=df_merge.index)
        exposure['qty'] = (df_merge.qty_y * df_merge.qty_x).values

    elif norm == 'number':
        exposure = pos.copy()
        print('--- NUMBER ---')

    midx = pd.MultiIndex(levels=[['cftc'], [type_of_trader]], codes=[[0], [0]])
    exposure.columns = midx

    exposure.dropna(inplace=True)
    # print(exposure.to_string())

    return exposure


####------------------------------------------------------------------------------
####-------------------------Gamma and Returns------------------------------------
####------------------------------------------------------------------------------

def getGamma(maxlag, regularization='d1', gammatype='sqrt', gammapara=1, naildownvalue0=0, naildownvalue1=0):
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

    if gammatype == 'loo':
        gammatype_tmp = 'loo'
        gammatype = 'flat'
    else:
        gammatype_tmp = 'xxx'

    # loop creates gamma
    for i in range(0, t):

        if gammatype == 'dom':
            gamma[i, i] = 1 - 1 / (i + 1) ** gammapara

        elif gammatype == 'flat':
            gamma[i, i] = 1

        elif gammatype == 'flat1':
            if i>0:
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
            # original is 1
            if gammapara is None:
                gammapara = 1
            gamma[i, i] = np.sqrt(i + gammapara)

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
        gamma[maxlag - 2, maxlag - 2] = naildownvalue1
        gamma[maxlag - 2, maxlag-1] = -naildownvalue1

    # nail_down and delete zero rows
    gamma[gamma.shape[0] - 1, gamma.shape[1] - 1] = naildownvalue0
    gamma = np.delete(gamma, np.where(~gamma.any(axis=1))[0], axis=0)

    if gammatype_tmp == 'loo':
        gamma = np.delete(gamma, gammapara-1, 0)

    return gamma


def getRetMat(_ret, maxlag, prefix='ret'):
    """
    Parameters
    ----------
    ret_ : pd.DataFrame()
        log return series
    maxlag : int
        DESCRIPTION.

    """

    # loop creates lagged returns in ret
    for i in range(0, maxlag):
        _ret[prefix, str(i + 1).zfill(3)] = _ret[prefix, '000'].shift(i + 1)

    _ret = _ret.iloc[maxlag:, :maxlag+1]  # delete the rows with nan due to its shift.
    return _ret


def getAlpha(alpha_type, y, x=None, gma=None, start=None):
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
    elif alpha_type == 'loocv':
        press1 = lambda z, z1=y.values, z2=x, z3=gma: press(z, z1, z2, z3) * 100000000
        res = op.minimize(press1, x0=start, method='BFGS', tol=0.0001, options={'disp': True,
                        'maxiter': 20, 'gtol': 0.00001, 'eps': 0.0001})
        alpha = res.x
    elif alpha_type == 'gcv':
        gcv1 = lambda z, z1=y.values, z2=x, z3=gma: gcv(z, z1, z2, z3)
        res = op.minimize(gcv1, x0=start, method='Nelder-Mead', options={'disp': True,
                        'maxiter': 500, 'xatol': 0.01, 'fatol': 0.1})
        alpha = res.x
    else:
        alpha = 1
    print(alpha)

    return alpha


def getGammaOpt(y, x=None, gma1=None, gma2=None, start=None):
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
    #bounds=bnds,
    press1 = lambda a1, z1=y.values, z2=x, z3=gma1, z4=gma2: press_2(a1, z1, z2, z3, z4)
    res = op.minimize(press_2, args=(y.values, x, gma1, gma2), x0=start, method='powell')

    alpha = np.squeeze(res.x)

    print(alpha)

    return alpha[0]*gma1+alpha[1]*gma2, alpha


def merge_pos_ret(pos, ret, diff):
    if diff:
        cr = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().dropna()
    else:   #level
        print('---------------')
        cr = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).dropna()
    return cr

# press and it's variants
def gcv(a, _y, _x, g):
    n = np.shape(_y)[0]
    iH = np.identity(n) - _x @ np.linalg.inv(np.transpose(_x) @ _x + a * a * np.transpose(g) @ g) @ np.transpose(_x)
    iHy = iH @ _y
    tiH = np.trace(iH)
    return 0.000001 * np.transpose(iHy) @ iHy / (tiH * tiH)


def press(a, _y, _x, g):
    n = np.shape(_y)[0]
    iH = np.identity(n) - _x @ np.linalg.inv(np.transpose(_x) @ _x + a * a * np.transpose(g) @ g) @ np.transpose(_x)
    B = np.diag(1 / np.diag(iH))
    BiHy = B @ iH @ _y
    k = 0.000000000001 * np.transpose(BiHy) @ BiHy / n
    return k[0][0]


def press_2(a1, _y, _x, g1, g2):
    # A is an array
    # G are Gamma matrices
    a1 = np.squeeze(a1)
    print(a1.shape)

    g = a1[0] * g1 + a1[1] * g2
    print(a1)
    n = np.shape(_y)[0]
    iH = np.identity(n) - _x @ np.linalg.inv(np.transpose(_x) @ _x + np.transpose(g) @ g) @ np.transpose(_x)
    B = np.diag(1 / np.diag(iH))
    BiHy = B @ iH @ _y
    return 0.000001 * np.transpose(BiHy) @ BiHy / n


def getBeta(engine, model_id = None, model_type_id=None, bb_tkr=None, bb_ykey='COMDTY',
            start_dt='1900-01-01', end_dt='2100-01-01', constr=None, expand=False, drop_beta_date=True):

    if constr is None:
        constr = ''
    else:
        constr = ' AND ' + constr

    if model_id is None:
        model_id = pd.read_sql_query("SELECT model_id FROM cftc.model_desc WHERE bb_tkr = '" + bb_tkr +
                                     "' AND bb_ykey = '" + bb_ykey + "' AND model_type_id = '" + str(model_type_id) +
                                     "'", engine1)
        model_id = str(model_id.values[0][0])
    else:
        model_id = str(model_id)

    h_1 = " WHERE px_date >= '" + str(start_dt) + "' AND px_date <= '" + str(end_dt) + "'"
    h_1b = " WHERE beta_date >= '" + str(start_dt) + "' AND beta_date <= '" + str(end_dt) + "'"
    h_2 = " AND model_id = " + model_id + constr + " order by px_date"
    beta0 = pd.read_sql_query('SELECT px_date, return_lag, qty FROM cftc.beta' + h_1 + h_2, engine)
    beta = beta0.pivot(index='px_date', columns='return_lag', values='qty')

    if expand:
        beta.index.rename(name='beta_date', inplace=True)
        dates = pd.read_sql_query('SELECT * FROM cftc.beta_date' + h_1b, engine)
        beta = pd.merge(left=dates, right=beta, on='beta_date', how='left').set_index(['px_date', 'beta_date']).dropna()
        if drop_beta_date:
            beta.index = beta.index.droplevel('beta_date')
    return beta, model_id