# -*- coding: utf-8 -*-
"""
Created on Tue April 20, 2022

@author: grbi, dominik
"""

# klar

import numpy as np
import sqlalchemy as sq
import statsmodels.api as sm
from datetime import datetime
import scipy.optimize as op
from cfunctions import *

#
model_id = 82

# crate engine
from cengine import cftc_engine
engine1 = cftc_engine()
print(engine1)

# speed up db
from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

import pandas as pd

#h = gets(engine1, type, data_tab='forecast', desc_tab='cot_desc', series_id=None, bb_tkr=None, bb_ykey='COMDTY',
#         start_dt='1900-01-01', end_dt='2100-01-01', constr=None, adjustment=None)
bb = pd.read_sql_query("select bb_tkr from cftc.order_of_things", engine1)
print(bb)
bb2 = bb.copy()
trader = 'long_pump'

for idx, tkr in bb.iterrows():
    print(tkr.bb_tkr)
    forecast1 = pd.read_sql_query("select FC.px_date, FC.qty as mom_pos_change from cftc.forecast FC inner join cftc.model_desc " +
                                 " MD ON FC.model_id = MD.model_id where MD.model_type_id = 82 and MD.bb_tkr = '"
                                 + str(tkr.bb_tkr) + "' order by FC.px_date ",
                                 engine1, index_col='px_date')

    forecast2 = pd.read_sql_query("select FC.px_date, FC.qty as prod_pos_change from cftc.forecast FC inner join cftc.model_desc " +
                                 " MD ON FC.model_id = MD.model_id where MD.model_type_id = 149 and MD.bb_tkr = '"
                                 + str(tkr.bb_tkr) + "' order by FC.px_date ",
                                 engine1, index_col='px_date')

    hh = pd.merge(forecast1, forecast2, on='px_date')
    print(hh)
    bb2.loc[idx, 'corr_MOM'] = hh['mom_pos_change'].corr(hh['prod_pos_change'])
    bb2.loc[idx, 'std_mom'] = hh['mom_pos_change'].std()
    bb2.loc[idx, 'std_prod'] = hh['prod_pos_change'].std()
    bb2.loc[idx, 'ratio'] = hh['mom_pos_change'].std() / hh['prod_pos_change'].std()


print(bb2)
# px = gets(engine1, type='px_last', desc_tab='fut_desc', data_tab='data', bb_tkr='C', adjustment='none')



