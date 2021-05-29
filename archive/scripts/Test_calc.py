# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:42:53 2020

@author: grbi
"""
import os
import uuid
import numpy as np
import pandas as pd

import sqlalchemy as sq
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging

m_user = 'dominik'

if m_user == 'dominik':
    path = 'C:\\Users\\bood\\PycharmProjects\\cftc'
else:
    path = 'C:\\Users\\grbi\\PycharmProjects\\cftc_neu'

os.chdir(path)
from cfunctions import *

os.chdir(path + '\\results')
mm_expo_sf = pd.read_excel('mm_exposure.xlsx',sheet_name ='R2_and_scalingFactor',index_col =3)

engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")

multiplier = pd.read_sql(sq.text('select bb_tkr,multiplier from cftc.fut_mult'),engine1)
multiplier = multiplier.set_index('bb_tkr')


results = pd.DataFrame(index = multiplier.index, columns = ['R2','scalingFactor'])
for tk in multiplier.index[4:]:
    print(tk)
    if tk == 'JO':
        continue
    x_mm = mm_expo_sf.loc[tk,'scalingFactor']/ multiplier.loc[tk,'multiplier']
    temp_mm = calcCFTC(type_of_exposure ='net_managed_money' ,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x_mm, bb_tkr = tk, start_dt='2000-12-31', end_dt='2019-12-31',regularization='d1')

    results.loc[tk,'R2'] = temp_mm['OOSR2'].iloc[0,0]
    results.loc[tk,'scalingFactor'] = x_mm
    
    
temp_mm  = calcCFTC(type_of_exposure ='net_managed_money' ,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x_mm, bb_tkr = tk, start_dt='2000-12-31', end_dt='2019-12-31',regularization='d1')


##### Test Exposure ###### 
# engine = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
# c = engine.connect()

# w_net_mm = pd.read_sql(sq.text('select * from cftc.vw_data where px_id = 86'),c)
# w_nonAdj = pd.read_sql(sq.text('select * from cftc.vw_data where px_id = 158'),c)
# w_contract = pd.read_sql(sq.text('select * from cftc.vw_data where px_id = 134'),c)
