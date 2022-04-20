# -*- coding: utf-8 -*-
"""
Created on Tue April 20, 2022

@author: grbi, dominik
"""

import numpy as np
import sqlalchemy as sq
import statsmodels.api as sm
from datetime import datetime
import scipy.optimize as op
from cfunctions import *

# crate engine
from cengine import cftc_engine
engine1 = cftc_engine()

# speed up db
from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

import pandas as pd

h = gets(engine1, type, data_tab='forecast', desc_tab='cot_desc', series_id=None, bb_tkr=None, bb_ykey='COMDTY',
         start_dt='1900-01-01', end_dt='2100-01-01', constr=None, adjustment=None)