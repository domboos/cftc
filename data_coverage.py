# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:49:19 2020

@author: grbi
"""

#Import Packages:
# import statsmodels.formula.api as smf
# import statsmodels.api as sm
import numpy as np
import pandas as pd
import datetime as dt
import os
import matplotlib.pyplot as plt
import sqlalchemy as sq
import ctypes
import io 
import psycopg2 
import time
import datetime as dt


engine = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
c = engine.connect()


query = """
SELECT * FROM patents.vw_price_and_patent_static_weekly 
    WHERE country_iso = 'US' and crncy = 'USD'
    AND (CAST(gics_code AS text) like '35%' 
    OR CAST(gics_code AS text) like '45%')
    --LIMIT 1
"""

q_Dat= """
SELECT zhaw_id
,MAX(px_date) max_date_fin
,min(px_date) min_date_fin
,MAX(ps_reporting_date) max_date_pat
,min(ps_reporting_date) min_date_pat
FROM patents.vw_price_and_patent_static_weekly 
    WHERE country_iso = 'US' and crncy = 'USD'
    AND (CAST(gics_code AS text) like '35%' 
    OR CAST(gics_code AS text) like '45%')
group by zhaw_id
"""



df_raw = pd.read_sql(sq.text(query_pat_daten),c)
df_dat = pd.read_sql(sq.text(q_Dat),c)


df_dat['max_date_fin'] = pd.to_datetime(df_dat['max_date_fin']).dt.date
df_dat['max_date_pat'] = pd.to_datetime(df_dat['max_date_pat']).dt.date
df_dat['min_date_fin'] = pd.to_datetime(df_dat['min_date_fin']).dt.date
df_dat['min_date_pat'] = pd.to_datetime(df_dat['min_date_pat']).dt.date
df_dat['diff_date_pat'] = (df_dat['max_date_pat'] - df_dat['min_date_pat'])
df_dat['diff_date_pat_days'] = df_dat['diff_date_pat'].apply(lambda x: x.days)
df_dat['diff_date_fin'] = (df_dat['max_date_fin'] - df_dat['min_date_fin'])
df_dat['diff_date_fin_days'] = df_dat['diff_date_fin'].apply(lambda x: x.days)


df_final_ids = df_dat[(df_dat.diff_date_pat_days > 260) & (df_dat.diff_date_fin_days > 260)].zhaw_id
df_sample = df_raw[df_raw.zhaw_id.isin(list(df_final_ids))]

comp_by_gics = df_sample.drop_duplicates(subset= ['zhaw_id']).groupby(['gics_code','gics_name']).zhaw_id.count()





df_dat.dtypes



a = df_raw.groupby('zhaw_id').count().sum()