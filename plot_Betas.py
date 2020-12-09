# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 02:11:00 2020

@author: Linus Grob
"""

import os
# os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc')
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\results')

import numpy as np
import statsmodels.api as sm
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import sqlalchemy as sq
engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")


def getYearEndDates(betas_stacked):
    """
    Parameters
    ----------
    betas_stacked : pd.DataFrame()
        betas of a specific Model --> Raw Version

    Returns
    -------
    list_yearEndDate : list()
        list of year end dates for Plots.
    """
    dates = pd.DataFrame( index = betas_stacked.reset_index().px_date.drop_duplicates().sort_values())
    dates['year'] = dates.index.year
    list_yearEndDate = list(dates['year'].drop_duplicates(keep = 'last').index)
    return list_yearEndDate



#get all models which are already calculated
model_list = pd.read_sql_query("SELECT * FROM cftc.vw_model_desc ORDER BY bb_tkr, bb_ykey", engine1)
temp = model_list[model_list.bb_tkr == 'PL']
#TODO: Define model_id for the plot
model_id = 764 # FC 
model_id = 963 # FC 
model_id = 202 # PL
model_id = 733 # PL
model_id = 1050

#Get Data and transform DataTypes (easier handling later on)
betas_stacked = pd.read_sql_query(str("SELECT px_date,return_lag,qty FROM cftc.beta where model_id = " + str(model_id)),engine1)
betas_stacked.px_date = betas_stacked.px_date.astype('datetime64[ns]')
betas_stacked.return_lag = betas_stacked.return_lag.astype('int') 
betas_stacked.qty = betas_stacked.qty.astype('float') 


uniqueDates = betas_stacked.px_date.drop_duplicates().reset_index(drop = True)

beta2 = betas_stacked.copy()
beta2 = beta2.set_index(['px_date','return_lag'])
beta2 = beta2.unstack()
betas = pd.DataFrame(index = uniqueDates, data= beta2.values) 
betas = betas.rename(columns={x:y for x,y in zip(betas.columns,range(1,len(betas.columns)+1))})
del beta2

yearEndDates = getYearEndDates(betas_stacked)
temp_betas = betas[betas.index.isin(yearEndDates)]

plt.figure(figsize = (15,8))
plt.title(str("Average Betas of "))
sns.set(font_scale = 1.5)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
sns.lineplot(data = temp_betas.T, dashes =False)
sns.despine()

plt.show()








