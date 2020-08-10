# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:07:03 2020

@author: grbi
"""


import os
import numpy as np
import pandas as pd
import sqlalchemy as sq
import statsmodels.api as sm
import statsmodels.formula.api as smf
import timeit
import matplotlib.pyplot as plt

# TODO: Check following warning on QS: C:\Users\grbi\Anaconda3\lib\site-packages\pandas\core\reshape\merge.py:618: UserWarning: merging between different levels can give an unintended result (2 levels on the left, 3 on the right)   warnings.warn(msg, UserWarning)
# TODO: Plot Betas by smoothing
# TODO: Check if smoothing ~ Open interest makes sense
# TODO: finer grid search - maybe at later stage
# TODO: Check why QS doesnt work when non_commercials are selected


type_ = 'net_managed_money' # 'net_managed_money',net_non_commercials



def getR2(dict_result):
    """
    Parameters
    ----------
    dict_result: dictinoary
        from the following Loop

    Returns
    -------
    R2
    """
    df_result = pd.DataFrame(index = list(results), columns = ['OOSR2_1','OOSR2_2'])
    for i in list(results):
        df_result.loc[i,'OOSR2_1'] = dict_result[i]['OOSR2'][0]
        df_result.loc[i,'OOSR2_2'] = dict_result[i]['OOSR2'][1]
    return df_result


print(os.getcwd())
os.chdir('C:\\Users\\bood\\PycharmProjects\\cftc_neu')
from cftc_functions import *

# tcker = pd.read_sql_query("select distinct bb_tkr from cftc.fut_desc where bb_tkr <> 'JO';", engine1)
tcker = pd.read_sql_query("select distinct bb_tkr from cftc.fut_desc where bb_tkr <> 'JO' --and bb_tkr <> 'QS';", engine1)

#---------------------------------------------------------------------------------------
#------------------------ Get Scaling-Factor -------------------------------------------
#---------------------------------------------------------------------------------------
#Runtime ~ gefühlt: 4h

alpha_scale_f = [0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001] # for exposure like net_non_commercials


optimalscalingfactor = pd.DataFrame(index = tcker.bb_tkr, columns = ['scaling_factor'])

# type_of_exposure = type_;bb_tkr = tk;gammatype = 'dom';alpha = ['ratio'];alpha_scale_factor= asf;  start_dt='2000-01-01';end_dt='2019-12-31'
results_= {}
for tk in tcker.bb_tkr:
    print(tk)
    for asf in alpha_scale_f:
        try: 
            print(asf)
            results_[str(tk +" " + str(asf))]  = calcCFTC(type_of_exposure = type_,bb_tkr = tk , gammatype = 'dom',alpha = ['ratio'],alpha_scale_factor= asf,  start_dt='1900-01-01', end_dt='2019-12-31')
              
        except Exception as error:
            
            print ("Oops! An exception has occured:", error)
            print ("Exception TYPE:", type(error))
    
    
    master_mm[tk] = results_nonc
    a = getR2(results)
    a[a.OOSR2_2 == a.OOSR2_2.max()].index
    print(a)
    optimalscalingfactor.loc[tk] = a[a.OOSR2_2 == a.OOSR2_2.max()].index.values[0]

os.getcwd()    
optimalscalingfactor.to_excel(str('scalingFactor_' + type_+'.xlsx'))