# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:22:55 2020

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
# TODO: Try and Except: psycopg2.OperationalError
# TODO: finer grid search
# TODO: Check why QS doesnt work when non_commercials are selected


type_nonc = 'ratio_nonc'
type_mm = 'ratio_mm'


print(os.getcwd())
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu')
from cftc_functions import *

# tcker = pd.read_sql_query("select distinct bb_tkr from cftc.fut_desc where bb_tkr <> 'JO';", engine1)
tcker = pd.read_sql_query("select distinct bb_tkr from cftc.fut_desc where bb_tkr <> 'JO' and bb_tkr <> 'QS';", engine1)

# ---------------------------------------------------------------------------------------
# ------------------------ Get Scaling-Factor -------------------------------------------
# ---------------------------------------------------------------------------------------

alpha_scale_f = [0.2,0.5,1,1.5,2.0,2.5,3.0] # for ratio




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
    df_result= pd.DataFrame(index = list(results), columns = ['OOSR2_1','OOSR2_2'])
    for i in list(results):
        df_result.loc[i,'OOSR2_1'] = dict_result[i]['OOSR2'][0]
        df_result.loc[i,'OOSR2_2'] = dict_result[i]['OOSR2'][1]
    return df_result


optimalscalingfactor = pd.DataFrame(index = tcker.bb_tkr, columns = ['scaling_factor'])

tk = 'KC'
asf = 2

# type_of_exposure = type_;bb_tkr = tk;gammatype = 'dom';alpha = ['ratio'];alpha_scale_factor= asf;  start_dt='2000-01-01';end_dt='2019-12-31'
results_mm = {}
results_nonc = {}
for tk in tcker.bb_tkr[0:1]:
    print(tk)
    for asf in alpha_scale_f:
        try: 
            print(asf)
            results_mm[str(str(asf)+" "+tk)] = calcCFTC(type_of_exposure=type_mm, bb_tkr=tk, gammatype='dom', alpha=['ratio'], alpha_scale_factor=asf, start_dt='2000-01-01', end_dt='2019-12-31')
            results_nonc[str(str(asf)+" "+tk)] = calcCFTC(type_of_exposure=type_nonc, bb_tkr=tk, gammatype='dom', alpha=['ratio'], alpha_scale_factor=asf, start_dt='2000-01-01', end_dt='2019-12-31')
              
        except Exception as error:
            
            print("Oops! An exception has occured:", error)
            print("Exception TYPE:", type(error))
    

# os.getcwd()
# optimalscalingfactor.to_excel(str('scalingFactor_' + type_+'.xlsx'))


#---------------------------------------------------------------------------------------
#-------------------- ♣Plot Betas - Crude Oil------------------------------------------------
#---------------------------------------------------------------------------------------
ab = {}
for asf in alpha_scale_f:
    ab[asf] = calcCFTC(type_of_exposure = type_,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=asf, bb_tkr = 'CL', start_dt='2000-01-01', end_dt='2100-01-01')['last_betas']
    
    plt.figure()
    pd.Series(data = ab[asf]).plot(title = str(asf))


#---------------------------------------------------------------------------------------
#--------------------get R2 with optimal Scaling Factor---------------------------------
#---------------------------------------------------------------------------------------


optimalscalingfactor = pd.read_excel(str('scalingFactor_' + type_+'.xlsx'), index_col = 0)

# TODO: Net non commercials CO - Failure [idx: 10]
results123 = {}
for tk in tcker.bb_tkr[11:]:
    
    print(tk)
    x = optimalscalingfactor.loc[tk][0]
    results123[tk]  = calcCFTC(type_of_exposure = type_,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01')

R2 = getR2(results)

R2.to_excel(str('R2_'+ type_+ '.xlsx'))





#---------------------------------------------------------------------------------------
#--------------------get optimal Gamma Distribution ------------------------------------
#---------------------------------------------------------------------------------------
results1 = {}
results2 = {}
results3 = {}
results4 = {}
results5 = {}
results6 = {}

for tk in tcker.bb_tkr[2:]:
    try:
        print(tk)
        x = optimalscalingfactor.loc[tk][0]
        results1[tk]  = calcCFTC(type_of_exposure = type_,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01')
        results2[tk] = calcCFTC(type_of_exposure = type_,gammatype = 'flat',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01')
        results3[tk] = calcCFTC(type_of_exposure = type_,gammatype = 'linear',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01')
        results4[tk] = calcCFTC(type_of_exposure = type_,gammatype = 'arctan',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01')
        results5[tk] = calcCFTC(type_of_exposure = type_,gammatype = 'log',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01')
        results6[tk] = calcCFTC(type_of_exposure = type_,gammatype = 'sqrt',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01')
        continue
    except  ValueError:
        print('whopsy dasy')
  
R2_functions = pd.DataFrame(index = tcker.bb_tkr , columns = ['dom','flat','linear','arctan','log','sqrt'])

R2_functions.loc[:,'dom'] = getR2(results1).loc[:,'OOSR2_2']
R2_functions.loc[:,'flat'] = getR2(results2).loc[:,'OOSR2_2']
R2_functions.loc[:,'linear'] = getR2(results3).loc[:,'OOSR2_2']
R2_functions.loc[:,'arctan'] = getR2(results4).loc[:,'OOSR2_2']
R2_functions.loc[:,'log'] = getR2(results5).loc[:,'OOSR2_2']
R2_functions.loc[:,'sqrt'] = getR2(results6).loc[:,'OOSR2_2']                            


R2_functions.to_excel(str('R2_' + type_ + 'R2_across_functions.xlsx'))






scalingFactor_nonC = pd.read_excel('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\scalingFactor_net_non_commercials.xlsx')
scalingFactor_nonC = scalingFactor_nonC.set_index('bb_tkr')


# TODO: look at CO (MissingDataError: exog contains inf or nans)
res = {}
for i in scalingFactor_nonC.index[11:]: 
    print(i)
    a = calcCFTC(type_of_exposure = 'net_non_commercials' ,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor= scalingFactor_nonC.loc[i,'scaling_factor'], bb_tkr = i, start_dt='2000-01-01', end_dt='2100-01-01')
    res[i] = a['OOSR2']

df_R2_nonc = pd.DataFrame(index = list(res), columns = ['OOSR2_1','OOSR2_2'])
for i in df_R2_nonc.index:
    print(i)
    df_R2_nonc.loc[i,'OOSR2_1'] = res[i][0]
    df_R2_nonc.loc[i,'OOSR2_2'] = res[i][1]


df_R2_nonc.to_excel('R2_Expo_nonC.xlsx')


a1 = pd.DataFrame([res])
    


