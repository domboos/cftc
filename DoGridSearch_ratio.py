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

alpha_scale_f = [0.5,1.0,1.5,2.0,2.5,3.0] # for ratio



# type_of_exposure = type_;bb_tkr = tk;gammatype = 'dom';alpha = ['ratio'];alpha_scale_factor= asf;  start_dt='2000-01-01';end_dt='2019-12-31'
results_mm = {}
results_nonc = {}
for tk in tcker.bb_tkr:
    print(tk)
    for asf in alpha_scale_f:
        try: 
            print(asf)
            results_mm[str(tk+" "+str(asf))] = calcCFTC(type_of_exposure=type_mm, bb_tkr=tk, gammatype='dom', alpha=['const'], alpha_scale_factor=asf, start_dt='2000-01-01', end_dt='2019-12-31')
            results_nonc[str(tk+" "+str(asf))]= calcCFTC(type_of_exposure=type_nonc, bb_tkr=tk, gammatype='dom', alpha=['const'], alpha_scale_factor=asf, start_dt='2000-01-01', end_dt='2019-12-31')
              
        except Exception as error:
            
            print("Oops! An exception has occured:", error)
            print("Exception TYPE:", type(error))
  
result_dict = results_mm

def getOOSResults(result_dict):
    keyds = list(result_dict)
    dicty = result_dict
    result = pd.DataFrame(columns = ['key','R2'])
    
    
    for tk in tcker.bb_tkr:
        temp = pd.DataFrame(columns = ['key','R2'])
        for el in keyds:
            if el.startswith(tk):
                # print(el)
                temp = temp.append(pd.DataFrame({'key':el ,'R2': dicty[el]['OOSR2'][0]}, index = [0]), ignore_index = True)
        result = result.append(temp[temp.R2 == temp.R2.max()], ignore_index = True)
    
    
    for i in result.index:
        if len(result.loc[i,'key']) == 3:
            
            result.loc[i,'scalingFactor'] = (result.loc[i,'key'][-1]).replace(" ", "")
            result.loc[i,'bb_tkr'] = (result.loc[i,'key'][0]).replace(" ", "")
        elif len(result.loc[i,'key']) == 4:
            result.loc[i,'scalingFactor'] = (result.loc[i,'key'][-2:]).replace(" ", "")
            result.loc[i,'bb_tkr'] = (result.loc[i,'key'][:2]).replace(" ", "")
        else:
            result.loc[i,'scalingFactor'] = (result.loc[i,'key'][-3:]).replace(" ", "")
            result.loc[i,'bb_tkr'] = (result.loc[i,'key'][0:3]).replace(" ", "")
    # result = result.drop(['key'], axis = 1)
    



    insample_mean_scores = pd.DataFrame(index=result.key.values, columns=['level','diff'])
    for i in insample_mean_scores.index:
        insample_mean_scores.loc[i,'level'] = dicty[keyds[0]]['scores-insample'].mean().values[0]
        insample_mean_scores.loc[i,'diff']=dicty[keyds[0]]['scores-insample'].mean().values[1]
    
    betas = pd.DataFrame(columns = result.key.values)
    for col in betas.columns:
        betas[col] = dicty[keyds[0]]['last_betas']
        
    return result,insample_mean_scores,betas
    

mm_OOSR2, mm_insample_mean_scores,mm_betas = getOOSResults(results_mm)
nonc_OOSR2, nonc_insample_scores, nonc_betas = getOOSResults(results_nonc)


writer = pd.ExcelWriter('mm_ratio.xlsx',engine='xlsxwriter')
mm_OOSR2.to_excel(writer,sheet_name='R2_and_scalingFactor')
mm_insample_mean_scores.to_excel(writer,sheet_name='insample_scores')
mm_betas.to_excel(writer,sheet_name = 'betas')
writer.save()

writer = pd.ExcelWriter('nonc_ratio.xlsx',engine='xlsxwriter')
nonc_OOSR2.to_excel(writer,sheet_name='R2_and_scalingFactor')
nonc_insample_scores.to_excel(writer,sheet_name='insample_scores')
nonc_betas .to_excel(writer,sheet_name = 'betas')
writer.save()


del mm_OOSR2; mm_insample_mean_scores;mm_betas; nonc_OOSR2; nonc_insample_scores; nonc_betas

#---------------------------------------------------------------------------------------------

