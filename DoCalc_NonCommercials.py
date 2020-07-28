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


type_ = 'ratio_mm' # 'net_managed_money',net_non_commercials



print(os.getcwd())
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu')
from cftc_functions import *

# tcker = pd.read_sql_query("select distinct bb_tkr from cftc.fut_desc where bb_tkr <> 'JO';", engine1)
tcker = pd.read_sql_query("select distinct bb_tkr from cftc.fut_desc where bb_tkr <> 'JO' and bb_tkr <> 'QS';", engine1)

#---------------------------------------------------------------------------------------
#------------------------ Grid Search (Get Scaling-Factor) -----------------------------
#---------------------------------------------------------------------------------------
#Runtime ~ gefühlt: 4h

# alpha_scale_f = [0.00001,0.000001,0.0000001,0.00000001,0.000000001] # for exposure like net_non_commercials

alpha_scale_f = [1,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0] # for ratio



# type_of_exposure = type_;bb_tkr = tk;gammatype = 'dom';alpha = ['ratio'];alpha_scale_factor= asf;  start_dt='2000-01-01';end_dt='2019-12-31'
results_mm = {}
results_nonc = {}
for tk in tcker.bb_tkr:
    print(tk)
    for asf in alpha_scale_f:
        try: 
            print(asf)
            results_mm[str(str(asf)+ " "  + tk)]  = calcCFTC(type_of_exposure = 'ratio_mm',bb_tkr = tk,gammatype = 'dom',alpha = ['ratio'],alpha_scale_factor= asf,  start_dt='2000-01-01', end_dt='2019-12-31')  
            
            results_nonc[str(str(asf)+ " "  + tk)]  = calcCFTC(type_of_exposure = 'ratio_nonc',bb_tkr = tk,gammatype = 'dom',alpha = ['ratio'],alpha_scale_factor= asf,  start_dt='2000-01-01', end_dt='2019-12-31')
              
        except Exception as error:
            
            print ("Oops! An exception has occured:", error)
            print ("Exception TYPE:", type(error))
    
  
#---------------------------------------------------------------------------------------
#------------------------ Get R2 and Scaling-Factor ------------------------------------
#---------------------------------------------------------------------------------------
keys_mm = list(results_mm)
results_R2_mm = pd.DataFrame(columns = ['key','R2'])

for tk in tcker.bb_tkr:
    temp = pd.DataFrame(columns = ['key','R2'])

    for el in keys_mm:
        if el.endswith(tk):
            temp = temp.append(pd.DataFrame({'key':el ,'R2': results_mm[el]['OOSR2'][1]}, index = [0]), ignore_index = True)
            
            # print(temp[temp.R2 == temp.R2.max()])
    results_R2_mm= results_R2_mm.append(temp[temp.R2 == temp.R2.max()], ignore_index = True)



keys_nonc = list(results_nonc)
results_R2_nonc = pd.DataFrame(columns = ['key','R2'])

for tk in tcker.bb_tkr:

    temp = pd.DataFrame(columns = ['key','R2'])

    for el in keys_nonc:
        if el.endswith(tk):
            temp = temp.append(pd.DataFrame({'key':el ,'R2': results_nonc[el]['OOSR2'][1]}, index = [0]), ignore_index = True)
            
            # print(temp[temp.R2 == temp.R2.max()])
    results_R2_nonc= results_R2_nonc.append(temp[temp.R2 == temp.R2.max()], ignore_index = True)


for i in results_R2_mm.index:
    results_R2_mm.loc[i,'scalingFactor'] = results_R2_mm.loc[i,'key'][0:3]
    results_R2_mm.loc[i,'bb_tkr'] = results_R2_mm.loc[i,'key'][-2:].replace(" ", "")


r2_mm = results_R2_mm.loc[:,['bb_tkr','R2']]
scaling_factor_ratio_mm = results_R2_mm.loc[:,['bb_tkr','scalingFactor']]


for i in results_R2_nonc.index:
    results_R2_nonc.loc[i,'scalingFactor'] = results_R2_nonc.loc[i,'key'][0:3]
    results_R2_nonc.loc[i,'bb_tkr'] = results_R2_nonc.loc[i,'key'][-2:].replace(" ", "")
    
r2_nonc_ratio = results_R2_nonc.loc[:,['bb_tkr','R2']]
scaling_factor_ratio_nonc = results_R2_nonc.loc[:,['bb_tkr','scalingFactor']]


r2_nonc_ratio.to_excel('r2_nonc_ratio.xlsx',index = False)
scaling_factor_ratio_nonc.to_excel('scaling_factor_ratio_nonc.xlsx',index = False)
r2_mm.to_excel('r2_mm_ratio.xlsx',index = False)
scaling_factor_ratio_mm.to_excel('scaling_factor_ratio_mm.xlsx', index = False)
    



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








    


