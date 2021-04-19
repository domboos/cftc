# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:17:58 2020

@author: grbi
"""
import os 
import pandas as pd

os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu')
from cftc_functions import *

os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\results')


mm_expo_sf = pd.read_excel('mm_exposure.xlsx',sheet_name ='R2_and_scalingFactor',index_col =3)
scalingFactordf = mm_expo_sf
def getbetas_and_R2_Results(scalingFactordf, type_, aalpha_typ,regularization_adj):
    betas = {}
    r2_df = pd.DataFrame(index = scalingFactordf.index, columns =['R2_dom'])
    for tk in scalingFactordf.index:
        try:
            x = scalingFactordf.loc[tk,'scalingFactor']*10
            temp  = calcCFTC(type_of_exposure = type_,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='2000-01-01', end_dt='2100-01-01',regularization=regularization_adj)
            print(tk)
            print(temp['OOSR2'][0])
            r2_df.loc[tk,'R2_dom'] = temp['OOSR2'][0]
            betas[tk] = temp['last_betas']
            
        except Exception as error:
            print ("Oops! An exception has occured:", error)
            print ("Exception TYPE:", type(error))
    
    beta_res = pd.DataFrame.from_dict(betas)
    return r2_df,beta_res


r2, betas = getbetas_and_R2_Results(scalingFactordf= mm_expo_sf, type_='net_managed_money',aalpha_typ= 'stdev',regularization_adj= 'd2_adj')
r2_unadj, betas_unadj = getbetas_and_R2_Results(scalingFactordf= mm_expo_sf, type_='net_managed_money',aalpha_typ= 'stdev',regularization_adj= 'd2_unadj')



