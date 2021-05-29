"""
Created on Mon Aug 24 22:02:46 2020

@author: grbi
"""
import os 
import pandas as pd
import numpy as np 

os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu')
from cfunctions import *

os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\results')


mm_expo_sf = pd.read_excel('mm_exposure.xlsx',sheet_name ='R2_and_scalingFactor',index_col =3)
nonc_expo_sf = pd.read_excel('nonc_exposure.xlsx',sheet_name ='R2_and_scalingFactor',index_col =3)


dates = pd.read_sql_query("SELECT MIN(d.px_date) px_date ,MAX(f.bb_tkr) bb_tkr, MAX(f.cot_type) cot_type FROM cftc.vw_data d RIGHT JOIN cftc.cot_desc f ON d.px_id = f.cot_id WHERE cot_type <> 'agg_open_interest' GROUP BY d.px_id;", engine1)
da


def getbetas_and_R2_Results_same_dates(tickerlist,nonc_df,mm_df,regularization_adj, dates = dates):
    betas = {}
    r2_df = pd.DataFrame(index =tickerlist, columns =['R2_nonc','R2_mm'])
    for tk in tickerlist:
        try:
            dat = dates[dates.bb_tkr == tk]
            dat = dat[dat.cot_type == 'net_managed_money'].px_date.values[0]
            dat = dat.strftime('%Y-%m-%d')
            
            if regularization_adj !='d1':
                x_mm = mm_df.loc[tk,'scalingFactor']*10
                x_nonc = nonc_df.loc[tk,'scalingFactor']*10
            else:
                x_mm = mm_df.loc[tk,'scalingFactor']
                x_nonc = nonc_df.loc[tk,'scalingFactor']
            temp_mm  = calcCFTC(type_of_exposure ='net_managed_money' ,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x_mm, bb_tkr = tk, start_dt=dat, end_dt='2019-12-31',regularization=regularization_adj)
            temp_nonc = calcCFTC(type_of_exposure ='net_non_commercials' ,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x_nonc, bb_tkr = tk, start_dt=dat, end_dt='2019-12-31',regularization=regularization_adj)
            print(tk)

            r2_df.loc[tk,'R2_nonc'] = temp_nonc['OOSR2'][0]
            r2_df.loc[tk,'R2_mm'] = temp_mm['OOSR2'][0]
            
            betas[str(str(tk)+' mm')] = temp_mm['last_betas']
            betas[str(str(tk)+' nonc')] = temp_nonc['last_betas']
                
        except Exception as error:
            print ("Oops! An exception has occured:", error)
            print ("Exception TYPE:", type(error))
    
    beta_res = pd.DataFrame.from_dict(betas)
    return r2_df,beta_res




r2_d1, betas_d1 = getbetas_and_R2_Results_same_dates(tickerlist=nonc_expo_sf.index, nonc_df=nonc_expo_sf, mm_df=mm_expo_sf, regularization_adj='d1', dates=dates)
r2_d2_unadj, betas_d2_unadj = getbetas_and_R2_Results_same_dates(tickerlist=nonc_expo_sf.index, nonc_df=nonc_expo_sf, mm_df=mm_expo_sf, regularization_adj='d2_unadj', dates=dates)
r2_d2_adj, betas_d2_adj = getbetas_and_R2_Results_same_dates(tickerlist=nonc_expo_sf.index, nonc_df=nonc_expo_sf, mm_df=mm_expo_sf, regularization_adj='d2_adj', dates=dates)



