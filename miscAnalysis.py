# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:12:34 2020

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

ga
#Gamma function:
gamma_all = {}
maxlag = 250
type_ls = ['flat','dom','linear','arctan','log','sqrt']
    

def kth_diag_indices(a, k):
      rows, cols = np.diag_indices_from(a)
      if k < 0:
          return rows[-k:], cols[:k]
      elif k > 0:
          return rows[:-k], cols[k:]
      else:
          return rows, cols
        

   
for gammatype in type_ls:
    gamma = np.zeros((maxlag + 1, maxlag + 1))
            
    # loop creates gamma and lagged returns in ret
    # ret = ret_series.iloc[0:10,:]
    for i in range(0, maxlag + 1):
        
        if gammatype == 'dom':
            gamma[i, i] = 1 - 1 / (i+1)**(1/4)
        
        elif gammatype == 'flat':
            gamma[i, i] = 1
        
        elif gammatype == 'linear':
            gamma[i, i] = (i+1)/(maxlag+1)
            gamma[0,0] = 0
            gamma[0,1] = 0
            
        elif gammatype == 'arctan':
            gamma[i,i] = np.arctan(0.2*i)
        
        elif gammatype == 'log':
            gamma[i,i] = np.log(1+ 5*i/maxlag)
 
        elif gammatype == 'sqrt':
            gamma[i,i] = np.sqrt(i+1)/15
            
        if  i < maxlag:
            gamma[i, i + 1] = - gamma[i, i]

    # gmax = gamma.max()
    
    # rows, cols = kth_diag_indices(gamma,1)
    # gamma[rows,cols] /=  gmax
    
    rowsm1, colsm1 = kth_diag_indices(gamma,-1)
    gamma[rowsm1,colsm1] =-gamma[rowsm1,colsm1+1]
    
    gamma[np.diag_indices_from(gamma)] = gamma[np.diag_indices_from(gamma)]*2
    gamma /= gamma.max()
     
    #naildown last value: f
    gamma[maxlag,maxlag-1] = -1
    gamma[maxlag,maxlag] = 1
    
    gamma = np.vstack([gamma, np.zeros(maxlag+1)])
    gamma[-1,-1] = 1
    # gamma = gamma[:-1,:]
    
    gamma = gamma[1:,:]
    
    gammas = pd.DataFrame.from_dict(gamma_all,orient = 'columns')
    
    




a = gamma

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

gmax = gamma.max()
gamma[np.diag_indices_from(gamma)] /= gmax


rows1, cols1 = kth_diag_indices(gamma,1)
rows, cols =  kth_diag_indices(gamma,-1)
gamma[rows,cols] = gamma[rows1,cols1]
rows_diag, cols_diag =kth_diag_indices(gamma,0)

gamma[rows_diag,cols_diag] = gamma[rows_diag,cols_diag]*2

gamma= gamma /2





#Appendix: 
    
for gammatype in type_ls:
    gamma = np.zeros((maxlag + 1, maxlag + 1))
    for i in range(0, maxlag + 1):
       
        if gammatype == 'dom':
            gamma[i, i] = 1 - 1 / (i+1)**(1/4)
        
        elif gammatype == 'flat':
            gamma[i, i] = 1
        
        elif gammatype == 'linear':
            gamma[i, i] = (i+1)/(maxlag+1)
            
        elif gammatype == 'arctan':
            gamma[i,i] = np.arctan(0.2*i)
        
        elif gammatype == 'log':
            gamma[i,i] = np.log(1+ 5*i/maxlag)
     
        elif gammatype == 'sqrt':
            gamma[i,i] = np.sqrt(i+1)/15
            
        if  i < maxlag:
            gamma[i, i + 1] = - gamma[i, i]
    gamma_all[gammatype] = gamma.diagonal() / gamma.max()
    
gammas = pd.DataFrame.from_dict(gamma_all,orient = 'columns')

gammas.plot()           

