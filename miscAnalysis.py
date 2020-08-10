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


#Gamma function:
gamma_all = {}
maxlag = 250
type_ls = ['flat','dom','linear','arctan','log','sqrt']
    
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
gammas.loc[:,plot()
           

