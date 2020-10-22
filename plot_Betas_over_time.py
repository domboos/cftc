"""
Created on Tue Aug 25 19:19:11 2020

@author: grbi
"""



import os 
import pandas as pd
import numpy as np 
import seaborn as sns


os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu')
from cftc_functions_nightly_betas import *

os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\results')
nonc_expo_sf = pd.read_excel('nonc_exposure.xlsx',sheet_name ='R2_and_scalingFactor',index_col =3)

def getbetas_over_time(tickerlist,df,regularization_adj):
    betas = {}
    r2= {}
    
    
    for tk in tickerlist[0:1]:
        # print(tk)
        # tk = 'W'
        if regularization_adj !='d1':
            x = df.loc[tk,'scalingFactor']*10
        else:
            x = df.loc[tk,'scalingFactor']
        try:
            temp  = calcCFTC(type_of_exposure ='net_non_commercials' ,gammatype = 'dom',alpha = ['stdev'],alpha_scale_factor=x, bb_tkr = tk, start_dt='1900-01-01', end_dt='2019-12-31',regularization=regularization_adj)
            betas[tk] = temp['insample_betas']
            r2[tk] = temp['OOSR2']
      
        except Exception as error:
            print ("Oops! An exception has occured:", error)
            print ("Exception TYPE:", type(error))  
    return betas,r2
       


betas_d1,r2_d1 = getbetas_over_time(tickerlist = nonc_expo_sf.index,df = nonc_expo_sf, regularization_adj = 'd1')
# betas_d2_adj,r2_d2_adj = getbetas_over_time(tickerlist = nonc_expo_sf.index,df = nonc_expo_sf, regularization_adj = 'd2_adj')


keys = list(betas_d1)

for i in list(betas_d1):
    print(i)
    betas = pd.DataFrame.from_dict(betas_d1[i])
    
    w = pd.DataFrame(columns = ['mean','mean+stdev','mean-stdev'])
    w['mean'] = betas.mean(axis =1)
    w['mean+stdev'] = betas.mean(axis =1) + betas.std(axis = 1)
    w['mean-stdev'] = betas.mean(axis =1) - betas.std(axis = 1)
    
    
    #Todo: insert title

    plt.figure(figsize = (15,8))
    plt.title(str("Average Betas of "+ str(i)))
    sns.set(font_scale = 1.5)
    sns.set_style('white')
    sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.lineplot(data = w,palette = ['red','blue','blue'], dashes =False,legend = False)
    sns.despine()

betas_W
betas.columns.count()

for i in list(betas_d1):
    betas = pd.DataFrame.from_dict(betas_d1[i])
    a = betas.iloc[:,range(0,betas.shape[1],100)]
    
    plt.figure(figsize = (15,8))
    plt.title(str("Average Betas of "+ str(i)))
    sns.set(font_scale = 1.5)
    sns.set_style('white')
    sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.lineplot(data = a, dashes =False)
    sns.despine()









# df_mean = df.mean(axis = 1)
# df_std = df.std(axis = 1)
# df_m_std = df_mean-df_std
# df_p_std = df_mean+df_std

# import matplotlib.pyplot as plt
# plt.plot(range(0,len(df_mean)),df_mean, label = 'mean')
# plt.plot(range(0,len(df_p_std)),df_p_std,label = 'stdev1')
# plt.plot(range(0,len(df_m_std)),df_m_std,label = 'stdev1')


# #betas across Time:
# plt.plot(df.iloc[250,:])


# print(df.iloc[5,:].max()-df.iloc[210,:].min())

# import datetime

# test_df['Date'].dt.strftime("%Y%m%d").astype(int)
# a2 = pd.DataFrame(df.columns,columns = ['Date']).values.strftime("%Y%m%d").astype(int)
# a2['Date']
# x = list(df.columns).datettime.strftime("%Y%m%d").astype(int)
# y = df.index
# X,Y = np.meshgrid(x,y)
# Z = df
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)