# -*- coding: utf-8 -*-
#%%
"""
Created on Wed Nov  4 02:11:00 2020

@author: Linus Grob
"""

import os
# os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc')
#os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\results')

import numpy as np
import statsmodels.api as sm
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import sqlalchemy as sq
engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
#%%
#get all models which are already calculated
#TODO: Define model_id for the plot
# Query: select * from cftc.model_desc where model_type_id = 82 and bb_tkr in ('W','CL','SB','GC');
# Model Type ID: 
# "2019"	"SB"
# "2023"	"CL"
# "2031"	"GC"
# "2039"	"W"
model_ids =[2019,2023,2031,2039]
px_dates = ['1999-12-28','2004-12-28','2009-12-29','2014-12-30','2019-12-31']
titles = ['Sugar','WTI','Gold','Chicago Wheat']

# %%
#*define Layout
color = ['lime', 'cyan','deepskyblue','blueviolet','crimson'] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
fig, axs = plt.subplots(2, 2, sharex=True, sharey= False ,figsize=(15,10))
fig.tight_layout()
# fig.suptitle(f"All Betas with model_id: {model_typeNonc},{model_typeMM}",fontsize=30)
# fig.subplots_adjust(top=0.95)

sns.set(font_scale = 1.5)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})


# DO Plots:
plot_matrix = np.arange(4).reshape(2,2)
for col in range(len(plot_matrix[0])):
    # print(f"Row: {row}")print(f"Row: {row}")
    for row in range(len(plot_matrix)):
        print(f"col: {col}");print(f"Row: {row}")

        model_id = model_ids[plot_matrix[row][col]]
        print(model_id)

        ax_curr = axs[row,col]
        
        for idx in range(len(px_dates)):
            beta = pd.read_sql_query(f"select * from cftc.beta where model_id = {model_id} and px_date = '{px_dates[idx]}'",engine1)
            sns.lineplot(x = beta.return_lag,y = beta.qty ,ax =ax_curr, linewidth = 3, legend = False,color =color[idx])
        
        ax_curr.axhline(0, ls='--', color ='black')
        # ax_curr.yaxis.set_major_formatter(FormatStrFormatter('%e'))
        title = str(titles[plot_matrix[row][col]])
        ax_curr.set_title(title)
        ax_curr.set_xlabel('')
        ax_curr.set_ylabel('')
        if plot_matrix[row][col] == 3:
            ax_curr.legend(px_dates, frameon = False)
        plt.savefig(f"Betas_over_Time.png",dpi=100)
        
        




#%%

beta = pd.read_sql_query(f"select * from cftc.beta where model_id = {model_ids[0]} and px_date = '{px_dates[0]}'",engine1)