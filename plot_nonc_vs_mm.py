# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:30:12 2020

@author: grbi
"""
import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\results')
r2 = pd.read_excel('mm_vs_nonc.xlsx', sheet_name = 'r2_comparison')
r2 = r2.sort_values(['Sector','bb_tkr'])





plt.figure(figsize = (15,8))
sns.set(font_scale = 1.5)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
sns.barplot(x="bb_tkr", y="R2", hue="type", data=r2, palette = "Blues_d").set(xlabel = 'Commodity Ticker',ylabel = 'Coefficent of Determination')
sns.despine()
plt.legend(frameon = False, loc = 9)
    


r2[r2.type == 'Non Commercials'].mean()
r2[r2.type == 'Managed Money'].mean()


