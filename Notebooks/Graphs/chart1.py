#%%
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/40734672/how-to-set-the-label-fonts-as-time-new-roman-by-drawparallels-in-python
plt.rcParams["font.family"] = "Times New Roman"

import seaborn as sns
from datetime import datetime

import pandas as pd

file = './data/Chart_Trend_Signals.xlsx'
file = 'C:\\Users\\grbi\\switchdrive\\Tracking Traders\\04_Abbildungen\\Chart_Trend_Signals.xlsx'
data_chart_1 = pd.read_excel(file, sheet_name='mom', index_col=0)
print(data_chart_1)

data_chart_2 = pd.read_excel(file, sheet_name='maX', index_col=0)
print(data_chart_2)



fig, (s1,s2) = plt.subplots(1,2, sharex=False, sharey= False ,figsize=(7,3))
fig.tight_layout()
plt.subplots_adjust(top=0.98, bottom=0.18, left=0.1, right=0.98, hspace=0.3,
                        wspace=0.3)
sns.set(font_scale = 1)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})

sns.lineplot(x = data_chart_1.index,y = data_chart_1.iloc[:, 0]*100 ,ax =s1, linewidth = 2, legend = 'brief',color ="black",label = 'mom(50)' )
sns.lineplot(x = data_chart_1.index,y = data_chart_1.iloc[:, 1]*100 ,ax =s1, linewidth = 2, legend = 'brief',color ="dimgrey", label = 'mom(120)')
sns.lineplot(x = data_chart_1.index,y = data_chart_1.iloc[:, 2]*100 ,ax =s1, linewidth = 2, legend = 'brief',color ="lightgrey", label = 'smom(60,5)' )
s1.lines[2].set_linestyle("--")
s1.set(yticks=np.array([0.0,0.5,1.0,1.5,2.0]))
s1.legend(frameon=False,fontsize=7)

sns.lineplot(x = data_chart_2.index,y = data_chart_2.iloc[:, 0]*100 ,ax =s2, linewidth = 2, legend = 'brief', color="black",label='ma(100)')
sns.lineplot(x = data_chart_2.index,y = data_chart_2.iloc[:, 1]*100, ax = s2, linewidth=2,legend ='brief', color="dimgrey", label='xmac(0.8,0.96)')
sns.lineplot(x =data_chart_2.index,y =  data_chart_2.iloc[:, 2]*100,ax = s2, linewidth= 2,legend ='brief', color="lightgrey", label='mac(5,60)')
s2.lines[0].set_linestyle("--")
s2.lines[2].set_linestyle(":")
s2.legend(frameon = False,fontsize=7)
s2.set(yticks=np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0]))
s1.set_xlabel('Lag') ;s1.set_ylabel('Weight (%)')
s2.set_xlabel('Lag') ;s2.set_ylabel('Weight (%)')


# plt.savefig(f"./reports/figures/signal.png",dpi=100)
plt.savefig(f"C:/Users/grbi/PycharmProjects/cftc_neu/Notebooks/Graphs/temp/F1_ReturnSignPlot.png",dpi=100)
plt.show()
