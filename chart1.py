#%%
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import pandas as pd

file = 'C:\\Users\\grbi\\switchdrive\\Tracking Traders\\04_Abbildungen\\Chart_Trend_Signals.xlsx'
data_chart_1 = pd.read_excel(file, sheet_name='mom', index_col=0)
print(data_chart_1)

data_chart_2 = pd.read_excel(file, sheet_name='maX', index_col=0)
print(data_chart_2)



color = ['crimson', 'cyan'] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
fig, (s1,s2) = plt.subplots(1,2, sharex=False, sharey= False ,figsize=(15,4))
fig.tight_layout()

sns.set(font_scale = 1.5)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})

sns.lineplot(x = data_chart_1.index,y = data_chart_1.iloc[:, 0]*100 ,ax =s1, linewidth = 3, legend = 'brief',color ="#53777a",label = 'mom(50)' )
sns.lineplot(x = data_chart_1.index,y = data_chart_1.iloc[:, 1]*100 ,ax =s1, linewidth = 3, legend = 'brief',color ="#c02942", label = 'mom(120)')
sns.lineplot(x = data_chart_1.index,y = data_chart_1.iloc[:, 2]*100 ,ax =s1, linewidth = 3, legend = 'brief',color ="#d95b43", label = 'smom(60,5)' )
s1.lines[2].set_linestyle("--")
s1.legend(frameon=False)

sns.lineplot(x = data_chart_2.index,y = data_chart_2.iloc[:, 0]*100 ,ax =s2, linewidth = 3, legend = 'brief', color="#53777a",label='ma(100)')
sns.lineplot(x = data_chart_2.index,y = data_chart_2.iloc[:, 1]*100, ax = s2, linewidth=3,legend ='brief', color="#c02942", label='mac(5,60)')
sns.lineplot(x =data_chart_2.index,y =  data_chart_2.iloc[:, 2]*100,ax = s2, linewidth=3,legend ='brief', color="#d95b43", label='xmac(0.8,0.96)')
s2.lines[0].set_linestyle("--")
s2.lines[2].set_linestyle(":")
s2.legend(frameon = False)
s1.set_xlabel('') ;s1.set_ylabel('')
s2.set_xlabel('') ;s2.set_ylabel('')

plt.savefig(f"./signal.png",dpi=100)


#%% Old

# from bokeh.io import show, output_file
# from bokeh.models import ColumnDataSource, FactorRange
# from bokeh.plotting import figure
# from bokeh.layouts import gridplot
# from bokeh.layouts import row
# import pandas_bokeh

# create two plots
# s1 = figure(background_fill_color="#fafafa")
# s1.line(data_chart_1.index, data_chart_1.iloc[:, 0]*100, alpha=0.8, line_color="#53777a", 
#         line_width=2, legend_label='mom(50)')
# s1.line(data_chart_1.index, data_chart_1.iloc[:, 1]*100, alpha=0.8, line_color="#c02942", 
#         line_width=2, legend_label='mom(120)')
# s1.line(data_chart_1.index, data_chart_1.iloc[:, 2]*100, alpha=0.8, line_dash="dotdash", 
#         line_color="#d95b43", line_width=2, legend_label='smom(60,5)')

# s2 = figure(background_fill_color="#fafafa")
# s2.line(data_chart_2.index, data_chart_2.iloc[:, 0]*100, alpha=0.8, line_dash="dotdash", 
#         line_color="#53777a", line_width=2, legend_label='ma(100)')
# s2.line(data_chart_2.index, data_chart_2.iloc[:, 1]*100, alpha=0.8, line_color="#c02942", 
#         line_width=2, legend_label='mac(5,60)')
# s2.line(data_chart_2.index, data_chart_2.iloc[:, 2]*100, alpha=0.8, line_dash="dotted", 
#         line_color="#d95b43", line_width=2, legend_label='xmac(0.8,0.96)')

# # make a grid
# grid = gridplot([[s1, s2]], plot_width=400, plot_height=200)

# show(grid)
