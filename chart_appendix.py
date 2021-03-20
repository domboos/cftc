#%%
import pandas as pd
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

file = 'C:\\Users\\grbi\\switchdrive\\Tracking Traders\\04_Abbildungen\\Chart_Trend_Signals2.xlsx'
data_chart_1 = pd.read_excel(file, sheet_name='mom', index_col=0)
print(data_chart_1)

data_chart_2 = pd.read_excel(file, sheet_name='ma', index_col=0)
print(data_chart_2)

# create two plots
color = ['crimson', 'cyan'] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
fig, (s1,s2) = plt.subplots(1,2, sharex=False, sharey= False ,figsize=(15,4))
fig.tight_layout()

sns.set(font_scale = 1.5)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
sns.lineplot(x = data_chart_1.index, y = data_chart_1.iloc[:, 0]*100,color ="#53777a", linewidth = 3,ax = s1 )
sns.lineplot(x = data_chart_2.index,y =  data_chart_2.iloc[:, 0]*100,color = "#53777a",linewidth =3, ax= s2)
s2.lines[0].set_linestyle("--")
s1.set_xlabel('') ;s1.set_ylabel('')
s2.set_xlabel('') ;s2.set_ylabel('')

plt.savefig(f"./signals_app.png",dpi=100)



#%% Old
# from bokeh.io import show, output_file
# from bokeh.models import ColumnDataSource, FactorRange
# from bokeh.plotting import figure
# from bokeh.layouts import gridplot
# from bokeh.layouts import row
# import pandas_bokeh

# s1 = figure(background_fill_color="#fafafa")
# s1.line(data_chart_1.index, data_chart_1.iloc[:, 0]*100, alpha=0.8, line_color="#53777a", line_width=2)

# s2 = figure(background_fill_color="#fafafa")
# s2.line(data_chart_2.index, data_chart_2.iloc[:, 0]*100, alpha=0.8, line_dash="dotdash", line_color="#53777a", line_width=2)

# # make a grid
# grid = gridplot([[s1, s2]], plot_width=400, plot_height=150)

# show(grid)
