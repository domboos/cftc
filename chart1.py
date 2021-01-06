from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.layouts import row
import pandas_bokeh
import pandas as pd

file = 'C:\\Users\\bood\\switchdrive\\Tracking Traders\\04_Abbildungen\\Chart_Trend_Signals.xlsx'
data_chart_1 = pd.read_excel(file, sheet_name='mom', index_col=0)
print(data_chart_1)

data_chart_2 = pd.read_excel(file, sheet_name='maX', index_col=0)
print(data_chart_2)

# create two plots
s1 = figure(background_fill_color="#fafafa")
s1.line(data_chart_1.index, data_chart_1.iloc[:, 0]*100, alpha=0.8, line_color="#53777a", 
        line_width=2, legend_label='mom(50)')
s1.line(data_chart_1.index, data_chart_1.iloc[:, 1]*100, alpha=0.8, line_color="#c02942", 
        line_width=2, legend_label='mom(120)')
s1.line(data_chart_1.index, data_chart_1.iloc[:, 2]*100, alpha=0.8, line_dash="dotdash", 
        line_color="#d95b43", line_width=2, legend_label='smom(60,5)')

s2 = figure(background_fill_color="#fafafa")
s2.line(data_chart_2.index, data_chart_2.iloc[:, 0]*100, alpha=0.8, line_dash="dotdash", 
        line_color="#53777a", line_width=2, legend_label='ma(100)')
s2.line(data_chart_2.index, data_chart_2.iloc[:, 1]*100, alpha=0.8, line_color="#c02942", 
        line_width=2, legend_label='mac(5,60)')
s2.line(data_chart_2.index, data_chart_2.iloc[:, 2]*100, alpha=0.8, line_dash="dotted", 
        line_color="#d95b43", line_width=2, legend_label='xmac(0.8,0.96)')

# make a grid
grid = gridplot([[s1, s2]], plot_width=400, plot_height=200)

show(grid)

