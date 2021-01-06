from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.layouts import row
import pandas_bokeh
import pandas as pd

file = 'C:\\Users\\bood\\switchdrive\\Tracking Traders\\04_Abbildungen\\Chart_Trend_Signals2.xlsx'
data_chart_1 = pd.read_excel(file, sheet_name='mom', index_col=0)
print(data_chart_1)

data_chart_2 = pd.read_excel(file, sheet_name='ma', index_col=0)
print(data_chart_2)

# create two plots
s1 = figure(background_fill_color="#fafafa")
s1.line(data_chart_1.index, data_chart_1.iloc[:, 0]*100, alpha=0.8, line_color="#53777a", line_width=2)

s2 = figure(background_fill_color="#fafafa")
s2.line(data_chart_2.index, data_chart_2.iloc[:, 0]*100, alpha=0.8, line_dash="dotdash", line_color="#53777a", line_width=2)

# make a grid
grid = gridplot([[s1, s2]], plot_width=400, plot_height=150)

show(grid)