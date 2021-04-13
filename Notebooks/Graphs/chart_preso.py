from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.layouts import row
import pandas_bokeh
import pandas as pd
import numpy as np
from bokeh.palettes import brewer
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png

#if coding env is VSC use: 
file = './data/Chart_Trend_Signals.xlsx'
#file = 'C:\\Users\\bood\\switchdrive\\Tracking Traders\\04_Abbildungen\\Chart_Trend_Signals.xlsx'
data_chart_1 = pd.read_excel(file, sheet_name='mom', index_col=0)
print(data_chart_1)

data_chart_2 = pd.read_excel(file, sheet_name='maX', index_col=0)
print(data_chart_2)

# create two plots
s1 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s1.line(data_chart_1.index, data_chart_1.iloc[:, 0]*100, alpha=0.8, line_color="#53777a",
        line_width=3, legend_label='mom(50)')
s2 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s2.line(data_chart_1.index, data_chart_1.iloc[:, 1]*100, alpha=0.8, line_color="#c02942",
        line_width=3, legend_label='mom(120)')
s3 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s3.line(data_chart_1.index, np.ones(122)/2, alpha=0.8, line_dash="dotdash",
        line_color="#d95b43", line_width=3, legend_label='mom(260)')
s4 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s4.line(data_chart_2.index, data_chart_2.iloc[:, 0]*100, alpha=0.8, line_dash="dotdash",
        line_color="#53777a", line_width=3, legend_label='ma(100)')
s5 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s5.line(data_chart_2.index, data_chart_2.iloc[:, 2]*100, alpha=0.8, line_color="#c02942",
        line_width=3, line_dash="dotted", legend_label='mac(5,60)')
s6 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s6.line(data_chart_2.index, data_chart_2.iloc[:, 1]*100, alpha=0.8,
        line_color="#d95b43", line_width=3, legend_label='xmac(0.8,0.96)')
s7 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s7.line(data_chart_1.index, data_chart_1.iloc[:, 2]*100, alpha=0.8, line_dash="dotdash",
        line_color="#d95b43", line_width=3, legend_label='smom(60,5)')

file = './data/Chart_Trend_Signals2.xlsx'
# file = 'C:\\Users\\bood\\switchdrive\\Tracking Traders\\04_Abbildungen\\Chart_Trend_Signals2.xlsx'


data_chart_3 = pd.read_excel(file, sheet_name='mom', index_col=0)

data_chart_4 = pd.read_excel(file, sheet_name='ma', index_col=0)


#p = figure(x_range=(0, len(df)-1), y_range=(0, 800))
#p.grid.minor_grid_line_color = '#eeeeee'


data_chart = data_chart_1.merge(data_chart_2, how='outer', right_index=True, left_index=True)\
        .merge(data_chart_3, how='outer', right_index=True, left_index=True)\
        .merge(data_chart_4, how='outer', right_index=True, left_index=True).fillna(0)

data_chart['mom(260)'] = 1/200

# create two plots
s8 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s8.line(data_chart.index, data_chart.iloc[:, 6]*100, alpha=0.8, line_color="#53777a", line_width=3,
        legend_label='pmom(30,0.2)')

s9 = figure(background_fill_color="#fafafa", y_range=(-0.2, 5))
s9.line(data_chart.index, data_chart.iloc[:, 7]*100, alpha=0.8, line_color="#53777a", line_width=3,
        legend_label='pma(30,0.2)')


names = list(data_chart.columns)
print(data_chart.columns)
print('----------------')
print(names)
p = figure(background_fill_color="#fafafa", plot_width=400, plot_height=200)
p.varea_stack(stackers=names, x="index", color=brewer['BrBG'][9], source=data_chart)


# make a grid
grid = gridplot([[s1, s2, s3], [s4, s5, s6], [s7, s8, s9]], plot_width=400, plot_height=200)

show(grid)


show(p, width=400, height=200)