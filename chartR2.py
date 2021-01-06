import pandas as pd
import sqlalchemy as sq
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import dodge

engine = sq.create_engine(
    "postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")

arrays = [['Energy', 'Energy', 'Energy', 'Energy', 'Energy', 'Energy', 'Grains', 'Grains'],
          ['CL', 'XB', 'HO', 'NG', 'CO', 'QS', 'C', 'W']]

arrays = pd.read_sql_query("SELECT sector, bb_tkr FROM cftc.order_of_things order by ranking", engine).transpose().values


tuples = list(zip(*arrays))
tuples2 = tuples + tuples

print(tuples2)

traders=['non_commercials', 'managed_money']
traders2=['non commercials', 'managed money']

non_commercials1 = [11, 5, 7, 6, 34, 13, 13, 4, 12, 13, 14, 2, 5, 47, 46, 43, 23, 23, 41, 11, 11, 11, 21 ]
managed_money1 = [12, 45, 67, 56, 33, 33, 33, 44, 1, 1, 1, 12, 45, 67, 56, 3, 33, 33, 44, 1, 1, 1, 2 ]

source = ColumnDataSource(data=dict(x=tuples, non_commercials=non_commercials1, managed_money=managed_money1))

p = figure(x_range=FactorRange(*tuples), plot_height=250, toolbar_location=None, tools="")

p.vbar(x=dodge('x', -0.2, range=p.x_range), width=0.4, top='non_commercials', source=source, color='red',
       legend_label="Non Commercials")
p.vbar(x=dodge('x', 0.2, range=p.x_range), width=0.4, top='managed_money', source=source, legend_label="Managed Money")

p.y_range.start = 0
p.x_range.range_padding = 0.02
p.xgrid.grid_line_color = None
p.yaxis.axis_label = 'linus lable'
#p.legend.location = "top_right"
#p.legend.orientation = "horizontal"
p.legend.label_text_font_size = '7pt'
p.yaxis.axis_label_text_font_size = "7pt"
p.yaxis.axis_label_text_font_style = "normal"
p.legend.background_fill_alpha = 0.0
#p.legend.border_line_color = 'blue'
p.legend.border_line_alpha = 0.0
p.xaxis.major_label_text_font_size = "7pt"
p.xaxis.major_label_text_font = "Bradley Hand ITC"
p.xaxis.group_text_font_size = "7pt"
p.xaxis.group_text_font = "Algerian"
show(p)

# https://stackoverflow.com/questions/51818724/plot-two-levels-of-x-ticklabels-on-a-pandas-multi-index-dataframe
# https://datascience.stackexchange.com/questions/10322/how-to-plot-multiple-variables-with-pandas-and-bokeh
# https://docs.bokeh.org/en/latest/docs/user_guide/categorical.html