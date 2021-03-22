# speed up to_sql inserts heavily with low latency db connection (https://github.com/pandas-dev/pandas/issues/8953)
from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

import pandas as pd
import sqlalchemy as sq

engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")

file = 'U:\\NetNC.xlsx'

pd.read_excel(file, 'import').to_sql('data', engine1, schema='cftc', if_exists='append', index=False)

# file2 = 'C:\\Users\\bood\\switchdrive\\Tracking Traders\\02_Daten\\02_readyforimport\\px_adj_none.xlsx'

#.dropna(axis=0, how='any')
#h= pd.read_excel(file2, 'load', skiprows=1, header=0, index_col=0).stack()\
#    .reset_index().rename(columns={'level_1': "px_id", 0: "qty"})

#h.to_sql('data', engine1, schema='cftc', if_exists='append', index=False)

