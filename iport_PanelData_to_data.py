# speed up to_sql inserts heavily with low latency db connection (https://github.com/pandas-dev/pandas/issues/8953)
import numpy as np
from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

import pandas as pd
import sqlalchemy as sq

engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")

file = 'C:\\Users\\grbi\\switchdrive\\Tracking Traders\\02_Daten\\02_readyforimport\\GernericFuts_2_3_4.xlsx'

df = pd.read_excel(file,sheet_name='Numeric')
df = df.set_index("dates")

df1 = df.stack()
df1 = df1.reset_index()
df1.columns = ['px_date','px_id','qty']

df2 = df1[['px_id','px_date','qty']]

print(df2)

df2.iloc[200000:, :].to_sql('data', engine1, schema='cftc', if_exists='append', index=False)
print('hi')




