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

file = 'C:\\Users\\bood\\switchdrive\\Tracking Traders\\02_Daten\\desc_tables.xlsx'

models = pd.read_excel(file, 'model_desc')

conn = engine1.connect()
loaded_models = pd.read_sql_query("SELECT * FROM cftc.model_desc", engine1)

models = models.astype({"decay": float, "naildown_value0": float, "naildown_value1": float})
loaded_models = loaded_models.astype({"decay": float, "naildown_value0": float, "naildown_value1": float})

print(loaded_models.dtypes)
print(models.dtypes)

new_models = pd.merge(right=loaded_models, left=models, how='left', indicator=True)

print(new_models.to_string())

new_models = new_models.loc[new_models._merge.isin(['left_only']), :].drop(columns=['model_id', '_merge'])
print(new_models)

if not new_models.empty:
    print('loading...')
    new_models.to_sql('model_desc', engine1, schema='cftc', if_exists='append', index=False)



#h.to_sql('data', engine1, schema='cftc', if_exists='append', index=False)
