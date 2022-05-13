from cfunctions import *

# crate engine
from cengine import cftc_engine
engine1 = cftc_engine()

# speed up db
from pandas.io.sql import SQLTable

def _execute_insert(self, conn, keys, data_iter):
    print("Using monkey-patched _execute_insert")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))

SQLTable._execute_insert = _execute_insert

import pandas as pd
import statsmodels.api as sm

bb = pd.read_sql_query("select bb_tkr from cftc.order_of_things", engine1)

for idx, tkr in bb.iterrows():
    print(tkr.bb_tkr)

    qry = """select FC.px_date, FC.qty as mom, 
    D1.qty - lag(D1.qty) over (ORDER BY D1.px_date) as prod 
    from cftc.data D1
    inner join cftc.forecast FC on FC.px_date = D1.px_date 
    inner join cftc.cot_desc CD1 on D1.px_id = CD1.cot_id 
    inner join cftc.model_desc MD on FC.model_id = MD.model_id 
    where MD.bb_tkr = '""" + str(tkr.bb_tkr) + """' and CD1.bb_tkr =  '""" + str(tkr.bb_tkr) + """'
    and MD.model_type_id = 131 and CD1.cot_type = 'net_non_commercials' 
    order by px_date"""

    mom_prod = pd.read_sql_query(qry, engine1).dropna()

    model_fit = sm.OLS(mom_prod['mom'], mom_prod['prod']-mom_prod['mom']).fit()
    print(str(model_fit.summary()))