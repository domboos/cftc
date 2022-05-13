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

bb = pd.read_sql_query("select bb_tkr from cftc.order_of_things order by ranking", engine1)

for idx, tkr in bb.iterrows():
    print(tkr.bb_tkr)

    qry = """select FC.px_date, FC.qty as mom, 
        D1.qty - lag(D1.qty) over (ORDER BY D1.px_date) as prod, 
        D2.qty - lag(D2.qty) over (ORDER BY D2.px_date) as cons, 
        D3.qty - lag(D3.qty) over (ORDER BY D3.px_date) as swap, 
        D4.qty - lag(D4.qty) over (ORDER BY D4.px_date) - FC.qty as oth_spec, 
        D5.qty - lag(D5.qty) over (ORDER BY D5.px_date) as oi,
        (D2.qty - lag(D2.qty) over (ORDER BY D2.px_date))
        + (D3.qty - lag(D3.qty) over (ORDER BY D3.px_date))
        + (D4.qty - lag(D4.qty) over (ORDER BY D4.px_date))
        - (D1.qty - lag(D1.qty) over (ORDER BY D1.px_date)) as non_rep
        from cftc.forecast FC
        inner join cftc.model_desc MD on FC.model_id = MD.model_id
        inner join cftc.data D1 on FC.px_date = D1.px_date
        inner join cftc.cot_desc CD1 on D1.px_id = CD1.cot_id
        inner join cftc.data D2 on FC.px_date = D2.px_date
        inner join cftc.cot_desc CD2 on D2.px_id = CD2.cot_id
        inner join cftc.data D3 on FC.px_date = D3.px_date
        inner join cftc.cot_desc CD3 on D3.px_id = CD3.cot_id
        inner join cftc.data D4 on FC.px_date = D4.px_date
        inner join cftc.cot_desc CD4 on D4.px_id = CD4.cot_id
        inner join cftc.data D5 on FC.px_date = D5.px_date
        inner join cftc.cot_desc CD5 on D5.px_id = CD5.cot_id
        where MD.bb_tkr = '""" + str(tkr.bb_tkr) + """' 
        and CD1.bb_tkr = '""" + str(tkr.bb_tkr) + """' 
        and CD2.bb_tkr = '""" + str(tkr.bb_tkr) + """'
        and CD3.bb_tkr = '""" + str(tkr.bb_tkr) + """' 
        and CD4.bb_tkr = '""" + str(tkr.bb_tkr) + """'
        and CD5.bb_tkr = '""" + str(tkr.bb_tkr) + """'
        and MD.model_type_id = 131 
        and CD1.cot_type = 'short_pump' and CD2.cot_type = 'long_pump' 
        and CD3.cot_type = 'net_swap' and CD4.cot_type = 'net_non_commercials'
        and CD5.cot_type = 'agg_open_interest' 
        order by px_date"""

    mom_prod = pd.read_sql_query(qry, engine1).dropna()

    model_fit = sm.OLS(mom_prod['mom'], mom_prod.loc[:, ['prod', 'cons', 'swap', 'oth_spec', 'non_rep']].values).fit()
    print(str(model_fit.summary()))