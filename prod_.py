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
print(bb)

df_res = pd.DataFrame(index=bb['bb_tkr'].values, columns=['prod1', 'cons1', 'swap1', 'oth_spec1', 'non_rep1', 'prod',
                                                          'cons', 'swap', 'oth_spec', 'non_rep', 'beta', 'Rsqrt'])
print(df_res)

for idx, tkr in bb.iterrows():
    print(tkr.bb_tkr)

    qry = """select FC.px_date, FC.qty as mom, 
        - D1.qty + lag(D1.qty) over (PARTITION BY D1.px_id ORDER BY D1.px_date) as prod, 
        D2.qty - lag(D2.qty) over (PARTITION BY D2.px_id ORDER BY D2.px_date) as cons, 
        D3.qty - lag(D3.qty) over (PARTITION BY D3.px_id ORDER BY D3.px_date) as swap, 
        D4.qty - lag(D4.qty) over (PARTITION BY D3.px_id ORDER BY D4.px_date) - FC.qty as oth_spec, 
        D5.qty - lag(D5.qty) over (PARTITION BY D5.px_id ORDER BY D5.px_date) as oi,
        - (D2.qty - lag(D2.qty) over (PARTITION BY D2.px_id ORDER BY D2.px_date))
        - (D3.qty - lag(D3.qty) over (PARTITION BY D3.px_id ORDER BY D3.px_date))
        - (D4.qty - lag(D4.qty) over (PARTITION BY D4.px_id ORDER BY D4.px_date))
        + (D1.qty - lag(D1.qty) over (PARTITION BY D1.px_id ORDER BY D1.px_date)) as non_rep
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

    _X = - mom_prod.loc[:, ['prod', 'cons', 'swap', 'oth_spec', 'non_rep']]
    model_fit = sm.OLS(mom_prod['mom'], _X.values).fit()
    for a in ['prod', 'cons', 'swap', 'oth_spec', 'non_rep']:
        model_fit2 = sm.OLS(mom_prod['mom'], mom_prod.loc[:, a].values).fit()
        df_res.loc[tkr.bb_tkr][a] = model_fit2.rsquared
    print(str(model_fit.summary()))
    tmp = _X.cov().sum() / mom_prod['mom'].var()
    print(type(tmp))
    print(_X.cov().iloc[0][0] / mom_prod['mom'].var())
    print(_X.corr())
    print(tmp.to_numpy())
    df_res.iloc[idx][:5] = tmp.to_numpy()

    qry2 = """ select FC1.px_date, FC1.qty as mom, FC2.qty as mprod 
        from cftc.forecast FC1 
        inner join cftc.forecast FC2 on FC1.px_date = FC2.px_date  
        inner join cftc.model_desc MD1 on FC1.model_id = MD1.model_id 
        inner join cftc.model_desc MD2 on FC2.model_id = MD2.model_id and MD1.bb_tkr = MD2.bb_tkr 
        where MD1.bb_tkr = '""" + str(tkr.bb_tkr) + """' and MD1.model_type_id = 131 and MD2.model_type_id = 173 
        order by FC1.px_date
        """
    mom_mprod = pd.read_sql_query(qry2, engine1)
    model_fit2 = sm.OLS(mom_mprod['mom'], mom_mprod['mprod']).fit()
    print(str(model_fit2.summary()))
    df_res.loc[tkr.bb_tkr]['Rsqrt'] = model_fit2.rsquared
    df_res.loc[tkr.bb_tkr]['beta'] = model_fit2.params[0]

print(df_res.to_string())