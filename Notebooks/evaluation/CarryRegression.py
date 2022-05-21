import re
from datetime import datetime

import pandas as pd
import numpy as np
import statsmodels.api as sm
from cfunctions import engine1

net_NoNcom = {'model_type_id': 82, 'cot_type': 'net_non_commercials'}
net_MM = {'model_type_id': 100, 'cot_type': 'net_managed_money'}
net_swap = {'model_type_id': 153, 'cot_type': 'net_swap'}
net_pump = {'model_type_id': 150, 'cot_type': 'net_pump'}
net_commercials = {'model_type_id': 147, 'cot_type': 'net_commercials'}


query = """
select FC.px_date, FC.qty as mom, 
        D5.qty * D6.qty as spec, 
                             D1.qty as P1,
                             D3.qty as P3
        from cftc.forecast FC
        inner join cftc.model_desc MD on FC.model_id = MD.model_id
        inner join cftc.data D1 on FC.px_date = D1.px_date
        inner join cftc.fut_desc CD1 on D1.px_id = CD1.px_id
        inner join cftc.data D3 on FC.px_date = D3.px_date
        inner join cftc.fut_desc CD3 on D3.px_id = CD3.px_id
        inner join cftc.data D5 on FC.px_date = D5.px_date
        inner join cftc.cot_desc CD5 on D5.px_id = CD5.cot_id
                             inner join cftc.data D6 on FC.px_date = D6.px_date
        inner join cftc.fut_desc CD6 on D6.px_id = CD6.px_id
        where MD.bb_tkr = 'BLOOMBERGTICKER' 
        and CD1.bb_tkr = 'BLOOMBERGTICKER'
                             and CD3.bb_tkr = 'BLOOMBERGTICKER'
                             and CD5.bb_tkr = 'BLOOMBERGTICKER' 
                             and CD6.bb_tkr = 'BLOOMBERGTICKER' 
        and MD.model_type_id = 136 and CD5.cot_type = 'net_managed_money'
                             and CD1.roll = 'active_futures'  and CD1.adjustment = 'by_ratio'
                             and CD3.roll = 'active_futures_2'  and CD3.adjustment = 'by_ratio'
                             and CD6.roll = 'active_futures'  and CD6.adjustment = 'none'
        order by px_date
        """




orderOFThings = pd.read_sql_query(f"Select * from cftc.order_of_things order by ranking",engine1)
tickers = orderOFThings['bb_tkr'].values


periods =\
    {
        '1st_period': [datetime.strptime('1998-01-01', '%Y-%m-%d').date(),
                       datetime.strptime('2003-06-30', '%Y-%m-%d').date()],
        '2nd_period': [datetime.strptime('2003-07-01', '%Y-%m-%d').date(),
                       datetime.strptime('2008-12-31', '%Y-%m-%d').date()],
        '3rd_period': [datetime.strptime('2009-01-01', '%Y-%m-%d').date(),
                       datetime.strptime('2014-06-30', '%Y-%m-%d').date()],
        '4th_period': [datetime.strptime('2014-07-01', '%Y-%m-%d').date(),
                       datetime.strptime('2019-12-31', '%Y-%m-%d').date()]
    }


df_res = pd.DataFrame(index=tickers)

for ticker in tickers:

    #Load Data
    q1 = re.sub('BLOOMBERGTICKER', ticker, query)
    df = pd.read_sql_query(q1, engine1)
    df = df.set_index('px_date')

    #Calculate essential Features:
    df['deltaExposure'] = df.spec.diff()
    df['deltaCarry'] = np.log(df.p1 / df.p1.shift(1)) - np.log(df.p3 / df.p3.shift(1))

    df = df.dropna()

    for per in list(periods):
        startDate = periods[per][0]
        endDate = periods[per][1]

        print(f"per: {per}; sDate: {startDate} ; EndDate: {endDate}")

        df_sub = df.loc[startDate:endDate,['deltaExposure','mom','deltaCarry']]

        if df_sub.empty:
            continue

        y = df_sub["deltaExposure"].values
        x = sm.add_constant(df_sub[['mom','deltaCarry']].values)
        #x = df_sub[['mom','deltaCarry']].values
        model = sm.OLS(y, x).fit()


        # df_res.loc[ticker,'coef_Mom'] = model.params[1]
        # df_res.loc[ticker, 'pval_Mom'] = model.pvalues[1]
        df_res.loc[ticker, f"coef_dCarry_{per}"] = model.params[2]
        df_res.loc[ticker, f"pval_dCarry_{per}"] = model.pvalues[2]
        df_res.loc[ticker, f"tstat_dCarry_{per}"] = model.tvalues[2]
        df_res.loc[ticker, f"obs_{per}"] = model.nobs



df_res.to_excel('CarryRegression_NetCommercials_withPeriods.xlsx')






