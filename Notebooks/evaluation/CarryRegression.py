import re
from datetime import datetime

import pandas as pd
import numpy as np
import statsmodels.api as sm
from cfunctions import engine1
from functions_eval import getData

spez = {'model_type_id': 136, 'cot_type': 'net_managed_money'}

net_NoNcom = {'model_type_id': 82, 'cot_type': 'net_non_commercials'}
net_MM = {'model_type_id': 100, 'cot_type': 'net_managed_money'}


query = """
select * from cftc.vw_return_w
where bb_tkr = 'BLOOMBERGTICKER'
order by px_date asc
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

models = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {spez['model_type_id']}",
                               engine1).set_index('model_id')

model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)  # get model_ids

# set index to model_type_id if not already is
if model_types.index.name != 'model_type_id':
    model_types = model_types.set_index('model_type_id')

for model in models.index:
    ticker = models.loc[model, 'bb_tkr']
    #Load Data
    q1 = re.sub('BLOOMBERGTICKER', ticker, query)
    df = pd.read_sql_query(q1, engine1)
    df = df.set_index('px_date')

    #Calculate essential Features:

    exposureData = getData(engine1, model_id=model, model_type_id=spez['model_type_id'], bb_tkr=ticker,
            model_types=model_types, start_date=None, end_date=None)[['forecast','cftc']]

    df_final = pd.merge(exposureData,df,right_index=True,left_index=True)

    for per in list(periods):
        startDate = periods[per][0]
        endDate = periods[per][1]

        print(f"per: {per}; sDate: {startDate} ; EndDate: {endDate}")

        df_sub = df_final.loc[startDate:endDate,['cftc','forecast','rel_carry_return']] # or rel_carry_return
        df_sub = df_sub.dropna()

        if df_sub.empty:
            continue

        y = df_sub["cftc"].values
        x = sm.add_constant(df_sub[['forecast','rel_carry_return']].values)
        #x = df_sub[['mom','deltaCarry']].values
        model = sm.OLS(y, x).fit()


        # df_res.loc[ticker,'coef_Mom'] = model.params[1]
        # df_res.loc[ticker, 'pval_Mom'] = model.pvalues[1]
        # df_res.loc[ticker, f"coef_dCarry_{per}"] = model.params[2]
        # df_res.loc[ticker, f"pval_dCarry_{per}"] = model.pvalues[2]
        df_res.loc[ticker, f"tstat_dCarry_{per}"] = model.tvalues[2]
        # df_res.loc[ticker, f"obs_{per}"] = model.nobs



df_res.to_excel('136_simple_Carry_Reg.xlsx')






