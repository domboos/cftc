import re

import pandas as pd
import statsmodels.api as sm
from cfunctions import *

#Engine:
engine1 = engine1


net_NoNcom = {'model_type_id': 82, 'cot_type': 'net_non_commercials'}
net_MM = {'model_type_id': 100, 'cot_type': 'net_managed_money'}
net_swap = {'model_type_id': 153, 'cot_type': 'net_swap'}
net_pump = {'model_type_id': 150, 'cot_type': 'net_pump'}
net_commercials = {'model_type_id': 147, 'cot_type': 'net_commercials'}


qCarry = """
select p1.px_date, p1.qty p1, p2.qty p2
from cftc.data p1
inner join cftc.fut_desc desc1 on p1.px_id = desc1.px_id
inner join cftc.data p2 on p1.px_date = p2.px_date
inner join cftc.fut_desc desc2 on desc2.px_id = p2.px_id
where desc1.bb_tkr = 'BLOOMBERGTICKER' and desc1.roll = 'active_futures' and desc1.adjustment = 'by_ratio'
and desc2.bb_tkr = 'BLOOMBERGTICKER' and desc2.roll = 'active_futures_2' and desc2.adjustment = 'by_ratio'

"""



def merge_pos_ret_carry(pos:pd.DataFrame,
                        ret:pd.DataFrame,
                        carry:pd.DataFrame,
                        diff:bool):
    if diff:
        cr = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().dropna()
        cr1 = pd.merge(cr,carry,how='inner',left_index=True,right_index=True).dropna()
    else:
         raise NotImplementedError()
    return cr1


def getCarry(bb_ticker:str):
    q = re.sub('BLOOMBERGTICKER', bb_ticker, qCarry)
    df = pd.read_sql_query(q,engine1)
    df = df.set_index('px_date')
    df['retF1'] = np.log(df.p1 / df['p1'].shift(1))
    df['retF2'] = np.log(df.p2 / df['p2'].shift(1))
    df['deltaCarry'] = df.retF1 - df.retF2

    return df[['deltaCarry']].copy()


getCarry('CT')





query_models = """
    select * from cftc.model_desc 
    where model_type_id = 82
    and cot_type = 'net_non_commercials' 
    """

models = pd.read_sql_query(query_models,engine1)


df_res = pd.DataFrame()

for idx, model in models.iterrows():
    if idx == 0 or (bb_tkr != model.bb_tkr or bb_ykey != model.bb_ykey):
        bb_tkr = model.bb_tkr
        bb_ykey = model.bb_ykey
        fut = gets(engine1, type='px_last', desc_tab='fut_desc', data_tab='data', bb_tkr=bb_tkr, adjustment='by_ratio')
        # calc rets:
        ret_series = pd.DataFrame(index=fut.index)
        ret_series.loc[:, 'ret'] = np.log(fut / fut.shift(1))
        ret_series = ret_series.dropna()  # deletes first value
        ret_series.columns = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0], [0]])


    #get Regularization Matrix
    gamma = getGamma(maxlag=model.lookback, regularization=model.regularization, gammatype=model.gamma_type,
                     gammapara=model.gamma_para, naildownvalue0=model.naildown_value0,
                     naildownvalue1=model.naildown_value1)

    pos = getexposure(engine1, type_of_trader=model.cot_type, norm=model.cot_norm, bb_tkr=bb_tkr, bb_ykey=bb_ykey)

    ret = getRetMat(ret_series, model.lookback)  # this is too long

    deltaCarry = getCarry(bb_tkr)

    cr = merge_pos_ret_carry(pos,ret,deltaCarry,model.diff)

    y0 = cr.iloc[:,0].values
    x0 = cr.iloc[:,1:].values
    gamma_final = np.append(gamma, np.zeros((gamma.shape[0],1)), axis=1) * 500
    y = np.append(y0,np.zeros((1,gamma_final.shape[0])))
    x = np.concatenate((x0, gamma_final))
    model_fit = sm.OLS(y, x).fit()
    # print(model_fit.summary())

    df_res.loc[model.bb_tkr,'tval'] = model_fit.tvalues[-1]
    df_res.loc[model.bb_tkr, 'pval'] = model_fit.pvalues[-1]
    df_res.loc[model.bb_tkr, 'coef'] = model_fit.params[-1]

print(df_res)

df_res.to_csv('Carry.csv')











