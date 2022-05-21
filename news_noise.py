import pandas as pd
import statsmodels.api as sm
from cfunctions import *
from scipy.stats import t
import numpy as np

#Engine:
engine1 = engine1

qry = """
    select model_id, MD.bb_tkr from cftc.model_desc MD 
    inner join cftc.order_of_things OOT 
    on OOT.bb_tkr = MD.bb_tkr 
    where model_type_id = 100 
    order by ranking  
    """
#

model_list = pd.read_sql_query(qry, engine1)

df_res = pd.DataFrame(index=model_list.bb_tkr)

for idx, model in model_list.iterrows():
    print(idx)
    print(model.model_id)
    _beta = getBeta(engine1, model_id=model.model_id)[0]
    z = np.zeros((len(_beta), 5))
    H = np.concatenate((_beta.values, z), axis=1) - np.concatenate((z, _beta.values), axis=1)
    G = H*H
    Gsum = G.sum(axis=1)
    G = G / Gsum[:, None]
    df_res.loc[model.bb_tkr, 'news'] = G[:, :5].sum(axis=1).mean()

print(df_res)