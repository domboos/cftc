import pandas as pd
import numpy as np
# import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf
# import seaborn as sns
# import matplotlib.pyplot as plt
# from datetime import datetime
import os

curdir = os.getcwd()
from cfunctions import getexposure, engine1, gets




def getDates_of_MM():
    """
    Summary:
        to compare Nonc and MM over the same timespan
    Returns:
        pd.DataFrame: includes Dates
    """
    model_ids = pd.read_sql_query(f" Select model_id,bb_tkr from cftc.model_desc where model_type_id = {95}", engine1)
    model_ids_mm = list(model_ids.model_id)

    min_max_dateMM = pd.read_sql_query(
        f"Select Min(px_date) startDate, max(px_date) endDate ,model_id from cftc.forecast where model_id IN ({str(model_ids_mm)[1:-1]} ) group by model_id",
        engine1)
    dates_of_MM = pd.merge(min_max_dateMM, model_ids, on='model_id', how='left')
    return dates_of_MM


def getDirection(df_sample: pd.DataFrame, cftcVariableName: str, fcastVariableName: str):
    ## DEPRICATED ?
    """
    Summary:
        Directional analysis of forecasts
    Args:
        df_sample (pd.DataFrame): Output from Function getData
    """

    def binarity(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    try:
        temp = df_sample.copy()
        temp['cftc_binary'] = temp[cftcVariableName].apply(binarity)
        temp['cftc_forecast'] = temp[fcastVariableName].apply(binarity)

        return temp[temp.cftc_forecast == temp.cftc_binary].cftc.count() / temp.shape[0]
    except:
        return np.nan


def getOpenInterest(bb_tkr):
    oi = gets(engine1, type='agg_open_interest', data_tab='vw_data', desc_tab='cot_desc', bb_tkr=bb_tkr)
    oi.columns = ['oi']
    oi['OIma52'] = oi.rolling(52).mean()
    return oi


def getData(engine1,model_id, model_type_id, bb_tkr, model_types, start_date=None, end_date=None):
    forecast = pd.read_sql_query(f"SELECT * FROM cftc.forecast WHERE model_id = {model_id}", engine1,
                                 index_col='px_date')
    exposure = getexposure(engine1,
        type_of_trader=model_types.loc[model_type_id, 'cot_type'],
        norm=model_types.loc[model_type_id, 'cot_norm'],
        bb_tkr=bb_tkr
    )

    exposure.columns = exposure.columns.droplevel(0)
    exposure['diff'] = exposure[exposure.columns[0]].diff()

    df_sample = pd.merge(left=forecast[['qty']], right=exposure[['diff']], left_index=True, right_index=True,
                         how='left')
    df_sample.columns = ['forecast', 'cftc']

    # print(df_sample.shape)
    # Adjust timespan
    if (start_date != None) & (end_date != None):
        df_sample = df_sample[(df_sample.index >= start_date) & (df_sample.index <= end_date)]
    elif (start_date != None) & (end_date == None):
        df_sample = df_sample[(df_sample.index >= start_date)]
    elif (start_date == None) & (end_date != None):
        df_sample = df_sample[df_sample.index <= end_date]

    # print(df_sample.shape)

    # get OpenInterst #? adjust dates in open interest?
    oi = getOpenInterest(bb_tkr)

    # merge with df_sample
    df_sample = pd.merge(df_sample, oi, right_index=True, left_index=True, how='left')
    df_sample['cftc_adj'] = df_sample.cftc / df_sample.OIma52
    df_sample['forecast_adj'] = df_sample.forecast / df_sample.OIma52
    df_sample = df_sample.dropna()

    return df_sample
