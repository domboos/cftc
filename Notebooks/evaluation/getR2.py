# %%
import pickle
from typing import Any

import pandas as pd
import numpy as np
import statsmodels.api as sm

from datetime import datetime
from functions_eval import getDates_of_MM, getData
from cfunctions import engine1


# Overview of Models:
# model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc where model_type_id IN (82,76,95,100)", engine1)


## Functions:
def getR2ByHand(df_sample, cftcVariableName, fcastVariableName):
    e_diff = df_sample[cftcVariableName] - df_sample[fcastVariableName]
    nobs = len(e_diff)
    mspe_diff = (e_diff ** 2).sum(axis=0)
    var_diff = ((df_sample[cftcVariableName] - df_sample[cftcVariableName].mean(axis=0)) ** 2).sum(axis=0)
    try:
        oosR2_diff = 1 - mspe_diff / var_diff
        return oosR2_diff, nobs
    except ValueError:
        print("by calculating the r2 by hand something went wrong look at the following values:")
        print(mspe_diff)
        print(var_diff)
        return np.nan, nobs


def getResiduals(model_type_id:int, cftcVariableName:str, fcastVariableName:str, fixedStartdate=None,
                 fixedEndDate=None, type_=None) -> dict[int | Any, str | Any]:
    """
    :param model_type_id:
    :param cftcVariableName: 'cftc' or 'cftc_adj'
    :param fcastVariableName:  'forecast' OR 'forecast_adj'
    :param timespan:
    :param fixedStartdate: None or pd.DateTime()
    :param fixedEndDate: None or pd.DateTime()
    :param type_: None or 'diff'
    :return: pd.Data
    """

    residuals = {}
    residuals[0] = f"residuals to model type: {model_type_id}"
    bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things", engine1).bb_tkr
    model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)
    if model_types.index.name != 'model_type_id':
        model_types = model_types.set_index('model_type_id')

    models = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {model_type_id}",
                               engine1).set_index('model_id')

    for i in models.index:  # iterates through model_id

        tkr = models.loc[i, 'bb_tkr']
        print(f"{tkr} , model_id: {i}")

        df_sample = getData(model_id=i, model_type_id=model_type_id, bb_tkr=tkr, model_types=model_types,
                            start_date=fixedStartdate, end_date=fixedEndDate, engine1=engine1)

        if type_ == 'diff':
            residuals[tkr] = (df_sample.cftc - df_sample.forecast).values
        else:
            # * y = ax + b
            x = sm.add_constant(df_sample[fcastVariableName]).values
            y = df_sample[cftcVariableName].values
            mod_fit = sm.OLS(y, x).fit()
            residuals[tkr] = mod_fit.resid

    return residuals



def getR2results(model_type_id: int,
                 cftcVariableName: str,
                 fcastVariableName: str,
                 note='test',
                 timespan=None,
                 fixedStartdate=None,
                 fixedEndDate=None):
    """
    :return:
    :param model_type_id:
    :param cftcVariableName: either 'cftc' OR 'cftc_adj'
    :param fcastVariableName: 'forecast' OR 'forecast_adj'
    :param note: A note for results..
    :param timespan: None or 'MM' -> Selects only dates from Managed Money.
    :param fixedStartdate: None or pd.DateTime : e.g. datetime.strptime('1998-01-01', '%Y-%m-%d').date()
    :param fixedEndDate: None or pd.DateTime : e.g. datetime.strptime('1998-01-01', '%Y-%m-%d').date()
    :return:
    """
    bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things", engine1).bb_tkr
    model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)

    if model_types.index.name != 'model_type_id':
        model_types = model_types.set_index('model_type_id')

    models = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {int(model_type_id)}",
                               engine1).set_index('model_id')

    # * result DFs:
    df_r2_fromHand = pd.DataFrame(index=bb_tkrs, columns=['r2', 'nobs', 'model_type_id'])
    df_r2_fromHand['model_type_id'] = model_type_id
    df_r2_fromHand['Note'] = note

    if timespan == 'MM':
        dates_of_MM = getDates_of_MM()

    for i in models.index: # iterates through model_id

        tkr = models.loc[i, 'bb_tkr']
        print(f"{model_type_id}, {tkr}, model_id: {i}")

        if timespan == 'MM':
            startdate = dates_of_MM[dates_of_MM.bb_tkr == tkr].startdate.values[0]
            enddate = dates_of_MM[dates_of_MM.bb_tkr == tkr].enddate.values[0]
            print(f"startdate: {startdate}; Enddate: {enddate}")
            df_sample = getData(model_id=i, model_type_id=model_type_id, bb_tkr=tkr, model_types=model_types,
                                start_date=startdate, end_date=enddate, engine1=engine1)

        elif (fixedStartdate != None) | (fixedEndDate != None):
            df_sample = getData(model_id=i, model_type_id=model_type_id, bb_tkr=tkr, model_types=model_types,
                                start_date=fixedStartdate, end_date=fixedEndDate, engine1=engine1)
        else:
            df_sample = getData(model_id=i, model_type_id=model_type_id, bb_tkr=tkr, model_types=model_types,
                                start_date=None, end_date=None, engine1=engine1)

        # Might have no data for pre defined period
        if df_sample.shape[0] == 0:
            continue

        # Calc R2 By hand:
        r2, nobs = getR2ByHand(df_sample, cftcVariableName, fcastVariableName)

        r2, nobs = getR2ByHand(df_sample, cftcVariableName, fcastVariableName)
        df_r2_fromHand.loc[tkr, 'r2'] = r2
        df_r2_fromHand.loc[tkr, "nobs"] = nobs

    return df_r2_fromHand


def getStartAndEndDates():
    return {
        '1st_period': [datetime.strptime('1998-01-01', '%Y-%m-%d').date(),
                       datetime.strptime('2003-06-30', '%Y-%m-%d').date()],
        '2nd_period': [datetime.strptime('2003-07-01', '%Y-%m-%d').date(),
                       datetime.strptime('2008-12-31', '%Y-%m-%d').date()],
        '3rd_period': [datetime.strptime('2009-01-01', '%Y-%m-%d').date(),
                       datetime.strptime('2014-06-30', '%Y-%m-%d').date()],
        '4th_period': [datetime.strptime('2014-07-01', '%Y-%m-%d').date(),
                       datetime.strptime('2019-12-31', '%Y-%m-%d').date()]
    }


def rSquaredComaprisonAcrossPeriods(model_type_ids: list, cftcVariableName: str, fcastVariableName: str):
    cftcVariableName = 'cftc'  # * OR cftc_adj
    fcastVariableName = 'forecast'  # *OR 'forecast_adj'
    result = {}
    periods = getStartAndEndDates()

    for id_ in model_type_ids:
        # complete Period:
        result[f"{id_}_allObs"] = getR2results(model_type_id=id_,
                                              cftcVariableName=cftcVariableName,
                                              fcastVariableName=fcastVariableName,
                                              note=f"{id_}_allObs")

        for period in list(periods):
            result[f"{id_}_{period}"] = getR2results(model_type_id=id_,
                                                     cftcVariableName=cftcVariableName,
                                                     fcastVariableName=fcastVariableName,
                                                     fixedStartdate=periods[period][0],
                                                     fixedEndDate=periods[period][1],
                                                     note=f"{id_}_{period}")
    return result





# # %% #* Misc R2 Results
# model_list = [76, 82, 95, 100, 137, 93, 133, 134, 135, 117, 118, 119, 136]
# cftcVariableName = 'cftc'  # * OR cftc_adj
# fcastVariableName = 'forecast'  # *OR 'forecast_adj'
# avgs = pd.DataFrame(index=model_list, columns=['avg_r2'])
# r2 = pd.DataFrame()
# for item in model_list:
#     df_r2_temp = getR2results(model_type_id=item, cftcVariableName=cftcVariableName,
#                               fcastVariableName=fcastVariableName, note=str(item), timespan=None)
#     r2 = r2.append(df_r2_temp)
#     avgs.loc[item, 'avg_r2'] = (df_r2_temp.r2 * df_r2_temp.nobs).sum() / df_r2_temp.nobs.sum()
#
# writer = pd.ExcelWriter(f'r2_results_referenced_in_paper.xlsx', engine='xlsxwriter')
# avgs.to_excel(writer, sheet_name='R2_avg')
# r2.to_excel(writer, sheet_name='r2')
# writer.save()
# os.chdir('/home/jovyan/work/')
#
# # %% Write above Results to excel
# only_r2s = pd.DataFrame()
# r2_results_all = pd.DataFrame()
# avgs = pd.DataFrame(index=list(results), columns=['avg_r2'])
#
# for item in list(results):
#     print(item)
#     temp = results[item]
#     avgs.loc[item, 'avg_r2'] = (temp.r2 * temp.nobs).sum() / temp.nobs.sum()
#     r2_results_all = r2_results_all.append(results[item])
#
#     x = temp[['r2']].T
#     x.index = [f"r2_{item}"]
#     only_r2s = only_r2s.append(x)
#
# # Write to results:
# writer = pd.ExcelWriter(f'R2_results_v3.xlsx', engine='xlsxwriter')
# avgs.to_excel(writer, sheet_name='r2_avg')
# only_r2s.to_excel(writer, sheet_name='only_r2')
# r2_results_all.to_excel(writer, sheet_name='r2_all_res')
# writer.save()
# os.chdir('/home/jovyan/work/')

if __name__ == '__main__':


    ids =[131, 172, 153]
    cftcVariableName = 'cftc'
    fcastVariableName = 'forecast'
    result = rSquaredComaprisonAcrossPeriods(model_type_ids=ids, cftcVariableName=cftcVariableName, fcastVariableName=fcastVariableName)



    df_r2_result = pd.DataFrame(index = result[list(result)[0]].index.values)
    df_nobs_result = pd.DataFrame(index=result[list(result)[0]].index.values)
    for el in list(result):
        a = pd.DataFrame(index=result[el].index, columns=[f"{el}_r2"], data=result[el]['r2'].values)
        df_r2_result.loc[:,f"{el}_r2"] = a
        a = pd.DataFrame(index=result[el].index, columns=[f"{el}_obs"], data=result[el]['nobs'].values)
        df_nobs_result.loc[:, f"{el}_obs"] = a
    df_r2_result.to_excel("r2_ComPumpSwap.xlsx")
    df_nobs_result.to_excel('obsForR2Calc.xlsx')









    # get MM results
    # result['95_whole'] = getR2results(model_type_id= 95,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '95_whole')
    # result['100_whole'] = getR2results(model_type_id=100, cftcVariableName=cftcVariableName,fcastVariableName=fcastVariableName, note='100_whole')
    # for id_ in ids:
    #     result[f"{id_}_length_like_MM"] = getR2results(model_type_id=id_, cftcVariableName=cftcVariableName,
    #                                            fcastVariableName=fcastVariableName, note=f"{id_}_length_like_MM",
    #                                            timespan='MM')
    print("hi")
    # res = pd.DataFrame.from_dict(result)
    # res.to_excel("test1.xlsx")






