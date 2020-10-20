import pandas as pd
import numpy as np
import sqlalchemy as sq
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging

engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")


def gets(engine, type, data_tab='data', desc_tab='cot_desc', series_id=None, bb_tkr=None, bb_ykey='COMDTY',
         start_dt='1900-01-01', end_dt='2100-01-01', constr=None, adjustment = None):

    if constr is None:
        constr = ''
    else:
        constr = ' AND ' + constr

    if series_id is None:
        if desc_tab == 'cot_desc':
            series_id = pd.read_sql_query("SELECT cot_id FROM cftc.cot_desc WHERE bb_tkr = '" + bb_tkr +
                                          "' AND bb_ykey = '" + bb_ykey + "' AND cot_type = '" + type + "'", engine1)
        else:
            series_id = pd.read_sql_query("SELECT px_id FROM cftc.fut_desc WHERE bb_tkr = '" + bb_tkr +
                                          "' AND adjustment= '" + adjustment + "' AND bb_ykey = '" + bb_ykey + "' AND data_type = '" + type + "'", engine1)
        series_id = str(series_id.values[0][0])
    else:
        series_id = str(series_id)

    print(series_id)

    h_1 = " WHERE px_date >= '" + str(start_dt) +  "' AND px_date <= '" + str(end_dt) + "' AND px_id = "
    h_2 = series_id + constr + " order by px_date"
    fut = pd.read_sql_query('SELECT px_date, qty FROM cftc.' + data_tab + h_1 + h_2, engine, index_col='px_date')
    return fut


# test
# hh = gets(engine1, 'agg_open_interest', data_tab='vw_data', bb_tkr='KC')
# print(hh)

#test:
# exposure = getexposure(type_of_exposure = 'ratio_nonc',bb_tkr = 'KC',start_dt ='1900-01-01',end_dt='2019-12-31')

# price_non_adj = gets(engine1, 'px_last',desc_tab= 'fut_desc',data_tab = 'data', bb_tkr='KC', adjustment = 'none')

# bb_ykey='COMDTY'

def getexposure(type_of_exposure,bb_tkr,start_dt ='1900-01-01',end_dt='2019-12-31',bb_ykey='COMDTY'):
    """
    Parameters
    ----------
    type_of_exposure : str()
        one of: 'net_managed_money','net_non_commercials','ratio_mm','ratio_nonc'
    bb_tkr : TYPE
        Ticker from the commofity; example 'KC'
    start_dt : str(), optional
        The default is '1900-01-01'.
    end_dt :  str(), optional
        The default is '2100-01-01'.
    bb_ykey :  str(), optional
        The default is 'COMDTY'.
    
    
    Returns
    -------
    exposure : pd.DataFrame() with Multiindex (cftc,net_specs)
        Returns the exposure of the underlying position in USD (net_pos * fut_price * (Multiplier(?)) )


    """
    #TODO: include Multiplier
    

    if type_of_exposure == 'ratio_mm':
        oi = gets(engine1,type = 'agg_open_interest', data_tab='vw_data',desc_tab='cot_desc',bb_tkr=bb_tkr,bb_ykey=bb_ykey,start_dt= start_dt, end_dt=end_dt)
        pos1 = gets(engine1,type = 'net_managed_money', data_tab='vw_data',bb_tkr=bb_tkr,bb_ykey=bb_ykey,start_dt= start_dt, end_dt=end_dt)
        
        pos_temp = pd.merge(left = pos1, right = oi, how = 'left', left_index = True, right_index = True, suffixes=('_pos', '_oi'))
        exposure = pd.DataFrame(index = pos_temp.index,data = (pos_temp.qty_pos/pos_temp.qty_oi),columns = ['qty'])
        
        
    elif type_of_exposure == 'ratio_nonc':
        oi = gets(engine1,type = 'agg_open_interest', data_tab='vw_data',desc_tab='cot_desc',bb_tkr=bb_tkr,bb_ykey=bb_ykey,start_dt= start_dt, end_dt=end_dt)
        pos1 = gets(engine1,type = 'net_non_commercials', data_tab='vw_data',bb_tkr=bb_tkr,bb_ykey=bb_ykey,start_dt= start_dt, end_dt=end_dt)
        
        pos_temp = pd.merge(left = pos1, right = oi, how = 'left', left_index = True, right_index = True, suffixes=('_pos', '_oi'))
        exposure = pd.DataFrame(index = pos_temp.index,data = (pos_temp.qty_pos/pos_temp.qty_oi),columns = ['qty'])

        
    elif type_of_exposure == 'net_managed_money':
        
        pos = gets(engine1,type = type_of_exposure, data_tab='vw_data',bb_tkr=bb_tkr,bb_ykey=bb_ykey,start_dt= start_dt, end_dt=end_dt, adjustment = None) # constr=constr,
        
        price_non_adj = gets(engine1,type = 'px_last',desc_tab= 'fut_desc', data_tab='data',bb_tkr=bb_tkr,bb_ykey=bb_ykey,start_dt= start_dt, end_dt=end_dt,adjustment = 'none')
        df_merge = pd.merge(left = pos, right = price_non_adj, left_index = True, right_index = True, how = 'left')

        exposure = pd.DataFrame(index = df_merge.index)
        exposure['qty'] = (df_merge.qty_y * df_merge.qty_x).values
        
    elif type_of_exposure == 'net_non_commercials':
        pos = gets(engine1,type = type_of_exposure, data_tab='vw_data',bb_tkr=bb_tkr,bb_ykey=bb_ykey,start_dt= start_dt, end_dt=end_dt, adjustment = None) # constr=constr
    
        price_non_adj = gets(engine1, 'px_last',desc_tab= 'fut_desc',data_tab = 'data',bb_tkr=bb_tkr,bb_ykey=bb_ykey, start_dt =start_dt,end_dt=end_dt, adjustment = 'none')
        df_merge = pd.merge(left = pos, right = price_non_adj, left_index = True, right_index = True, how = 'left')

        exposure = pd.DataFrame(index = df_merge.index)
        exposure['qty'] = (df_merge.qty_y * df_merge.qty_x).values
    else:
        print('wrong type_of_exposur')
    

    midx = pd.MultiIndex(levels=[['cftc'], ['net_specs']], codes=[[0], [0]])    
    exposure.columns = midx
        
    return exposure
    
      
    
    
    


####------------------------------------------------------------------------------
####-------------------------Gamma and Returns------------------------------------
####------------------------------------------------------------------------------

def getGamma_and_Retmat(ret,gammatype,maxlag,regularization = 'd1'):
    """
    Parameters
    ----------
    ret : pd.DataFrame()
        log return series
    gammatype :  (str):    'flat','linear','dom','arctan','log','sqrt'
        How to calc the gamma_function (one of:'flat','linear','dom','arctan','log')
    maxlag : int
        DESCRIPTION.

    """
    def kth_diag_indices(a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols
        
    if gammatype == 'dom':
        rr = 1/4
        
    gamma = np.zeros((maxlag + 1, maxlag + 1))
            
        # loop creates gamma and lagged returns in ret
    # ret = ret_series.iloc[0:10,:]
    for i in range(0, maxlag + 1):
        ret['ret', str(i+1).zfill(3)] = ret['ret', '000'].shift(i+1)
         
        #ret[str(i+1).zfill(3)] = ret['001'].shift(i+1)
        
        if gammatype == 'dom':
            gamma[i, i] = 1 - 1 / (i+1)**rr
        
        elif gammatype == 'flat':
            gamma[i, i] = 1
        
        elif gammatype == 'linear':
            gamma[i, i] = (i+1)/(maxlag+1)
            gamma[0,0] = 0
            gamma[0,1] = 0
            
        elif gammatype == 'arctan':
            gamma[i,i] = np.arctan(0.2*i)
        
        elif gammatype == 'log':
            gamma[i,i] = np.log(1+ 5*i/maxlag)
 
        elif gammatype == 'sqrt':
            gamma[i,i] = np.sqrt(i+1)/15
            
        if  i < maxlag:
            gamma[i, i + 1] = - gamma[i, i]

    #Standardize the gamma Matrix from 0 to 1. 
    if regularization == 'd1':
        gmax = gamma.max()
        gamma[np.diag_indices_from(gamma)] /= gmax
        
        
        rows, cols = kth_diag_indices(gamma,1)
        gamma[rows,cols] /= gmax
    elif  regularization == 'd2_unadj':
         rowsm1, colsm1 = kth_diag_indices(gamma,-1)
         gamma[rowsm1,colsm1] =-gamma[rowsm1,colsm1+1]
    
         gamma[np.diag_indices_from(gamma)] = gamma[np.diag_indices_from(gamma)]*2
         gamma /= gamma.max()
    elif regularization == 'd2_adj':
         rowsm1, colsm1 = kth_diag_indices(gamma,-1)
         gamma[rowsm1,colsm1] =-gamma[rowsm1,colsm1+1]
    
         gamma[np.diag_indices_from(gamma)] = gamma[np.diag_indices_from(gamma)]*2
         gamma /= gamma.max()
     
         #naildown last value: f
         gamma[maxlag,maxlag-1] = -1
         gamma[maxlag,maxlag] = 1
        
         gamma = np.vstack([gamma, np.zeros(maxlag+1)])
         gamma[-1,-1] = 1
         gamma = gamma[1:,:]
    
    
    #naildown last value: f
    #TODO: shouldn't it be gamma[:,maxlag] = 0
    gamma[maxlag, maxlag] = 0
    
        
    gamma = gamma[:-1,:]
    ret = ret.iloc[maxlag:,:] #delete the rows with nan due to its shift.
    return gamma,ret




def getAlpha(alpha,y_diff,y):
    """
    Parameters
    ----------
    alpha : str()
        either 'stdev',var'
    y_diff : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    alpha_diff : TYPE
        DESCRIPTION.
    alpha_norm : TYPE
        DESCRIPTION.

    """
    if alpha[0] == 'stdev':
        alpha_diff = y_diff[0:250].std()
        alpha_norm = y[0:250].std()
        
    elif alpha[0] == 'var':
        alpha_diff = y_diff[0:250].var()
        alpha_norm = y[0:250].var()
        
    elif alpha[0] == 'const':
        alpha_diff = 1 ; alpha_norm = 1
    else:
        print('wrong alpha')
    
    return alpha_diff, alpha_norm
    
def merge_pos_ret(pos,ret):
    """
    Parameters
    ----------
    pos : 
        position
    ret : 
        ret-matrix
        
    """
    cc_ret = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).dropna()
    cc_ret_diff = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().dropna()
    return cc_ret, cc_ret_diff


def calcinsampleReg(getGamma_and_Retmat_obj, pos, decay,alpha,alpha_scale_factor,window,maxlag):
    """
    Parameters
    ----------
    getGamma_and_Retmat_obj : TYPE
        return object from getGamma_and_Retmat
    pos : TYPE
        object from get exposure
    decay : float
        between 0 and ...
    alpha : []
        string, at the moment only 
    window : int 
        normally 250
    maxlag: int
        Maximum lag for returns (Normally 250)

    Returns
    -------
    None.

    """
    # get Gamma and ret:
    gamma = getGamma_and_Retmat_obj[0]
    ret = getGamma_and_Retmat_obj[1]
    
    # Merge pos and rets:
    tpy = merge_pos_ret(pos,ret)
    cc_ret = tpy[0]
    cc_ret_diff = tpy[1]
    
    model_idx = cc_ret.index
    model_clm = pd.MultiIndex.from_product([alpha, ['level', 'diff_'], ['dod']])
    models = pd.DataFrame(index=model_idx, columns=model_clm)
    scores = pd.DataFrame(index=model_idx, columns=model_clm)
    prediction = pd.DataFrame(index=model_idx, columns=model_clm)
    df_alpha = pd.DataFrame(index = model_idx, columns =['alpha_diff','alpha_dod'])
    
    for idx,day in enumerate(cc_ret.index[0:-(window+1)]):
        
        if tpy == []:
            print('no match')
            break
        ##  rolling window parameters:
        w_start = cc_ret.index[idx]
        w_end = cc_ret.index[idx + window]
        forecast_period = cc_ret.index[idx+window+1] # includes the day x in [:x]
        
        if decay !=0:
            retFac = np.fromfunction(lambda i, j: decay ** i, cc_ret['ret'].loc[w_start:w_end,:].values.shape)[::-1]
            cc_ret_est = cc_ret['ret'].loc[w_start:w_end,:].values * retFac
            cc_ret_diff_est = cc_ret_diff['ret'].loc[w_start:w_end,:].values*retFac
        else:
            cc_ret_est = cc_ret['ret'].loc[w_start:w_end,:].values
            cc_ret_diff_est = cc_ret_diff['ret'].loc[w_start:w_end,:].values
         
            
        y = np.concatenate((cc_ret['cftc'].loc[w_start:w_end,:].values, np.zeros((maxlag, 1))))
        y_diff = np.concatenate((cc_ret_diff['cftc'].loc[w_start:w_end,:].values, np.zeros((maxlag, 1))))
        
        alpha_obj = getAlpha(alpha,y_diff,y)
        alpha_diff = alpha_obj[0] * alpha_scale_factor
        alpha_norm = alpha_obj[1] * alpha_scale_factor
        df_alpha.loc[w_end,'alpha_diff'] = alpha_diff
        df_alpha.loc[w_end,'alpha_dod'] = alpha_norm
         
        X_dod = np.concatenate((cc_ret_est,gamma * alpha_norm), axis=0)
        X_dod_diff = np.concatenate((cc_ret_diff_est,gamma * alpha_diff), axis=0)
        
    ##  fit the models
        models.loc[w_end, (alpha, 'level', 'dod')] = sm.OLS(y,X_dod).fit() #sm.add_constant(X_dod)
        models.loc[w_end, (alpha, 'diff_', 'dod')] = sm.OLS(y_diff,X_dod_diff).fit()
    
    ##  Rsquared - insample:
        scores.loc[w_end, (alpha, 'level', 'dod')] = models.loc[w_end, (alpha, 'level', 'dod')][0].rsquared
        scores.loc[w_end, (alpha, 'diff_', 'dod')] = models.loc[w_end, (alpha, 'diff_', 'dod')][0].rsquared
        
    ##  forecast    
        prediction.loc[forecast_period, (alpha, 'diff_', 'dod')] = \
        sum(models.loc[w_end, (alpha, 'diff_', 'dod')][0].params * cc_ret_diff['ret'].loc[forecast_period,:])
           
        prediction.loc[forecast_period, (alpha, 'level', 'dod')] = \
        sum(models.loc[w_end, (alpha, 'level', 'dod')][0].params * cc_ret['ret'].loc[forecast_period,:])
    
    #OOS Regression:
    emp = cc_ret_diff['cftc','net_specs']
    pred = prediction.loc[:,(alpha, ['diff_'], ['dod'])].astype(float)   
    results_diff = pd.merge(emp,pred, how='inner', left_index=True, right_index=True).dropna()
    results_diff.columns = ['cftc', 'forecast']
            
    mod = smf.ols('cftc ~ forecast',results_diff).fit()
            
    
    #out-of-sample R-Squared
    e = results_diff['cftc'] - results_diff['forecast']
    mspe = (e**2).sum(axis = 0)
    var = ((results_diff['cftc']-results_diff['cftc'].mean(axis=0))**2).sum(axis = 0)
    oosR2 = 1 - mspe/var            
    
    result = {}
    result['scores-insample'] = scores
    result['fcast'] = prediction
    result['OOSR2'] = [oosR2,mod.rsquared]
    result['alpha'] = df_alpha
    result['last_betas'] = models.loc[w_end, (alpha, 'diff_', 'dod')][0].params
    return result
        




    
def calcCFTC(type_of_exposure, bb_tkr,alpha,gammatype = 'dom',maxlag = 250, window = 250,alpha_scale_factor = 0.000001, decay = 0,start_dt ='1900-01-01',end_dt='2019-12-31',series_id=None,bb_ykey='COMDTY', constr=None, adjustment = None,regularization ='d1'):
    """
    Parameters
    ----------
    type_of_exposure : str()
        one of: 'net_managed_money','net_non_commercials','ratio_mm','ratio_nonc'
    bb_tkr : str()
            
    gammatype : str(), optional
        The default is 'dom' but can also be one of: 'flat','linear','arctan','log','sqrt'    
    maxlag : int, optional
        DESCRIPTION. The default is 250.
    window : int, optional
         The default is 250.
    decay : float, optional
        The default is 0.
    alpha : [str()], optional
        DESCRIPTION. The default is ['stdev'].
    start_dt : TYPE, optional
        DESCRIPTION. The default is '1900-01-01'.
    end_dt : TYPE, optional
        DESCRIPTION. The default is '2019-12-31'.
    series_id : TYPE, optional
        DESCRIPTION. The default is None.
    bb_ykey : TYPE, optional
        DESCRIPTION. The default is 'COMDTY'.
    constr : TYPE, optional
        DESCRIPTION. The default is None.
    adjustment : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    res1 : TYPE
        DESCRIPTION.

    """
    
    
    # Get prices:
    fut = gets(engine1,type = 'px_last',desc_tab= 'fut_desc',data_tab = 'data', bb_tkr = bb_tkr, adjustment = 'by_ratio', start_dt =start_dt,end_dt=end_dt)
    
    #calc rets:
    ret_series = pd.DataFrame(index = fut.index)
    ret_series.loc[:,'ret'] = np.log(fut/fut.shift(1))
    ret_series = ret_series.dropna() #deletes first value
    ret_series.columns = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0],[0]])

    # getPos:
    pos = getexposure(type_of_exposure,bb_tkr,start_dt =start_dt,end_dt=end_dt,bb_ykey='COMDTY')
    
    #get Gamma and Retmat:
    tupl = getGamma_and_Retmat(ret= ret_series,gammatype = gammatype ,maxlag= maxlag,regularization=regularization)
    
    res1 = calcinsampleReg(getGamma_and_Retmat_obj = tupl, pos = pos, decay = decay,alpha = alpha,alpha_scale_factor=alpha_scale_factor, window = window, maxlag = maxlag)
    
    return res1



#### Tests:
    
# test
# hh = gets(engine1, 'agg_open_interest', data_tab='vw_data', bb_tkr='KC')
# print(hh)

# #get fut price, adjustment = none
# aa =gets(engine1, 'px_last',desc_tab= 'fut_desc',data_tab = 'data', bb_tkr = 'KC', adjustment = 'none')

# #get fut price, adjustment = 'by_roll' -> exposure berechung
# aa =gets(engine1, 'px_last',desc_tab= 'fut_desc',data_tab = 'data', bb_tkr = 'KC', adjustment = 'none')

# #get fut price, adjustment = 'by_ratio' -> rets berechung
# aa =gets(engine1, 'px_last',desc_tab= 'fut_desc',data_tab = 'data', bb_tkr = 'KC', adjustment = 'by_ratio')

# #TEST:
# type_of_exposure = 'net_managed_money' ; bb_tkr = 'KC' ;start_dt ='1900-01-01';end_dt='2100-01-01';series_id=None;bb_ykey='COMDTY'; constr=None; adjustment = None

# a = getexposure('net_managed_money', 'KC')


# res = calcCFTC('net_managed_money','CT')


# getGamma_and_Retmat(ret,gammatype,maxlag)


# calcinsampleReg(getGamma_and_Retmat_obj, pos, decay,alpha,window,maxlag)
    
# fut = gets(engine1, 'px_last',desc_tab= 'fut_desc',data_tab = 'data', bb_tkr = 'KC', adjustment = 'by_ratio')
# ret_series = pd.DataFrame(index = fut.index)
# ret_series.loc[:,'ret'] = np.log(fut/fut.shift(1))
# ret_series = ret_series.dropna() #deletes first value
# ret_series.columns = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0],[0]])


# pos = getexposure('net_managed_money', 'KC')

# tupl = getGamma_and_Retmat(ret_series,gammatype = 'dom',maxlag= 250)


# res1 = calcinsampleReg(getGamma_and_Retmat_obj = tupl, pos = pos, decay = 0,alpha = ['stdev'],window = 250,maxlag= 250)


# gamma = tupl[0]
# ret = tupl[1]


# cc_ret = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).dropna()
# cc_ret_diff = pd.merge(pos, ret.iloc[:, :-1], how='inner', left_index=True, right_index=True).diff().dropna()





#---------------------------------------------test:-------------------------------------------------
# decay = 0; alpha = ['stdev']; window = 250; maxlag = 250
# bb_tkr = 'CO'; type_of_exposure = 'net_non_commercials' ;gammatype = 'dom';alpha = ['stdev'];alpha_scale_factor= 0.01;  start_dt='2000-01-01'; end_dt='2100-01-01'

# #  # Get prices:
# fut = gets(engine1, 'px_last',desc_tab= 'fut_desc',data_tab = 'data', bb_tkr = bb_tkr, adjustment = 'by_ratio')

# #------------------------------------------- Vorberechnungnen:----------------------------------------------
# #calc rets:
# ret_series = pd.DataFrame(index = fut.index)
# ret_series.loc[:,'ret'] = np.log(fut/fut.shift(1))
# ret_series = ret_series.dropna() #deletes first value
# ret_series.columns = pd.MultiIndex(levels=[['ret'], [str(0).zfill(3)]], codes=[[0],[0]])
# getGamma_and_Retmat_obj = getGamma_and_Retmat(ret= ret_series,gammatype = gammatype ,maxlag= maxlag)

##getPos:
# pos = getexposure(type_of_exposure, bb_tkr)


