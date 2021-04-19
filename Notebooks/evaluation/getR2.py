#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# import matplotlib as plt 
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os 
from functions_eval import engine1, getDates_of_MM, getDirection, getData

# Overview of Models:
model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc where model_type_id IN (82,76,95,100)", engine1)

#%% Functions: getR2ByHand & getResiduals
def getR2ByHand(df_sample,cftcVariableName,fcastVariableName):
    e_diff = df_sample[cftcVariableName] - df_sample[fcastVariableName]
    nobs = len(e_diff)
    mspe_diff = (e_diff**2).sum(axis = 0)
    var_diff = ((df_sample[cftcVariableName]-df_sample[cftcVariableName].mean(axis=0))**2).sum(axis = 0)
    try:
        oosR2_diff = 1 - mspe_diff/var_diff
        return oosR2_diff, nobs
    except ValueError:
        print("by calculating the r2 by hand something went wrong look at the following values:")
        print(mspe_diff)
        print(var_diff)
        return np.nan, nobs

def getResiduals(model_type_id,cftcVariableName,fcastVariableName,timespan= None,fixedStartdate = None, fixedEndDate = None,type_ = None):
    residuals = {}
    residuals[0] = f"residuals to model type: {model_type_id}"
    bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things",engine1).bb_tkr
    model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)
    if model_types.index.name != 'model_type_id':
        model_types = model_types.set_index('model_type_id')
    
    models = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {model_type_id}", engine1).set_index('model_id')

    for i in models.index: #iterates through model_id
        tkr = models.loc[i,'bb_tkr']
        print(f"{tkr} , model_id: {i}")
        
        df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = tkr, model_types = model_types,start_date = fixedStartdate, end_date = fixedEndDate)

        if type_ == 'diff':
            residuals[tkr] = (df_sample.cftc - df_sample.forecast).values
        else:
            #* y = ax + b
            x = sm.add_constant(df_sample[fcastVariableName]).values
            y = df_sample[cftcVariableName].values
            mod_fit = sm.OLS(y,x).fit()
            residuals[tkr] = mod_fit.resid
    return residuals

#%%
def getR2results(model_type_id,cftcVariableName,fcastVariableName,note= 'test',timespan= None,fixedStartdate = None, fixedEndDate = None):

    bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things",engine1).bb_tkr
    model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)
    
    if model_types.index.name != 'model_type_id':
        model_types = model_types.set_index('model_type_id')
    
    models = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {int(model_type_id)}", engine1).set_index('model_id')

    #* result DFs:
    df_r2_fromHand = pd.DataFrame(index  = bb_tkrs, columns = ['r2','nobs','model_type_id'])
    df_r2_fromHand['model_type_id'] = model_type_id
    df_r2_fromHand['Note'] = note 

    if timespan == 'MM':
        dates_of_MM = getDates_of_MM()
   
    for i in models.index: #iterates through model_id
        tkr = models.loc[i,'bb_tkr']
        print(f"{model_type_id}, {tkr}, model_id: {i}")

        if timespan == 'MM':
            startdate = dates_of_MM[dates_of_MM.bb_tkr == tkr].startdate.values[0]
            enddate = dates_of_MM[dates_of_MM.bb_tkr == tkr].enddate.values[0]
            print(f"startdate: {startdate}; Enddate: {enddate}")
            df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = tkr, model_types = model_types,start_date = startdate, end_date = enddate)
        
        elif (fixedStartdate != None) | (fixedEndDate != None):
            df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = tkr, model_types = model_types,start_date = fixedStartdate, end_date = fixedEndDate)
        else:
            df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = tkr, model_types = model_types,start_date = None, end_date = None )
        
        #Might have no data for pre defined period
        if df_sample.shape[0] == 0:
            continue

        #Calc R2 By hand:
        r2, nobs = getR2ByHand(df_sample,cftcVariableName,fcastVariableName)
        
        r2, nobs = getR2ByHand(df_sample,cftcVariableName,fcastVariableName)
        df_r2_fromHand.loc[tkr,'r2'] = r2
        df_r2_fromHand.loc[tkr,"nobs"] = nobs
        
    return df_r2_fromHand
    
#%% #*Exposure R2 Comparison
first_start = datetime.strptime('1998-01-01', '%Y-%m-%d').date()
first_end = datetime.strptime('2003-06-30', '%Y-%m-%d').date()
second_start = datetime.strptime('2003-07-01', '%Y-%m-%d').date()
second_end = datetime.strptime('2008-12-31', '%Y-%m-%d').date()
third_start = datetime.strptime('2009-01-01', '%Y-%m-%d').date()
third_end = datetime.strptime('2014-06-30', '%Y-%m-%d').date()
fourth_start = datetime.strptime('2014-07-01', '%Y-%m-%d').date()
fourth_end = datetime.strptime('2019-12-31', '%Y-%m-%d').date()

cftcVariableName = 'cftc' #* OR cftc_adj
fcastVariableName = 'forecast' #*OR 'forecast_adj'

results = {}
results['76_whole'] = getR2results(model_type_id= 76,cftcVariableName = cftcVariableName,fcastVariableName= fcastVariableName,note = '76_whole',timespan= None)
results['82_whole'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_whole',timespan= None)

results['76_first_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_first_period',fixedStartdate = first_start, fixedEndDate = first_end)
results['82_first_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_first_period',fixedStartdate = first_start, fixedEndDate = first_end)

results['76_second_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_second_period',fixedStartdate = second_start, fixedEndDate = second_end)
results['82_second_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_second_period',fixedStartdate = second_start, fixedEndDate = second_end)

results['76_third_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_third_period',fixedStartdate = third_start, fixedEndDate = third_end)
results['82_third_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_third_period',fixedStartdate = third_start, fixedEndDate = third_end)

results['76_fourth_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_fourth_period',fixedStartdate = fourth_start, fixedEndDate = fourth_end)
results['82_fourth_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_fourth_period',fixedStartdate = fourth_start, fixedEndDate = fourth_end)

results['95_whole'] = getR2results(model_type_id= 95,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '95_whole',timespan= None)
results['100_whole'] = getR2results(model_type_id= 100,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '100_whole',timespan= None)

results['76_length_like_MM'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_length_like_MM',timespan= 'MM')
results['82_length_like_MM'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_length_like_MM',timespan= 'MM')



#%% Write above Results to excel
only_r2s = pd.DataFrame()
r2_results_all = pd.DataFrame()
avgs = pd.DataFrame(index = list(results), columns = ['avg_r2'])

for item in list(results):
    print(item)
    temp = results[item]    
    avgs.loc[item,'avg_r2'] = (temp.r2 * temp.nobs).sum() / temp.nobs.sum()
    r2_results_all = r2_results_all.append(results[item])
    
    x = temp[['r2']].T
    x.index = [f"r2_{item}"]
    only_r2s = only_r2s.append(x)

# Write to results:
os.chdir('/home/jovyan/work/reports/results')
writer = pd.ExcelWriter(f'R2_results_v3.xlsx', engine='xlsxwriter')
avgs.to_excel(writer, sheet_name= 'r2_avg')
only_r2s.to_excel(writer, sheet_name='only_r2')
r2_results_all.to_excel(writer, sheet_name = 'r2_all_res') 
writer.save()
os.chdir('/home/jovyan/work/')

#%% #*Autocorrelation result: x_t = a + b* x_(t-1): 
# - x: emp - fcast
#https://openstax.org/books/introductory-business-statistics/pages/13-2-testing-the-significance-of-the-correlation-coefficient
from statsmodels.stats.stattools import durbin_watson

cftcVariableName = 'cftc' #* OR cftc_adj
fcastVariableName = 'forecast' #*OR 'forecast_adj'

os.chdir('/home/jovyan/work/reports/results')
writer = pd.ExcelWriter(f'Autocorr.xlsx', engine='xlsxwriter')

for model_type_id in [76,82,93,137]:
    residuals = getResiduals(model_type_id,cftcVariableName,fcastVariableName,timespan= None,fixedStartdate = None, fixedEndDate = None,type_ = 'diff')
    bb_tkrs = list(residuals)[1:]
    temp_result = pd.DataFrame(index =bb_tkrs,columns = ['corr_emp_lag1','lag1-tstat','corr_emp_lag2','lag2-tstat'])
    for bb_tkr in bb_tkrs:
        
        df = pd.DataFrame(data = residuals[bb_tkr], columns =['diff_'])
        df[f"lag_1"] = df.diff_.shift(1)
        df[f"lag_2"] = df.diff_.shift(2)
          
        #get Autocorr values:
        corr = df.corr()
        temp_result.loc[bb_tkr,'corr_emp_lag1'] = corr.loc['diff_','lag_1']
        temp_result.loc[bb_tkr,'corr_emp_lag2'] = corr.loc['diff_','lag_2']
        

        df = df.dropna()
        x1 = sm.add_constant(df['lag_1']).values
        x2 = sm.add_constant(df['lag_2']).values
        y = df['diff_'].values
        
        mod_ac1 = sm.OLS(y,x1).fit()
        mod_ac2 = sm.OLS(y,x2).fit()
        
        temp_result.loc[bb_tkr,'lag1-tstat'] = mod_ac1.tvalues[1]
        temp_result.loc[bb_tkr,'lag2-tstat'] = mod_ac2.tvalues[1]
        
    temp_result.to_excel(writer, sheet_name= f"{model_type_id}")        


writer.save()
os.chdir('/home/jovyan/work')

#%% #* Plot Autocorrelation
cftcVariableName = 'cftc' #* OR cftc_adj
fcastVariableName = 'forecast' #*OR 'forecast_adj'
# model_type_id = 100

for model_type_id in [76]: #,95,82,76]:
    residuals = getResiduals(model_type_id,cftcVariableName,fcastVariableName,type_ = 'diff',timespan= None,fixedStartdate = None, fixedEndDate = None)
    
    bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things",engine1).bb_tkr
    oot = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things",engine1)
    fig, axs = plt.subplots(8, 3, sharex=False, sharey= False ,figsize=(15,20))
    fig.tight_layout()
    sns.set(font_scale = 1.2)
    sns.set_style('white')
    sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})

    # fig.text(0.5, 0.00, 'Return Lag', ha='center', fontsize = 20)
    # fig.text(0.00, 0.5, 'Beta', va='center', rotation='vertical', fontsize = 20)

    plot_matrix = np.arange(24).reshape(8, -1)
    for col in range(len(plot_matrix[0])):
        # print(f"Row: {row}")print(f"Row: {row}")
        for row in range(len(plot_matrix)):
            try:
                bb_tkr = bb_tkrs[plot_matrix[row][col]]
                # print(bb_tkr)
            except:
                break
            
            ax_curr = axs[row,col]
            plot_acf(residuals[bb_tkr], lags=np.arange(100)[1:], ax=ax_curr)
            
            ax_curr.set_xlabel('')
            ax_curr.set_ylabel('')
            ax_curr.set_title(f"Autocorr: {oot[oot.bb_tkr == bb_tkr].name.values[0]}")

    fig.delaxes(axs[7,2])
    plt.savefig(f"Autocorr-{model_type_id}.png",dpi=100) #'./reports/figures/'+
    plt.show()
