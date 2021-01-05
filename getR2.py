#%%
import matplotlib.pyplot as plt
# import matplotlib as plt 
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
#* for other calcs:
from cfunctions import *

#For overview
model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc where model_type_id IN (82,76,95,100)", engine1)

#%% Functions:

def getDates_of_MM():
    model_ids = pd.read_sql_query(f" Select model_id,bb_tkr from cftc.model_desc where model_type_id = {95}", engine1)
    model_ids_mm = list(model_ids.model_id)

    min_max_dateMM = pd.read_sql_query(f"Select Min(px_date) startDate, max(px_date) endDate ,model_id from cftc.forecast where model_id IN ({str(model_ids_mm)[1:-1]} ) group by model_id",engine1)
    dates_of_MM  = pd.merge(min_max_dateMM,model_ids, on = 'model_id', how = 'left')
    return dates_of_MM

def getDirection(df_sample):
    def binarity(x):
        if x>0:
            return 1
        elif x<0:
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
    oi.columns =[ 'oi']
    oi['OIma52'] = oi.rolling(52).mean()
    return oi

def getData(model_id,model_type_id,bb_tkr, model_types,start_date = None, end_date = None ):

    forecast = pd.read_sql_query(f"SELECT * FROM cftc.forecast WHERE model_id = {model_id}",engine1,index_col = 'px_date')
    exposure = getexposure(type_of_trader = model_types.loc[model_type_id,'cot_type'], norm = model_types.loc[model_type_id,'cot_norm'], bb_tkr = bb_tkr)
    exposure.columns = exposure.columns.droplevel(0)
    exposure['diff'] = exposure.net_specs.diff()

    df_sample = pd.merge(left = forecast[['qty']], right = exposure[['diff']] , left_index = True, right_index = True, how = 'left')
    df_sample.columns = ['forecast','cftc']

    # print(df_sample.shape)
    #Adjust timespan
    if (start_date != None) & (end_date != None):
        df_sample = df_sample[(df_sample.index >= start_date)& (df_sample.index <= end_date)]
    elif (start_date != None) & (end_date == None):
        df_sample = df_sample[(df_sample.index >= start_date)]
    elif (start_date == None) & (end_date != None):
        df_sample = df_sample[df_sample.index <= end_date]
    
    # print(df_sample.shape)
    
    #get OpenInterst #? adjust dates in open interest? 
    oi = getOpenInterest(bb_tkr)
    
    #merge with df_sample
    df_sample = pd.merge(df_sample,oi, right_index=True,left_index = True, how = 'left')
    df_sample['cftc_adj'] = df_sample.cftc / df_sample.OIma52
    df_sample['forecast_adj'] = df_sample.forecast / df_sample.OIma52
    df_sample = df_sample.dropna()

    return df_sample 

#calc R2 by hand:
def getR2ByHand(df_sample,cftcVariableName,fcastVariableName):
    e_diff = df_sample[cftcVariableName] - df_sample[fcastVariableName]
    mspe_diff = (e_diff**2).sum(axis = 0)
    var_diff = ((df_sample[cftcVariableName]-df_sample[cftcVariableName].mean(axis=0))**2).sum(axis = 0)
    try:
        oosR2_diff = 1 - mspe_diff/var_diff
        return oosR2_diff
    except ValueError:
        print("by calculating the r2 by hand something went wrong look at the following values:")
        print(mspe_diff)
        print(var_diff)
        return np.nan

def getResiduals(model_type_id,cftcVariableName,fcastVariableName,timespan= None,fixedStartdate = None, fixedEndDate = None):
    residuals = {}
    residuals[0] = f"residuals to model type: {model_type_id}"
    bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things",engine1).bb_tkr
    model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)
    if model_types.index.name != 'model_type_id':
        model_types = model_types.set_index('model_type_id')
    
    models = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {int(model_type_id)}", engine1).set_index('model_id')

    for i in models.index: #iterates through model_id
        # print(i)
        df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = models.loc[i,'bb_tkr'], model_types = model_types,start_date = fixedStartdate, end_date = fixedEndDate)

        #* y = ax + b
        x = sm.add_constant(df_sample[fcastVariableName]).values
        y = df_sample[cftcVariableName].values
        mod_fit = sm.OLS(y,x).fit()
        residuals[ models.loc[i,'bb_tkr']] = mod_fit.resid
    
    return residuals


#%% #? new
# #* tEST:
# model_type_id= 82
# cftcVariableName ='cftc_adj'
# fcastVariableName= 'forecast_adj'
# writetoExcel= True
# excelName= 'test'
# timespan= 'MM'


#%%



def getR2results(model_type_id,cftcVariableName,fcastVariableName,note= 'test',timespan= None,fixedStartdate = None, fixedEndDate = None):

    bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things",engine1).bb_tkr
    model_types = pd.read_sql_query("SELECT * from cftc.model_type_desc", engine1)
    if model_types.index.name != 'model_type_id':
        model_types = model_types.set_index('model_type_id')
    models = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {int(model_type_id)}", engine1).set_index('model_id')


    #* result DFs:
    df_r2_MincerZarnowitz = pd.DataFrame(index  = [model_type_id], columns = bb_tkrs)
    df_r2_fromHand = pd.DataFrame(index  = [model_type_id], columns = bb_tkrs)
    df_direction = pd.DataFrame(index  = [model_type_id], columns = bb_tkrs)
    df_coefs = pd.DataFrame( index =models.index, columns =['const','beta','tstat_beta','tstat_const','975Int_const','025Int_const','025Int_beta','975Int_beta','obs','bb_tkr'])

    df_coefs['Note'] =note ; df_direction['Note'] = note ; df_r2_fromHand['Note'] = note ;  df_r2_MincerZarnowitz['Note'] = note


    if timespan == 'MM':
        dates_of_MM = getDates_of_MM()
        
    
    for i in models.index: #iterates through model_id
        print(i)

        if timespan == 'MM':
            startdate = dates_of_MM[dates_of_MM.bb_tkr == models.loc[i,'bb_tkr']].startdate.values[0]
            enddate = dates_of_MM[dates_of_MM.bb_tkr == models.loc[i,'bb_tkr']].enddate.values[0]
            print(f"startdate: {startdate}; Enddate: {enddate}")
            df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = models.loc[i,'bb_tkr'], model_types = model_types,start_date = startdate, end_date = enddate)
        
        elif (fixedStartdate != None) | (fixedEndDate != None):
            df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = models.loc[i,'bb_tkr'], model_types = model_types,start_date = fixedStartdate, end_date = fixedEndDate)
        else:
            df_sample = getData(model_id = i,model_type_id= model_type_id,bb_tkr = models.loc[i,'bb_tkr'], model_types = model_types,start_date = None, end_date = None )
        
        #Might have no data for pre defined period
        if df_sample.shape[0] == 0:
            continue

        #Calc R2 By hand:
        df_r2_fromHand.loc[model_type_id,models.loc[i,'bb_tkr']] = getR2ByHand(df_sample,cftcVariableName,fcastVariableName) 
        
        #* Mincer Zarnowitz Regression:y-x = a + (b-1)x
        y = df_sample[cftcVariableName] - df_sample[fcastVariableName]
        x = sm.add_constant(df_sample[fcastVariableName]).values
        mod_MZ = sm.OLS(y,x).fit()
        del y

        #* y = ax + b
        y = df_sample[cftcVariableName].values
        mod_fit = sm.OLS(y,x).fit()
        # print(mod_fit.summary())
        
        #get T-statistics for Autocorr:
        resid = mod_fit.resid
        acf, ci = sm.tsa.acf(resid, alpha=0.05)
        
        k=1
        for corr in acf[1:5]:
            t = (corr*np.sqrt(len(resid)-k-2)) / np.sqrt(1- corr**2)
            df_coefs.loc[i,f"AC_tstat_l{k}"] =  t       
            k = k+1
                
        df_r2_MincerZarnowitz.loc[model_type_id,models.loc[i,'bb_tkr']] = mod_fit.rsquared
        
        df_coefs.loc[i,'tstat_beta'] = mod_MZ.tvalues[1] #* test if beta1 is significantly diffent from 1
        df_coefs.loc[i,'tstat_const'] = mod_fit.tvalues[0] #* test if beta1 is significantly diffent from 1
        df_coefs.loc[i,'const'] = mod_fit.params[0]
        df_coefs.loc[i,'beta'] = mod_fit.params[1]
        # print(mod_fit.conf_int(alpha = 0.05))
        
        df_coefs.loc[i,'975Int_const'] = mod_fit.conf_int(alpha = 0.05)[0][1]
        df_coefs.loc[i,'025Int_const'] = mod_fit.conf_int(alpha = 0.05)[0][0]
        df_coefs.loc[i,'975Int_beta'] = mod_fit.conf_int(alpha = 0.05)[1][1]
        df_coefs.loc[i,'025Int_beta'] = mod_fit.conf_int(alpha = 0.05)[1][0]
        df_coefs.loc[i,'obs'] = len(mod_fit.fittedvalues)
        df_coefs.loc[i,'model_type_id'] = model_type_id
        df_coefs.loc[i,'bb_tkr'] = models.loc[i,'bb_tkr']
                    

        # Direction right:
        df_direction.loc[model_type_id,models.loc[i,'bb_tkr']] = getDirection(df_sample)
        
    return df_coefs,df_r2_MincerZarnowitz,df_r2_fromHand,df_direction
    
#%% Generate Results: 

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
results['95_whole'] = getR2results(model_type_id= 95,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '95_whole',timespan= None)
results['100_whole'] = getR2results(model_type_id= 100,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '100_whole',timespan= None)

results['76_length_like_MM'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_length_like_MM',timespan= 'MM')
results['82_length_like_MM'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_length_like_MM',timespan= 'MM')

results['76_first_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_first_period',fixedStartdate = first_start, fixedEndDate = first_end)
results['82_first_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_first_period',fixedStartdate = first_start, fixedEndDate = first_end)
results['95_first_period'] = getR2results(model_type_id= 95,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '95_first_period',fixedStartdate = first_start, fixedEndDate = first_end)
results['100_first_period'] = getR2results(model_type_id= 100,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '100_first_period',fixedStartdate = first_start, fixedEndDate = first_end)

results['76_second_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_second_period',fixedStartdate = second_start, fixedEndDate = second_end)
results['82_second_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_second_period',fixedStartdate = second_start, fixedEndDate = second_end)
results['95_second_period'] = getR2results(model_type_id= 95,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '95_second_period',fixedStartdate = second_start, fixedEndDate = second_end)
results['100_second_period'] = getR2results(model_type_id= 100,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '100_second_period',fixedStartdate = second_start, fixedEndDate = second_end)

results['76_third_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_third_period',fixedStartdate = third_start, fixedEndDate = third_end)
results['82_third_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_third_period',fixedStartdate = third_start, fixedEndDate = third_end)
results['95_third_period'] = getR2results(model_type_id= 95,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '95_third_period',fixedStartdate = third_start, fixedEndDate = third_end)
results['100_third_period'] = getR2results(model_type_id= 100,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '100_third_period',fixedStartdate = third_start, fixedEndDate = third_end)

results['76_fourth_period'] = getR2results(model_type_id= 76,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '76_fourth_period',fixedStartdate = fourth_start, fixedEndDate = fourth_end)
results['82_fourth_period'] = getR2results(model_type_id= 82,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '82_fourth_period',fixedStartdate = fourth_start, fixedEndDate = fourth_end)
results['95_fourth_period'] = getR2results(model_type_id= 95,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '95_fourth_period',fixedStartdate = fourth_start, fixedEndDate = fourth_end)
results['100_fourth_period'] = getR2results(model_type_id= 100,cftcVariableName =cftcVariableName ,fcastVariableName= fcastVariableName,note = '100_fourth_period',fixedStartdate = fourth_start, fixedEndDate = fourth_end)


df_coefs = pd.DataFrame()
df_r2_MincerZarnowitz = pd.DataFrame()
df_r2_fromHand = pd.DataFrame()
df_direction = pd.DataFrame()

for item in list(results):
    df_coefs = df_coefs.append(results[item][0])
    df_r2_MincerZarnowitz =  df_r2_MincerZarnowitz.append(results[item][1])
    df_r2_fromHand =  df_r2_fromHand.append(results[item][2])
    df_direction = df_direction.append(results[item][3])


#* Write to results:
writer = pd.ExcelWriter(f'reports\\results\\Final_results_non_adj_02.xlsx', engine='xlsxwriter')

df_coefs.to_excel(writer, sheet_name= 'coefs')
df_r2_MincerZarnowitz.to_excel(writer, sheet_name = 'r2_MZ') 
df_r2_fromHand.to_excel(writer, sheet_name='r2_byHand')
df_direction.to_excel(writer, sheet_name= 'direction')
writer.save()





#%% #* Plot Autocorrelation
cftcVariableName = 'cftc' #* OR cftc_adj
fcastVariableName = 'forecast' #*OR 'forecast_adj'
# model_type_id = 100

for model_type_id in [76]: #,95,82,76]:
    residuals = getResiduals(model_type_id,cftcVariableName,fcastVariableName,timespan= None,fixedStartdate = None, fixedEndDate = None)

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
    plt.savefig(f"reports/figures/Autocorr-{model_type_id}.png",dpi=100) #'./reports/figures/'+
    plt.show()

# %%
cftcVariableName = 'cftc' #* OR cftc_adj
fcastVariableName = 'forecast' #*OR 'forecast_adj'




#%%
df = pd.DataFrame()
print(list(result))
for item in result:
    df = df.append(result[item])
df.to_excel('reports/Autocorrelation_tstats.xlsx')
# %%
#%% #* Test for autocorrelation: https://openstax.org/books/introductory-business-statistics/pages/13-2-testing-the-significance-of-the-correlation-coefficient
from statsmodels.stats.stattools import durbin_watson

cftcVariableName = 'cftc' #* OR cftc_adj
fcastVariableName = 'forecast' #*OR 'forecast_adj'

result = {}
for model_type_id in [76]: #100,95,82,76]:
    residuals = getResiduals(model_type_id,cftcVariableName,fcastVariableName,timespan= None,fixedStartdate = None, fixedEndDate = None)

    temp_result = pd.DataFrame(index = list(residuals),columns = ['lag1','lag2','lag3','lag4'])
    for bb_tkr in list(residuals)[1:]:
        # print(bb_tkr)
        # plot_acf(residuals[bb_tkr], lags=np.arange(100)[1:])
        # print(durbin_watson(residuals[bb_tkr]))

        acf, ci = sm.tsa.acf(residuals[bb_tkr], alpha=0.05)
        # print(acf)

        tstat = list()
        k=1
        for corr in acf[1:5]:
            if (bb_tkr == 'CT') & (k ==1):
                print(acf)
                print(f"corr: {corr}")
                print(f"k: {k}")
                print((len(residuals[bb_tkr])-k-2))
            t = (corr*np.sqrt(len(residuals[bb_tkr])-k-2)) / np.sqrt(1- corr**2)
            k = k+1
            tstat.append(t)
        # print(tstat)
        try:
            temp_result.loc[bb_tkr,:] = tstat
        except:
            print(temp_result.columns)
            print(tstat)
            break
        
    temp_result['Note'] = model_type_id
    result[model_type_id] = temp_result
# %%
