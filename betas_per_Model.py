#%%
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlalchemy as sq
from datetime import datetime


#TODO Reversal trades in NG (?)

#get all models which are already calculated
# TODO: Select a model_id:

#%% Functions:
def getSpecificData(dates,model_type):
    models = pd.read_sql_query(f"SELECT * FROM cftc.model_desc where model_type_id ={model_type}",engine1)
    model_ids = list(models.model_id)
    temp_ids = len(model_ids)
    
    betas_all = pd.read_sql_query(f"SELECT * from cftc.beta where  px_date IN( {str(dates)[1:-1]} ) and model_id IN ( {str(model_ids)[1:-1]} )",engine1)
    betas_all.px_date = betas_all.px_date.astype('datetime64[ns]')

    if temp_ids != len(model_ids):
        print(models[~models.model_id.isin(model_ids)].bb_tkr)
        print('^Do not exist in Betas')

    return betas_all,models 


def getBetas(model_id,betas_all,date):
    needed_betas = betas_all[(betas_all.px_date == date) & (betas_all.model_id == model_id)]
    return needed_betas


def createFigurePerModel(model_type,betas_all,dates):
    model_ids= list(betas_all.groupby('model_id').qty.max().sort_values(ascending = False).index)
    
    fig, axs = plt.subplots(6, 4, sharex=True, sharey= False ,figsize=(20, 10))
    main_title = str('All Betas with model_id:'+str(model_type))
    # print(f"main_title: {main_title}")
    fig.suptitle(main_title)
    fig.tight_layout()
    
    k = 0
    for row in range(0,6):
        #print(f"Row: {row}")
        for col in range(0,4):
            #print(f"Col: {col}")
            # print('col: '+str(col)+' row: '+str(row))
            # print(models[models.model_id == model_ids[k]].bb_tkr)
            for date in dates:
                beta = getBetas(model_ids[k],betas_all,date)
                axs[row, col].plot(beta.return_lag,beta.qty)
            
            title = str(models[models.model_id == model_ids[k]].bb_tkr.values)[1:-1]
            # print(f"Title: {title}")
            axs[row, col].set_title(title)
            
            if k == len(model_ids)-1:
                break
            k =k+1
                    
    #Set labels
    for ax in axs.flat:
        ax.set(xlabel='Lag', ylabel='Beta')

    #set Ticks
    for ax in fig.get_axes():
        ax.label_outer()
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels=dates, bbox_to_anchor=(0.9, 0.15))
    # plt.savefig('C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\graphs_coefs_and_scores\\betas\\foo.png')
    plt.savefig(f"\\reports\\figures\\{model_type}.png") #'./reports/figures/'+

#%% main()
    engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
    #define Dates
    dates = ['2015-12-29','2019-12-31']
    
    df_model_type_id  = pd.read_sql_query("SELECT * FROM cftc.model_type_desc order by model_type_desc", engine1)
    df_model_type_id = df_model_type_id[df_model_type_id.alpha_type.isin(['gcv','loocv'])]

    for model_type in df_model_type_id.model_type_id:
        print(model_type)
        try:
            betas_all,models = getSpecificData(dates,model_type)
            createFigurePerModel(model_type,betas_all,dates)
        except:
            print(f"Error at: {model_type}")

    
    





# %%
