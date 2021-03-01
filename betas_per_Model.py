#%%
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlalchemy as sq
import seaborn as sns
from datetime import datetime
from cfunctions import engine1
from matplotlib.ticker import FormatStrFormatter

#TODO if you work in vsc, ctrl+shift+P --> rebuild and open in container Else Adjust Path in Safe Fig code Section
#TODO: Adjust Y-Axis in Plots (task for Mr. Boos)
#%% Functions:
def getSpecificData(dates,model_type):
    models = pd.read_sql_query(f"SELECT * FROM cftc.model_desc where model_type_id ={model_type}",engine1)
    oot = pd.read_sql_query("SELECT ranking,bb_tkr FROM cftc.order_of_things",engine1)
    models = pd.merge(models, oot,on = 'bb_tkr', how = 'left').sort_values('ranking')
    model_ids = list(models.model_id)
    temp_ids = len(model_ids)

    betas_all = pd.read_sql_query(f"SELECT * from cftc.beta where  px_date IN( {str(dates)[1:-1]} ) and model_id IN ( {str(model_ids)[1:-1]} )",engine1)
    betas_all.px_date = betas_all.px_date.astype('datetime64[ns]')
    betas_all.return_lag = betas_all.return_lag.astype('int')
    betas_all.qty = betas_all.qty.astype('float')
    betas_all = pd.merge(betas_all,models[['model_id','bb_tkr','ranking']]).sort_values('ranking')
    

    if temp_ids != len(model_ids):
        print(models[~models.model_id.isin(model_ids)].bb_tkr)
        print('^Do not exist in Betas')

    return betas_all,models

def getBetas2(model_id,betas_all, date):
    beta = betas_all[(betas_all.model_id == model_id) & (betas_all.px_date.isin(date))]
    return beta[['return_lag','qty']]

def createFigurePerModelwithDates(model_type,dates,savefig = False,pathSaveFig = "/home/jovyan/work/reports/figures/Betas/"):
    
    betas_all,models = getSpecificData(dates=dates,model_type = model_type)
    model_ids= list(models.model_id)
    color = ['crimson', 'cyan','blue','green','grey','black'] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    #def Layout
    fig, axs = plt.subplots(8, 3, sharex=False, sharey= False ,figsize=(15,20))
    fig.tight_layout()
    # fig.suptitle(f"All Betas with model_id: {model_type}",fontsize=30)
    # fig.subplots_adjust(top=0.95)
    
    sns.set(font_scale = 1.2)
    sns.set_style('white')
    sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})

    # fig.suptitle(f"All Betas with model_id: {model_type}",fontsize=30)

    # fig.text(0.5, 0.00, 'Return Lag', ha='center', fontsize = 20)
    # fig.text(0.00, 0.5, 'Beta', va='center', rotation='vertical', fontsize = 20)

    plot_matrix = np.arange(24).reshape(8, -1)
    for col in range(len(plot_matrix[0])):
        # print(f"Row: {row}")print(f"Row: {row}")
        for row in range(len(plot_matrix)):
            try:
                model_id = model_ids[plot_matrix[row][col]]
                print(model_id)
            except:
                break
            ax_curr = axs[row,col]

            
            k =0
            for date in dates:
                beta = getBetas2(model_id,betas_all,[date])
                sns.lineplot(x = beta.return_lag,y = beta.qty ,ax =ax_curr, linewidth = 3, legend = False,color =color[k])
                k= k+1
                # plt.xticks([])
                # plt.yticks([])
            ax_curr.set_xlabel('')
            ax_curr.set_ylabel('')
            
            title = str(models[models.model_id == model_id].bb_tkr.values)[2:-2]
            title1 = pd.read_sql_query(f"SELECT name FROM cftc.order_of_things where bb_tkr = '{title}'",engine1).name.values[0]
            ax_curr.set_title(title1)

    fig.delaxes(axs[7,2])
    # plt.gca().axes.get_yaxis().set_visible(False)
    
    
    # handles, labels = ax.get_legend_handles_labels()
    fig.legend(labels=dates, bbox_to_anchor=(0.9, 0.12),fontsize = 20,frameon = False)
    if savefig == True:
        plt.savefig(f"{pathSaveFig}/Betas_for_Model_{model_type}_to_decide.png",dpi=100) #'./reports/figures/'+
    
    plt.show()

def compare2Models(model_typeMM,model_typeNonc,labels,dates,savefig = False,pathSaveFig = "/home/jovyan/work/reports/figures/Betas/"):
    """
    Parameters:
    -----------

    model_typeNonc: int
    model_typeMM: int
    dates: str()
    savefig: boolean
    pathSaveFig: None
    """

    
    #get Betas and Modelids:    
    betas_NonC, models_NonC = getSpecificData(dates,model_typeNonc)
    betas_MM, models_MM = getSpecificData(dates,model_typeMM)

    model_idsNonc= list(betas_NonC.groupby('model_id').ranking.min().sort_values(ascending = True).index)
    model_idsMM= list(betas_MM.groupby('model_id').ranking.min().sort_values(ascending = True).index)
    
    
    #*define Layout
    color = ['crimson', 'cyan'] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    fig, axs = plt.subplots(8, 3, sharex=False, sharey= False ,figsize=(15,20))
    fig.tight_layout()
    # fig.suptitle(f"All Betas with model_id: {model_typeNonc},{model_typeMM}",fontsize=30)
    # fig.subplots_adjust(top=0.95)
    
    sns.set(font_scale = 1.2)
    sns.set_style('white')
    sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})


    # fig.text(0.5, 0.00, 'Return Lag', ha='center', fontsize = 20)
    # fig.text(0.00, 0.5, 'Beta', va='center', rotation='vertical', fontsize = 20)

    # DO Plots:
    plot_matrix = np.arange(24).reshape(8, -1)
    for col in range(len(plot_matrix[0])):
        # print(f"Row: {row}")print(f"Row: {row}")
        for row in range(len(plot_matrix)):
            try:
                model_id_nonc = model_idsNonc[plot_matrix[row][col]]
                model_id_mm = model_idsMM[plot_matrix[row][col]]
                
            except:
                break
            ax_curr = axs[row,col]
            
            betaNonC = getBetas2(model_id_nonc,betas_NonC,dates)
            betaMM = getBetas2(model_id_mm,betas_MM,dates)
            sns.lineplot(x = betaMM.return_lag,y = betaMM.qty ,ax =ax_curr, linewidth = 3, legend = False,color ='crimson')
            sns.lineplot(x = betaNonC.return_lag,y = betaNonC.qty ,ax =ax_curr, linewidth = 3, legend = False,color ='cyan')
            ax_curr.axhline(0, ls='--', color ='black')
            # ax_curr.yaxis.set_major_formatter(FormatStrFormatter('%e'))
            title = str(models_NonC[models_NonC.model_id == model_id_nonc].bb_tkr.values)[2:-2]
            title1 = title1 = pd.read_sql_query(f"SELECT name FROM cftc.order_of_things where bb_tkr = '{title}'",engine1).name.values[0]
            print(title1)
            ax_curr.set_title(title1)

            ax_curr.set_xlabel('')
            ax_curr.set_ylabel('')
            
            

    fig.delaxes(axs[7,2])
    # plt.gca().axes.get_yaxis().set_visible(False)
    
    
    # handles, labels = ax.get_legend_handles_labels() #first model is MM , second is NONC
    fig.legend(labels=labels, bbox_to_anchor=(0.98, 0.12),fontsize = 20,frameon = False)
    if savefig == True:
        plt.savefig(f"{pathSaveFig}/Beta_comparison_{model_typeNonc}-{model_typeMM}.png",dpi=100) #'./reports/figures/'+
    
    plt.show()




#%% #* Create Plot with Dates per Model ID
# dates = ['2015-12-29','2019-12-31']
dates = ['2002-12-31','2006-12-26','2010-12-28','2014-12-30','2018-12-25']
pathSaveFig = "reports/figures/Betas/"
for model_type in [82,76,95,100]:
    try:
        createFigurePerModelwithDates(model_type= model_type,dates= dates,savefig = True,pathSaveFig =pathSaveFig )
    except:
        continue

#%% #* Model Comparison: 100 - 95 // 82 - 76 // 100-82 // 95 -76
engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
#define Dates

df_model_type_id  = pd.read_sql_query("SELECT * FROM cftc.model_type_desc order by model_type_desc", engine1)
#%% # *  Sample Data for new plot: createFigure2
engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
#define Dates
dates = ['2019-12-31']

pathSaveFig = "/home/jovyan/work/reports/figures/Betas/"
model_typeMM = 100 #*Model2
model_typeNonc = 95  #*model1 

compare2Models(model_typeMM=model_typeMM,model_typeNonc=model_typeNonc ,labels= ['MM-square root','MM-flat'],dates=dates,savefig = True,pathSaveFig = "reports//figures//Betas//")

model_typeMM = 82 #*Model2
model_typeNonc = 76  #*model1 
compare2Models(model_typeMM=model_typeMM,model_typeNonc=model_typeNonc ,labels= ['NonC-square root','NonC-flat'],dates=dates,savefig = True,pathSaveFig = "reports//figures//Betas//")
#%%%
model_typeMM = 95 #*Model2
model_typeNonc = 76  #*model1 
compare2Models(model_typeMM=model_typeMM,model_typeNonc=model_typeNonc ,labels= ['NonC-square root','MM-flat'],dates=dates,savefig = True,pathSaveFig = "reports//figures//Betas//")

model_typeMM = 100 #*Model2
model_typeNonc = 82  #*model1 
compare2Models(model_typeMM=model_typeMM,model_typeNonc=model_typeNonc ,labels= ['MM-square root','NonC-square root'],dates=dates,savefig = True,pathSaveFig = "reports//figures//Betas//")

#%%
engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
#define Dates
dates = ['2019-12-31']
labels = [] #['window 280','window 260']

pathSaveFig = "/home/jovyan/work/reports/figures/Betas/"
model_typeMM = 100 #*Model2
model_typeNonc = 95  #*model1 

