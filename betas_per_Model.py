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
    beta = betas_all[(betas_all.model_id == model_id) & (betas_all.px_date == date)]
    return beta[['return_lag','qty']]

#%%

# def createFigurePerModel(model_type,betas_all,dates,savefig = False):
#TODO: Adjust Y-Axis maybe
model_ids= list(betas_all.groupby('model_id').ranking.min().sort_values(ascending = True).index)
color = ['crimson', 'cyan'] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
#def Layout
fig, axs = plt.subplots(8, 3, sharex=False, sharey= False ,figsize=(15,20))
fig.tight_layout() # for nicer layout
fig.subplots_adjust(top=0.95)

sns.set(font_scale = 1.2)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})

fig.suptitle(f"All Betas with model_id: {model_type}",fontsize=30)

fig.text(0.5, 0.00, 'Return Lag', ha='center', fontsize = 20)
fig.text(0.00, 0.5, 'Beta', va='center', rotation='vertical', fontsize = 20)


plot_matrix = np.arange(24).reshape(8, -1)
for col in range(len(plot_matrix[0])):
    # print(f"Row: {row}")print(f"Row: {row}")
    for row in range(len(plot_matrix)):
        try:
            model_id = model_ids[plot_matrix[row][col]]
        except:
            break
        ax_curr = axs[row,col]
        # ax_curr.xaxis.label.set_visible(False)
        # ax_curr.set_xlabel()
        # print(f"Row: {row}")
        # print(f"col: {col}")
        
        k =0
        for date in dates:
            beta = getBetas2(model_id,betas_all,date)
            sns.lineplot(x = beta.return_lag,y = beta.qty ,ax =ax_curr, linewidth = 4, legend = False,color =color[k])
            k= k+1
            # plt.xticks([])
            # plt.yticks([])
        ax_curr.set_xlabel('')
        ax_curr.set_ylabel('')
        
        title = str(models[models.model_id == model_id].bb_tkr.values)[2:-2]
        ax_curr.set_title(title)

fig.delaxes(axs[7,2])

# handles, labels = ax.get_legend_handles_labels()
fig.legend(labels=dates, bbox_to_anchor=(0.9, 0.15),fontsize = 20,frameon = False)
plt.savefig(f"C:\\Users\\grbi\\PycharmProjects\\cftc_neu\\reports\\figures\Betas\{model_type}.png") #'./reports/figures/'+

# %%sample data
engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
#define Dates
dates = ['2015-12-29','2019-12-31']
model_type =40

df_model_type_id  = pd.read_sql_query("SELECT * FROM cftc.model_type_desc order by model_type_desc", engine1)

df_model_type_id = df_model_type_id[df_model_type_id.alpha_type.isin(['gcv','loocv'])]
lst_model_types = list(df_model_type_id.model_type_id)
betas_all,models = getSpecificData(dates,model_type= model_type)

#%% main()
engine1 = sq.create_engine("postgresql+psycopg2://grbi@iwa-backtest:grbizhaw@iwa-backtest.postgres.database.azure.com:5432/postgres")
#* define Variables
dates = ['2015-12-29','2019-12-31']
saveFig = True

df_model_type_id  = pd.read_sql_query("SELECT * FROM cftc.model_type_desc order by model_type_desc", engine1)

df_model_type_id = df_model_type_id[df_model_type_id.alpha_type.isin(['gcv','loocv'])]

for model_type in df_model_type_id.model_type_id[0:1]:
    print(model_type)
    try:
        betas_all,models = getSpecificData(dates,model_type)
        createFigurePerModel(model_type,betas_all,dates,savefig=saveFig)
    except:
        print(f"Error at: {model_type}")



# %%
