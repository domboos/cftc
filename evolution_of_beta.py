#%%
from cfunctions import engine1,gets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
#TODO: define:

model_type_id = 77
model_id = 1923

#%%
models = pd.read_sql_query(f"SELECT model_id, bb_tkr from cftc.model_desc where model_type_id = {model_type_id}",engine1)
bb_tkrs = list(pd.read_sql_query(f"SELECT bb_tkr from cftc.order_of_things",engine1).bb_tkr)

print(models)


# %% #* Functions for Graphs:

def getMeanBeta(betas_per_model):
    mean_betas = betas_per_model.groupby('px_date').apply(lambda x: sum(x.qty * x.return_lag)/sum(x.qty))
    return mean_betas

def getExposureVsOI(betas_per_model,bb_tkr):

#%%
    a2 = betas_per_model.groupby('px_date').apply(lambda x: sum(x.qty * x.return_lag))
    oi = gets(engine1, type='agg_open_interest', data_tab='vw_data', desc_tab='cot_desc', bb_tkr=bb_tkr)
    return xy

#%%


for idx in model_types.index: #iterates through model_type_ids
    print(f"model_type : {idx}")
    temp = model_types.loc[idx,:].T
    
    ongoingQuery = pd.read_sql_query(f" Select * from cftc.model_desc where model_type_id = {int(idx)}", engine1).set_index('model_id')
    



#* get Evolution of Betas:
def getEvolutionofBetaPeak(betas_per_model):
    temp = betas_per_model.set_index('return_lag')
    n10 = temp.groupby('px_date').qty.nlargest(10)
    n10 = n10.reset_index()

    n10_lowerb  = n10.set_index('qty').groupby('px_date').return_lag.nsmallest(1).reset_index()
    n10_upperb = n10.set_index('qty').groupby('px_date').return_lag.nlargest(1).reset_index()
    n10_max = n10.set_index('return_lag').groupby('px_date').qty.nlargest(1).reset_index()
    return n10_lowerb, n10_max, n10_upperb



#%% do plots: 

plotEvolutionOfBetaPeak = True

model_type_id = 77

models = pd.read_sql_query(f"SELECT model_id, bb_tkr from cftc.model_desc where model_type_id = {model_type_id}",engine1)

fig, axs = plt.subplots(8, 3, sharex=False, sharey= False ,figsize=(15,20))
fig.tight_layout()
fig.suptitle(f"define title",fontsize=30)
fig.subplots_adjust(top=0.95)

sns.set(font_scale = 1.2)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})

# fig.suptitle(f"ACF model_id: {wantedModeltypeid}",fontsize=30)

if plotEvolutionOfBetaPeak:
    fig.text(0.5, 0.00, 'Date', ha='center', fontsize = 20)
    fig.text(0.00, 0.5, 'Return Lag', va='center', rotation='vertical', fontsize = 20)

plot_matrix = np.arange(24).reshape(8, -1)

for col in range(len(plot_matrix[0])):
    # print(f"Row: {row}")print(f"Row: {row}")
    for row in range(len(plot_matrix)):
        try:
            bb_tkr = bb_tkrs[plot_matrix[row][col]]
            print(bb_tkr)
        except:
            break
        
        ax_curr = axs[row,col]
        model_id = models[models.bb_tkr == bb_tkr].model_id.values[0]

        print(model_id)
        betas_per_model = pd.read_sql_query(f"SELECT * from cftc.beta where model_id = {model_id}",engine1)
        betas_per_model = betas_per_model.sort_values(['px_date','return_lag'], ascending = True)


        if plotEvolutionOfBetaPeak == True:
            n10_lowerb, n10_max, n10_upperb = getEvolutionofBetaPeak(betas_per_model)
            sns.lineplot(x = n10_lowerb.px_date,y = n10_lowerb.return_lag, ax=ax_curr, linewidth = 2, legend = False,color ='crimson')
            sns.lineplot(x = n10_max.px_date,y = n10_max.return_lag, ax=ax_curr, linewidth = 2, legend = False,color ='cyan')
            sns.lineplot(x = n10_upperb.px_date,y = n10_upperb.return_lag, ax=ax_curr, linewidth = 2, legend = False,color ='crimson')

            ax_curr.set_xlabel('')
            ax_curr.set_ylabel('')
            ax_curr.set_title(f"{bb_tkr}")

    # fig.delaxes(axs[7,2])
    if plotEvolutionOfBetaPeak:
        plt.savefig(f"reports/figures/Evo_Beta_Model_type_{model_type_id}_draft.png",dpi=100) #'./reports/figures/'+
plt.show()





# %%
sns.lineplot(x = n10_lowerb.px_date,y = n10_lowerb.return_lag,style = True, linewidth = 2, legend = False,dashes=[(2, 2)],color ='crimson')
# %%
