#%%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df_raw = pd.read_excel("C:/Users/grbi/switchdrive/Tracking Traders/04_Abbildungen/panel.xlsx",
                   sheet_name='Sheet1',skiprows=12)

df = df_raw[['lag','coef']]
df = df.iloc[:260,:]
#%%
fig, axs = plt.subplots(1,1, figsize=(10, 5))  # sharex=False, sharey= False
fig.tight_layout()
# fig.suptitle(f"All Betas with model_id: {model_type}",fontsize=30)
# fig.subplots_adjust(top=0.95)

sns.set(font_scale=1.2)
sns.set_style('white')
sns.set_style('white', {'font.family': 'serif', 'font.serif': 'Times New Roman'})

sns.lineplot(x=df.lag, y=df.coef, linewidth=2.5, ax=axs, legend=False, color='cyan')

xstart, xend = axs.get_xlim()

axs.set(xticks=np.arange(0,xend,25))
axs.set(yticks=np.arange(0,1,0.25))
axs.set_xlabel('')
axs.set_ylabel('')
plt.savefig(f"./temp/panel_regression.png", dpi=100)

plt.show()

#%%
np.arange(0,xend,25)

